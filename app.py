"""
HX-695A Predictive Maintenance Dashboard
------------------------------------------
Streamlit app that surfaces:
  - Current health snapshot (efficiency, fouling, failure risk)
  - Failure probability & thermal-efficiency trends over time
  - A "what-if" manual reading form for on-demand predictions
  - A maintenance-priority table (highest risk readings)

Run locally:
    streamlit run app.py

Deploy free on Streamlit Community Cloud:
    1. Push this folder to a public (or private) GitHub repo
    2. Go to https://share.streamlit.io -> "New app"
    3. Point it at the repo, branch, and app.py
    4. Deploy (build installs requirements.txt automatically)
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import joblib
import os

from train_models import (
    load_and_clean, engineer_features, train, clip_efficiency,
    CLASSIFIER_FEATURES, REGRESSOR_FEATURES, DATA_PATH,
)

# Bump this whenever train_models.py changes what it returns -- forces a
# retrain instead of silently loading an old cached model from disk.
ARTIFACT_VERSION = 2

st.set_page_config(
    page_title="HX-695A Predictive Maintenance",
    page_icon="🔧",
    layout="wide",
)

ARTIFACT_PATH = "model_artifacts.joblib"
DATA_CACHE_PATH = "processed_data.parquet"


# ---------------------------------------------------------------------
# Data & model loading (cached so the app is fast after first load)
# ---------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading data and models...")
def get_artifacts_and_data():
    cached = os.path.exists(ARTIFACT_PATH) and os.path.exists(DATA_CACHE_PATH)
    artifacts = joblib.load(ARTIFACT_PATH) if cached else None

    # Old cached artifacts (from before the fix) won't have "rf_reg" or a
    # version tag -- retrain from scratch instead of reusing the old model.
    stale = artifacts is None or artifacts.get("artifact_version") != ARTIFACT_VERSION

    if not stale:
        df = pd.read_parquet(DATA_CACHE_PATH)
    else:
        df = load_and_clean(DATA_PATH)
        df = engineer_features(df)
        artifacts = train(df)
        artifacts["artifact_version"] = ARTIFACT_VERSION
        joblib.dump(artifacts, ARTIFACT_PATH)
        df.to_parquet(DATA_CACHE_PATH, index=False)
    return artifacts, df


artifacts, df = get_artifacts_and_data()
log_model = artifacts["log_model"]
scaler = artifacts["scaler"]
rf_reg = artifacts["rf_reg"]
metrics = artifacts["metrics"]
reg_feature_ranges = artifacts["reg_feature_ranges"]

# Full-history predictions (for charts)
X_full = df[CLASSIFIER_FEATURES]
df["Failure_Probability"] = log_model.predict_proba(scaler.transform(X_full))[:, 1]
df["Predicted_Efficiency"] = [clip_efficiency(p) for p in rf_reg.predict(df[REGRESSOR_FEATURES])]


def risk_label(prob):
    if prob >= 0.6:
        return "🔴 High"
    elif prob >= 0.3:
        return "🟡 Medium"
    return "🟢 Low"


df["Risk_Level"] = df["Failure_Probability"].apply(risk_label)

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
st.sidebar.title("🔧 HX-695A")
st.sidebar.caption("Heat exchanger predictive maintenance")

page = st.sidebar.radio(
    "View",
    ["Overview", "Trends", "What-If Prediction", "Maintenance Priorities", "Model Performance"],
)

date_range = st.sidebar.slider(
    "Date range",
    min_value=df["Datetime"].min().to_pydatetime(),
    max_value=df["Datetime"].max().to_pydatetime(),
    value=(df["Datetime"].min().to_pydatetime(), df["Datetime"].max().to_pydatetime()),
)
mask = (df["Datetime"] >= date_range[0]) & (df["Datetime"] <= date_range[1])
view_df = df.loc[mask]

# ---------------------------------------------------------------------
# Overview page
# ---------------------------------------------------------------------
if page == "Overview":
    st.title("HX-695A Overview")
    latest = df.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Thermal Efficiency", f"{latest['Thermal_Efficiency_pct']:.1f}%",
               f"{latest['Efficiency_Change']:+.2f}")
    c2.metric("Fouling Factor", f"{latest['Fouling_Factor_m2K_W']:.2e} m²K/W")
    c3.metric("Failure Probability (7d)", f"{latest['Failure_Probability']*100:.1f}%")
    c4.metric("Days Since Cleaning", f"{latest['Days_Since_Cleaning']:.0f}")

    st.markdown(f"### Current Risk: {latest['Risk_Level']}")
    if latest["Failure_Probability"] >= 0.6:
        st.error("High risk of failure within 7 days — schedule cleaning/maintenance soon.")
    elif latest["Failure_Probability"] >= 0.3:
        st.warning("Medium risk — monitor closely over the next few days.")
    else:
        st.success("Low risk — normal operation.")

    st.divider()
    col1, col2 = st.columns(2)

    with col1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view_df["Datetime"], y=view_df["Thermal_Efficiency_pct"],
                                  name="Efficiency", line=dict(width=1, color="steelblue")))
        fig.update_layout(title="Thermal Efficiency Over Time", yaxis_title="Efficiency %",
                           height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width='stretch')

    with col2:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=view_df["Datetime"], y=view_df["Fouling_Factor_m2K_W"],
                                  name="Fouling", line=dict(width=1, color="firebrick")))
        fig.update_layout(title="Fouling Factor Over Time", yaxis_title="m²K/W",
                           height=350, margin=dict(t=40, b=20))
        st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------
# Trends page
# ---------------------------------------------------------------------
elif page == "Trends":
    st.title("Failure Probability & Efficiency Trends")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=view_df["Datetime"], y=view_df["Failure_Probability"],
                              name="Failure Probability", line=dict(color="darkred", width=1)))
    fig.add_hline(y=0.3, line_dash="dash", line_color="orange", annotation_text="Medium threshold")
    fig.add_hline(y=0.6, line_dash="dash", line_color="red", annotation_text="High threshold")
    fig.update_layout(title="Predicted Failure Probability Over Time", yaxis_title="Probability",
                       height=400)
    st.plotly_chart(fig, width='stretch')

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=view_df["Datetime"], y=view_df["Thermal_Efficiency_pct"],
                               name="Actual", line=dict(width=1, color="steelblue")))
    fig2.add_trace(go.Scatter(x=view_df["Datetime"], y=view_df["Predicted_Efficiency"],
                               name="Predicted", line=dict(width=1, color="orange", dash="dot")))
    fig2.update_layout(title="Actual vs. Predicted Thermal Efficiency", yaxis_title="Efficiency %",
                        height=400)
    st.plotly_chart(fig2, width='stretch')

    st.subheader("Efficiency by Shift")
    shift_avg = view_df.groupby("Shift")["Thermal_Efficiency_pct"].mean().reset_index()
    fig3 = go.Figure(go.Bar(x=shift_avg["Shift"], y=shift_avg["Thermal_Efficiency_pct"]))
    fig3.update_layout(height=300, yaxis_title="Avg Efficiency %")
    st.plotly_chart(fig3, width='stretch')

# ---------------------------------------------------------------------
# What-If Prediction page
# ---------------------------------------------------------------------
elif page == "What-If Prediction":
    st.title("What-If Prediction")
    st.caption("Enter a hypothetical reading to see predicted failure risk and efficiency.")

    last = df.iloc[-1]
    with st.form("whatif"):
        col1, col2, col3 = st.columns(3)
        with col1:
            hot_in = st.number_input("Hot Inlet Temp (°C)", value=float(last["Hot_Inlet_Temp_C"]))
            hot_out = st.number_input("Hot Outlet Temp (°C)", value=float(last["Hot_Outlet_Temp_C"]))
            cold_in = st.number_input("Cold Inlet Temp (°C)", value=float(last["Cold_Inlet_Temp_C"]))
            cold_out = st.number_input("Cold Outlet Temp (°C)", value=float(last["Cold_Outlet_Temp_C"]))
        with col2:
            fouling = st.number_input("Fouling Factor (m²K/W)", value=float(last["Fouling_Factor_m2K_W"]), format="%.6f")
            u_value = st.number_input("U Value (W/m²K)", value=float(last["U_Value_W_m2K"]))
            effectiveness = st.number_input("Effectiveness (%)", value=float(last["Effectiveness_pct"]))
            days_since_cleaning = st.number_input("Days Since Cleaning", value=float(last["Days_Since_Cleaning"]))
        with col3:
            pdrop_tube = st.number_input("Pressure Drop Tube (bar)", value=float(last["Pressure_Drop_Tube_bar"]))
            pdrop_shell = st.number_input("Pressure Drop Shell (bar)", value=float(last["Pressure_Drop_Shell_bar"]))
            alarms = st.number_input("Alarm Count", value=float(last["Alarm_Count"]))
            ambient = st.number_input("Ambient Temp (°C)", value=float(last["Ambient_Temp_C"]))

        submitted = st.form_submit_button("Predict")

    if submitted:
        row = last.copy()
        row["Hot_Inlet_Temp_C"] = hot_in
        row["Hot_Outlet_Temp_C"] = hot_out
        row["Cold_Inlet_Temp_C"] = cold_in
        row["Cold_Outlet_Temp_C"] = cold_out
        row["Fouling_Factor_m2K_W"] = fouling
        row["U_Value_W_m2K"] = u_value
        row["Effectiveness_pct"] = effectiveness
        row["Days_Since_Cleaning"] = days_since_cleaning
        row["Pressure_Drop_Tube_bar"] = pdrop_tube
        row["Pressure_Drop_Shell_bar"] = pdrop_shell
        row["Alarm_Count"] = alarms
        row["Ambient_Temp_C"] = ambient

        # --- Recompute engineered/derived features so they stay consistent
        # with the manually-entered values above. Without this, the model
        # sees a hand-picked fouling/pressure reading paired with a *stale*
        # rolling average and change-rate from the last real row, which is
        # an out-of-distribution combination the Linear Regression was never
        # trained on -- that's what produced impossible >100% outputs.
        row["Fouling_Change"] = fouling - last["Fouling_Factor_m2K_W"]
        row["PressureDrop_Change"] = pdrop_tube - last["Pressure_Drop_Tube_bar"]
        row["Fouling_Roll72h"] = fouling
        row["PressureDrop_Roll72h"] = pdrop_tube
        row["Days_x_Ambient"] = days_since_cleaning * ambient

        X_clf = pd.DataFrame([row[CLASSIFIER_FEATURES]])
        X_reg = pd.DataFrame([row[REGRESSOR_FEATURES]])

        prob = log_model.predict_proba(scaler.transform(X_clf))[0, 1]
        # Random Forest can only predict values it saw during training, so it
        # can't blow past a realistic range the way Linear Regression could --
        # the clip below is just a belt-and-suspenders backup, not doing the
        # real work anymore.
        pred_eff = clip_efficiency(rf_reg.predict(X_reg)[0])

        # Still worth telling the user if they typed in a value the model has
        # never actually seen -- the prediction is a reasonable estimate, but
        # not something to bet a maintenance decision on.
        out_of_range = [
            col for col in REGRESSOR_FEATURES
            if row[col] < reg_feature_ranges[col][0] or row[col] > reg_feature_ranges[col][1]
        ]

        st.divider()
        c1, c2 = st.columns(2)
        c1.metric("Predicted Failure Probability (7 days)", f"{prob*100:.1f}%")
        c2.metric("Predicted Thermal Efficiency", f"{pred_eff:.1f}%")
        if out_of_range:
            st.caption(
                f"⚠️ These inputs are outside the range seen in training data: "
                f"{', '.join(out_of_range)}. The prediction is a rough estimate, "
                "not something to rely on for a maintenance decision."
            )
        st.markdown(f"**Risk Level:** {risk_label(prob)}")

# ---------------------------------------------------------------------
# Maintenance Priorities page
# ---------------------------------------------------------------------
elif page == "Maintenance Priorities":
    st.title("Maintenance Priorities")
    st.caption("Readings sorted by predicted failure probability, highest first.")

    top_risk = view_df.sort_values("Failure_Probability", ascending=False).head(25)
    st.dataframe(
        top_risk[[
            "Datetime", "Shift", "Operator_ID", "Thermal_Efficiency_pct",
            "Fouling_Factor_m2K_W", "Days_Since_Cleaning", "Failure_Probability", "Risk_Level"
        ]].reset_index(drop=True),
        width='stretch',
    )

    st.subheader("Risk level distribution")
    dist = view_df["Risk_Level"].value_counts().reset_index()
    dist.columns = ["Risk_Level", "Count"]
    fig = go.Figure(go.Bar(x=dist["Risk_Level"], y=dist["Count"]))
    fig.update_layout(height=300)
    st.plotly_chart(fig, width='stretch')

# ---------------------------------------------------------------------
# Model Performance page
# ---------------------------------------------------------------------
elif page == "Model Performance":
    st.title("Model Performance")

    st.subheader("Classifier — Failure Within 7 Days (Logistic Regression)")
    c1, c2 = st.columns(2)
    c1.metric("Accuracy", f"{metrics['classifier_accuracy']*100:.1f}%")
    report = metrics["classifier_report"]
    c2.metric("Recall (catches real failures)", f"{report['1']['recall']*100:.1f}%")
    st.json(report)

    st.subheader("Regressor — Thermal Efficiency (Random Forest)")
    st.caption(
        "Switched from Linear Regression: it could extrapolate past 100% "
        "efficiency on unusual inputs. Random Forest can't do that, and "
        "gets essentially the same accuracy."
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{metrics['regressor_mae']:.2f}")
    c2.metric("RMSE", f"{metrics['regressor_rmse']:.2f}")
    c3.metric("Test R²", f"{metrics['regressor_r2']:.3f}")
    c4.metric("Train R²", f"{metrics['regressor_train_r2']:.3f}",
              help="Close to Test R² means the model isn't overfitting.")

st.sidebar.divider()
st.sidebar.caption("Data: HX695A_Dataset.csv · Models retrain automatically if artifacts are missing.")
