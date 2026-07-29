"""
train_models.py
----------------
Data loading, feature engineering, and model training for the HX-695A
predictive maintenance app.

Two models:
  - log_model : Logistic Regression -> Failure_Within_7Days (probability, always 0-100%)
  - rf_reg    : Random Forest Regressor -> Thermal_Efficiency_pct

NOTE: we used to use Linear Regression for the efficiency prediction, but a
straight-line model has no idea 100% is a ceiling. Feed it an out-of-range
reading (e.g. a very overdue cleaning) and it just keeps extrapolating --
we saw it output 500%+ in testing. Random Forest can only predict values it
saw (or averages of values it saw) during training, so it can't blow past a
realistic range even on weird inputs. Accuracy is basically the same
(R^2 ~0.90 either way), so there's no trade-off, just a safer model.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)

DATA_PATH = "HX695A_Dataset.csv"
ROLL_WINDOW = 24  # hours

CLASSIFIER_FEATURES = [
    'Hot_Inlet_Temp_C', 'Hot_Outlet_Temp_C', 'Cold_Inlet_Temp_C', 'Cold_Outlet_Temp_C',
    'Fouling_Factor_m2K_W', 'U_Value_W_m2K', 'Thermal_Efficiency_pct', 'Effectiveness_pct',
    'Pressure_Drop_Tube_bar', 'Pressure_Drop_Shell_bar', 'Operating_Hours', 'Days_Since_Cleaning',
    'Alarm_Count', 'Equipment_Health_Score', 'Ambient_Temp_C',
    'Efficiency_Roll72h', 'Fouling_Roll72h', 'PressureDrop_Roll72h',
    'Efficiency_Change', 'Fouling_Change', 'PressureDrop_Change', 'Efficiency_Slope72h',
    'Month_sin', 'Month_cos', 'Days_x_Ambient',
]

REGRESSOR_FEATURES = [
    'Hot_Inlet_Temp_C', 'Cold_Inlet_Temp_C', 'Cold_Outlet_Temp_C',
    'U_Value_W_m2K', 'Effectiveness_pct',
    'Pressure_Drop_Tube_bar', 'Pressure_Drop_Shell_bar', 'Operating_Hours', 'Days_Since_Cleaning',
    'Alarm_Count', 'Ambient_Temp_C',
    'Fouling_Roll72h', 'PressureDrop_Roll72h',
    'Fouling_Change', 'PressureDrop_Change',
    'Month_sin', 'Month_cos', 'Days_x_Ambient',
]


def load_and_clean(path=DATA_PATH):
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])
    df = df.sort_values("Datetime").reset_index(drop=True)
    df = df.drop_duplicates(subset="Record_ID", keep="first").reset_index(drop=True)
    df["Sensor_Fault_Flag"] = (df["Sensor_Status"] == "Fault").astype(int)

    sensor_cols = [
        "Hot_Inlet_Temp_C", "Hot_Outlet_Temp_C", "Cold_Flow_Rate_kg_hr", "Heat_Duty_kW",
        "Fouling_Factor_m2K_W", "Equipment_Health_Score", "Electricity_Consumption_kWh",
        "U_Value_W_m2K",
    ]
    df[sensor_cols] = df[sensor_cols].interpolate(method="linear", limit_direction="both")
    return df


def engineer_features(df, window=ROLL_WINDOW):
    df = df.copy()
    df["Efficiency_Roll72h"] = df["Thermal_Efficiency_pct"].rolling(window, min_periods=1).mean()
    df["Fouling_Roll72h"] = df["Fouling_Factor_m2K_W"].rolling(window, min_periods=1).mean()
    df["PressureDrop_Roll72h"] = df["Pressure_Drop_Tube_bar"].rolling(window, min_periods=1).mean()

    df["Efficiency_Change"] = df["Thermal_Efficiency_pct"].diff()
    df["Fouling_Change"] = df["Fouling_Factor_m2K_W"].diff()
    df["PressureDrop_Change"] = df["Pressure_Drop_Tube_bar"].diff()
    df["Efficiency_Slope72h"] = df["Thermal_Efficiency_pct"].diff(window) / window

    df["Month"] = df["Datetime"].dt.month
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["Days_x_Ambient"] = df["Days_Since_Cleaning"] * df["Ambient_Temp_C"]

    df = df.iloc[window:].reset_index(drop=True)
    return df


def clip_efficiency(value):
    """Efficiency physically can't be below 0% or above 100% -- extra safety net
    on top of the Random Forest already being bounded to realistic values."""
    return float(np.clip(value, 0, 100))


def train(df, test_frac=0.2):
    split_idx = int(len(df) * (1 - test_frac))

    # ---- Classifier: Logistic Regression -> Failure_Within_7Days ----
    X = df[CLASSIFIER_FEATURES]
    y = df["Failure_Within_7Days"]
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    log_model.fit(X_train_scaled, y_train)
    y_pred_clf = log_model.predict(X_test_scaled)

    classifier_accuracy = accuracy_score(y_test, y_pred_clf)
    classifier_report = classification_report(y_test, y_pred_clf, output_dict=True)

    # ---- Regressor: Random Forest -> Thermal_Efficiency_pct ----
    # max_depth=6 was picked after checking a few depths -- it gave the best
    # train/test balance (deeper trees fit training data too closely).
    X_reg = df[REGRESSOR_FEATURES]
    y_reg = df["Thermal_Efficiency_pct"]
    X_reg_train, X_reg_test = X_reg.iloc[:split_idx], X_reg.iloc[split_idx:]
    y_reg_train, y_reg_test = y_reg.iloc[:split_idx], y_reg.iloc[split_idx:]

    rf_reg = RandomForestRegressor(
        n_estimators=300, max_depth=6, min_samples_leaf=5, random_state=42
    )
    rf_reg.fit(X_reg_train, y_reg_train)
    y_pred_reg = rf_reg.predict(X_reg_test)

    regressor_mae = mean_absolute_error(y_reg_test, y_pred_reg)
    regressor_rmse = np.sqrt(mean_squared_error(y_reg_test, y_pred_reg))
    regressor_r2 = r2_score(y_reg_test, y_pred_reg)
    regressor_train_r2 = r2_score(y_reg_train, rf_reg.predict(X_reg_train))

    # Training range per feature, so the app can flag out-of-range What-If inputs
    # instead of silently returning an unreliable prediction.
    reg_feature_ranges = {
        col: (float(X_reg_train[col].min()), float(X_reg_train[col].max()))
        for col in REGRESSOR_FEATURES
    }

    metrics = {
        "classifier_accuracy": classifier_accuracy,
        "classifier_report": classifier_report,
        "regressor_mae": regressor_mae,
        "regressor_rmse": regressor_rmse,
        "regressor_r2": regressor_r2,
        "regressor_train_r2": regressor_train_r2,
    }

    return {
        "log_model": log_model,
        "scaler": scaler,
        "rf_reg": rf_reg,
        "metrics": metrics,
        "reg_feature_ranges": reg_feature_ranges,
    }
