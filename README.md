# Heat Exchanger Fouling Prediction using Machine Learning

A Chemical Engineering + Machine Learning project that predicts the thermal efficiency of a shell-and-tube heat exchanger and provides a failure risk assessment using operating conditions.

The project was developed using Python, Scikit-Learn, and Streamlit as part of an academic project.

---

## Project Overview

Heat exchanger fouling gradually reduces heat transfer performance, increases energy consumption, and may lead to unexpected shutdowns.

This project uses historical operating data to:

- Predict Thermal Efficiency
- Estimate Failure Risk
- Perform What-If Analysis
- Compare Multiple ML Models
- Visualize Feature Importance
- Deploy an interactive dashboard using Streamlit

---

## Features

- Data preprocessing
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Model Training
- Linear Regression
- Random Forest Regression
- Logistic Regression
- Model Comparison
- Safe Prediction Function
- Streamlit Dashboard

---

## Machine Learning Models

| Model | Purpose |
|--------|----------|
| Random Forest Regressor | Predict Thermal Efficiency |
| Logistic Regression | Predict Failure Risk |
| Linear Regression | Baseline model for comparison |

Random Forest was selected as the deployment model because it provides realistic predictions even for unseen operating conditions.

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Plotly
- Streamlit

---

## Dataset

The dataset contains operating parameters of a shell-and-tube heat exchanger including:

- Inlet Temperature
- Outlet Temperature
- Mass Flow Rate
- Pressure Drop
- Fouling Factor
- Operating Hours
- Heat Duty
- Thermal Efficiency

---

## Project Structure

```
heat-exchanger-fouling-prediction/
│
├── app.py
├── HX-695A.ipynb
├── HX695A_Dataset.csv
├── requirements.txt
├── models/
│   ├── random_forest.pkl
│   └── logistic_model.pkl
├── images/
├── README.md
└── LICENSE
```

---

## Dashboard

The Streamlit dashboard allows users to:

- Enter operating conditions
- Predict thermal efficiency
- Estimate failure risk
- Perform What-If Analysis
- View model predictions instantly

---

## Results

- Random Forest achieved the best balance between accuracy and robustness.
- Linear Regression was retained only for model interpretation.
- Logistic Regression was used for binary failure prediction.

---

## Future Improvements

- Real plant data integration
- Live sensor monitoring
- Time-series prediction
- Remaining Useful Life (RUL) estimation
- Cloud deployment

---

## Author

**Muhammad Faishal Araman**

Major in Chemical Engineering| Minor in Management

National Institute of Technology Warangal

---
