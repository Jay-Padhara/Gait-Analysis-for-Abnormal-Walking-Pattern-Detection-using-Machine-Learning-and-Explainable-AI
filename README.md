# Explainable Anomaly Detection on Multivariate Time-Series Data

## Project Overview

This project presents a supervised explainable anomaly detection system for multivariate gait time-series data using Random Forest and SHAP.

The system detects abnormal gait patterns and explains predictions using Explainable AI (XAI) techniques.

---

## Dataset

Dataset:  
UCI Multivariate Gait Dataset

Link:  
https://archive.ics.uci.edu/dataset/760/multivariate+gait+data

Total Samples:  
181800

Dataset Attributes:

- Subject
- Condition
- Replication
- Leg
- Joint
- Time
- Angle

---

## Technologies Used

- Python
- Pandas
- NumPy
- Streamlit
- Plotly
- Scikit-learn
- SHAP
- Matplotlib

---

## Methodology

### 1. Data Loading

Load multivariate gait dataset.

### 2. Data Preprocessing

- Missing value checking
- Time sorting
- Data filtering
- Label preparation

### 3. Feature Expansion

Categorical features:

- Subject Code
- Condition Code
- Leg Code
- Joint Code

### 4. Feature Engineering

Generated time-series features:

- Angle
- Velocity
- Acceleration
- Rolling Mean
- Rolling Standard Deviation
- Rolling Range

### 5. Supervised Learning

Model:

Random Forest Classifier

### 6. Anomaly Detection

Random Forest predicts normal and anomalous gait patterns.

### 7. Explainable AI

SHAP (SHapley Additive exPlanations)

### 8. Interactive Dashboard

Visual analytics dashboard for anomaly exploration.

---

## Dashboard Features

- Subject selection
- Joint selection
- Condition selection
- Replication selection
- Leg selection
- Interactive gait visualization
- Prediction probability visualization
- Severe anomaly table
- SHAP explainability
- CSV export of anomalies

---

## Results

Model Accuracy:

96.62%

Detected Anomalies:

4 anomalies detected in selected gait sequence

Most Important Features:

1. Rolling Mean
2. Rolling Range
3. Rolling Standard Deviation
4. Velocity
5. Acceleration

---

## Explainability

SHAP explains how each feature contributes to anomaly prediction.

Feature importance interpretation:

- Positive SHAP value → increases anomaly probability
- Negative SHAP value → decreases anomaly probability
- Red color → high feature value
- Blue color → low feature value

---

## Research Papers

Paper 1:

https://www.mdpi.com/2227-7390/14/2/230

Paper 2:

https://arxiv.org/pdf/2509.16472

---

## Run Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run dashboard:

```bash
streamlit run src/app.py
```

---

## Project Structure

Explainable_Anomaly_Detection/

├── data/  
│ └── gait.csv

├── notebooks/  
│ └── data_analysis.ipynb

├── results/  
│ └── screenshots/

├── src/  
│ └── app.py

├── requirements.txt

└── README.md

---

## Authors

- Jay Padhara
- Ravi Gangani
- Hitesh Ramani

---

Visual Analytics Project

Supervised Explainable Gait Anomaly Detection using Random Forest and SHAP
