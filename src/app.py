import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

st.set_page_config(page_title="Gait Analytics Dashboard", layout="wide")

st.markdown("""
<style>

.main{
    background:#0e1117;
}

h1,h2,h3{
    color:#00d4ff;
}

/* KPI Cards */

[data-testid="stMetric"]{
    background:linear-gradient(135deg,#0f172a,#1e293b);
    padding:20px;
    border-radius:18px;
    border:1px solid #334155;
    box-shadow:0 4px 20px rgba(0,0,0,0.35);
}

/* Metric Label */

[data-testid="stMetricLabel"]{
    color:#cbd5e1 !important;
    font-size:16px !important;
    font-weight:600 !important;
}

/* Metric Number */

[data-testid="stMetricValue"]{
    color:white !important;
    font-size:42px !important;
    font-weight:700 !important;
}

/* Metric Delta */

[data-testid="stMetricDelta"]{
    color:#22c55e !important;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='padding:25px;border-radius:20px;
background:linear-gradient(90deg,#0f172a,#1e293b);color:white;text-align:center;'>
<h1>🚀 Explainable Gait Anomaly Detection</h1>
<p>Random Forest • Logistic Regression • SHAP • Visual Analytics</p>
</div>
""", unsafe_allow_html=True)

df = pd.read_csv("../data/gait.csv")

st.sidebar.title("Filters")

st.sidebar.success(
"""
✅ Random Forest

✅ SHAP Enabled

✅ Window Size = 10

✅ Explainable AI
"""
)

subject = st.sidebar.selectbox("Subject", sorted(df["subject"].unique()))
joint = st.sidebar.selectbox("Joint", sorted(df["joint"].unique()))
condition = st.sidebar.selectbox("Condition", sorted(df["condition"].unique()))
replication = st.sidebar.selectbox("Replication", sorted(df["replication"].unique()))
leg = st.sidebar.selectbox("Leg", sorted(df["leg"].unique()))

sample = df[
(df["subject"] == subject) &
(df["joint"] == joint) &
(df["condition"] == condition) &
(df["replication"] == replication) &
(df["leg"] == leg)
].copy()

sample = sample.sort_values("time").reset_index(drop=True)

window_size = 10
sample["velocity"] = sample["angle"].diff()
sample["acceleration"] = sample["velocity"].diff()
sample["rolling_mean"] = sample["angle"].rolling(window=window_size, center=True).mean()
sample["rolling_std"] = sample["angle"].rolling(window=window_size, center=True).std()
sample["rolling_range"] = sample["angle"].rolling(window=window_size, center=True).max() - sample["angle"].rolling(window=window_size, center=True).min()
sample = sample.dropna()

sample["label"] = (sample["angle"] > sample["angle"].quantile(0.95)).astype(int)

features = ["velocity","acceleration","rolling_mean","rolling_std","rolling_range"]

X = sample[features]
y = sample["label"]

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42,stratify=y
)

model = RandomForestClassifier(n_estimators=100,random_state=42)
model.fit(X_train,y_train)


lr_model = LogisticRegression(
    max_iter=1000
)

lr_model.fit(
    X_train,
    y_train
)

lr_pred = lr_model.predict(
    X_test
)

lr_acc = round(
    accuracy_score(
        y_test,
        lr_pred
    ) * 100,
    2
)

# =====================================
# MODEL EVALUATION
# =====================================

pred = model.predict(
    X_test
)

proba = model.predict_proba(
    X_test
)[:,1]

rf_acc = round(
    accuracy_score(
        y_test,
        pred
    ) * 100,
    2
)

# RANDOM FOREST METRICS

rf_precision = round(
    precision_score(
        y_test,
        pred,
        zero_division=0
    ),
    3
)

rf_recall = round(
    recall_score(
        y_test,
        pred,
        zero_division=0
    ),
    3
)

rf_f1 = round(
    f1_score(
        y_test,
        pred,
        zero_division=0
    ),
    3
)

# LOGISTIC REGRESSION METRICS

lr_precision = round(
    precision_score(
        y_test,
        lr_pred,
        zero_division=0
    ),
    3
)

lr_recall = round(
    recall_score(
        y_test,
        lr_pred,
        zero_division=0
    ),
    3
)

lr_f1 = round(
    f1_score(
        y_test,
        lr_pred,
        zero_division=0
    ),
    3
)
# FULL DATA PREDICTIONS

sample["anomaly"] = model.predict(X)

sample["prediction_probability"] = (
    model.predict_proba(X)[:,1]
)

anomalies = sample[
    sample["anomaly"] == 1
]

# CLASSIFICATION REPORT

report_df = pd.DataFrame(
    classification_report(
        y_test,
        pred,
        zero_division=0,
        output_dict=True
    )
).transpose()

# =====================================
# SIDEBAR NAVIGATION
# =====================================

page = st.sidebar.radio(
    "🚀 Navigation",
    [
        "🏠 Overview",
        "🚨 Detection",
        "🧠 Explainability",
        "📈 Evaluation",
        "📋 Report"
    ]
)

# =====================================
# OVERVIEW
# =====================================

if page == "🏠 Overview":

    st.markdown("## 📌 Executive Summary")

    left,right = st.columns(2)

    with left:

        st.info(f"""
        **Selected Subject:** {subject}

        **Total Samples:** {len(sample)}

        **Detected Anomalies:** {len(anomalies)}

        **Window Size:** {window_size}
        """)

    with right:

        st.success(f"""
        **Models:** Random Forest + Logistic Regression

        **Explainability:** SHAP

        **RF Accuracy:** {rf_acc}%

        **LR Accuracy:** {lr_acc}%

        **Features Used:** {len(features)}
        """)

    # =====================================
    # KPI CARDS
    # =====================================

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "📊 Samples",
        f"{len(sample):,}"
    )

    c2.metric(
        "🚨 Anomalies",
        f"{len(anomalies):,}"
    )

    c3.metric(
        "⚙️ Window Size",
        str(window_size)
    )

    c4.metric(
        "👤 Subject",
        str(subject)
    )

    # =====================================
    # ADVANCED METRICS
    # =====================================

    top_feature = max(
        zip(features, model.feature_importances_),
        key=lambda x:x[1]
    )[0]

    d1,d2,d3,d4 = st.columns(4)

    d1.metric(
        "🎯 Risk Score",
        f"{round(sample['prediction_probability'].mean()*100,2)}%"
    )

    d2.metric(
        "🧠 Top Feature",
        top_feature
    )

    d3.metric(
        "🌲 RF Accuracy",
        f"{rf_acc}%"
    )

    d4.metric(
        "📈 LR Accuracy",
        f"{lr_acc}%"
    )

    st.markdown("---")

    # =====================================
    # RISK GAUGE
    # =====================================

    st.subheader(
        "🎯 Anomaly Risk Score"
    )

    risk_score = round(
        sample["prediction_probability"].mean()*100,
        2
    )

    gauge = go.Figure(
        go.Indicator(

            mode="gauge+number",

            value=risk_score,

            number={
                "suffix":"%",
                "font":{
                    "size":40,
                    "color":"white"
                }
            },

            title={
                "text":"Current Risk Level",
                "font":{
                    "size":22
                }
            },

            gauge={

                "axis":{
                    "range":[0,100]
                },

                "bar":{
                    "color":"#00d4ff"
                },

                "steps":[

                    {
                        "range":[0,40],
                        "color":"#16a34a"
                    },

                    {
                        "range":[40,70],
                        "color":"#f59e0b"
                    },

                    {
                        "range":[70,100],
                        "color":"#dc2626"
                    }

                ]
            }
        )
    )

    gauge.update_layout(
        template="plotly_dark",
        height=400
    )

    st.plotly_chart(
        gauge,
        use_container_width=True
    )

    st.markdown("---")

    # =====================================
    # DATASET PREVIEW
    # =====================================

    st.subheader(
        "🗄 Dataset Preview"
    )

    st.dataframe(
        sample.head(10),
        use_container_width=True,
        height=350
    )

    st.markdown("---")

    # =====================================
    # MEAN GAIT TRAJECTORY
    # =====================================

    st.subheader(
        "📈 Mean Gait Trajectory Across Subjects"
    )

    subject_mean = (

        df.groupby(
            ["subject","time"]
        )["angle"]

        .mean()

        .reset_index()

    )

    fig = px.line(

        subject_mean,

        x="time",

        y="angle",

        color="subject",

        template="plotly_dark",

        title="Mean Gait Trajectory Across Subjects"

    )

    fig.update_layout(

        height=650,

        title_x=0.5,

        legend_title="Subject",

        paper_bgcolor="#0e1117",

        plot_bgcolor="#0e1117",

        hovermode="x unified",

        xaxis_title="Time",

        yaxis_title="Joint Angle"

    )

    fig.update_traces(
        line=dict(width=3)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# =====================================
# DETECTION
# =====================================

elif page == "🚨 Detection":

    st.subheader(
        "🚨 Anomaly Detection Dashboard"
    )

    # KPI ROW

    c1,c2,c3 = st.columns(3)

    c1.metric(
        "Total Samples",
        len(sample)
    )

    c2.metric(
        "Detected Anomalies",
        len(anomalies)
    )

    c3.metric(
        "Anomaly Rate",
        f"{round(len(anomalies)/len(sample)*100,2)}%"
    )

    st.markdown("---")

    # GAIT SIGNAL

    fig = px.line(

        sample,

        x="time",

        y="angle",

        title="Gait Signal with Detected Anomalies",

        template="plotly_dark"

    )

    fig.update_traces(
        line=dict(
            width=3,
            color="#00d4ff"
        )
    )

    fig.add_scatter(

        x=anomalies["time"],

        y=anomalies["angle"],

        mode="markers",

        marker=dict(

            size=18,

            color="red",

            symbol="diamond",

            line=dict(
                width=2,
                color="white"
            )

        ),

        name="Detected Anomalies"

    )

    fig.update_layout(

        height=600,

        title_x=0.5,

        paper_bgcolor="#0e1117",

        plot_bgcolor="#0e1117"

    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # PROBABILITY CHART

    st.subheader(
        "📈 Prediction Probability"
    )

    fig2 = px.line(

        sample,

        x="time",

        y="prediction_probability",

        template="plotly_dark"

    )

    fig2.add_hline(

        y=0.5,

        line_dash="dash",

        annotation_text="Decision Threshold"

    )

    fig2.update_layout(

        height=500,

        title="Anomaly Probability Over Time",

        title_x=0.5

    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    # DISTRIBUTION

    st.subheader(
        "📊 Anomaly Distribution"
    )

    pie = px.pie(

        names=["Normal","Anomaly"],

        values=[
            len(sample)-len(anomalies),
            len(anomalies)
        ],

        color=["Normal","Anomaly"],

        color_discrete_map={
            "Normal":"#00d4ff",
            "Anomaly":"#ff4b4b"
        },

        hole=0.45,

        title="Normal vs Anomaly Samples"

    )

    pie.update_layout(
        height=450
    )

    st.plotly_chart(
        pie,
        use_container_width=True
    )

    # TABLE

    st.subheader(
        "📋 Most Severe Anomalies"
    )

    st.dataframe(

        anomalies.sort_values(

            by="prediction_probability",

            ascending=False

        ),

        use_container_width=True

    )


# =====================================
# EXPLAINABILITY
# =====================================

elif page == "🧠 Explainability":

    imp = pd.DataFrame({

        "Feature": features,

        "Importance": model.feature_importances_

    })

    imp = imp.sort_values(

        by="Importance",

        ascending=False

    )

    top_feature = imp.iloc[0]["Feature"]

    st.metric(
        "🏆 Top Feature",
        top_feature
    )

    col1,col2 = st.columns([1,2])

    with col1:

        st.dataframe(
            imp,
            use_container_width=True
        )

    with col2:

        fig_imp = px.bar(

            imp,

            x="Importance",

            y="Feature",

            orientation="h",

            template="plotly_dark",

            title="Feature Importance"

        )

        st.plotly_chart(
            fig_imp,
            use_container_width=True
        )

    st.subheader(
        "🧠 SHAP Explainability"
    )

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    if isinstance(shap_values, list):
        shap_data = shap_values[1]
    else:
        shap_data = shap_values[:,:,1]

    fig3, ax = plt.subplots(
        figsize=(12,6)
    )

    shap.summary_plot(
        shap_data,
        X,
        show=False
    )

    st.pyplot(fig3)


# =====================================
# EVALUATION
# =====================================

elif page == "📈 Evaluation":

    st.subheader(
        "📊 Model Performance Comparison"
    )

    comparison = pd.DataFrame({

        "Model":[
            "Random Forest",
            "Logistic Regression"
        ],

        "Accuracy (%)":[
            rf_acc,
            lr_acc
        ]

    })

    fig_cmp = px.bar(

        comparison,

        x="Model",

        y="Accuracy (%)",

        text="Accuracy (%)",

        color="Model",

        template="plotly_dark",

        title="Model Accuracy Comparison"

    )

    fig_cmp.update_traces(
        textposition="outside"
    )

    fig_cmp.update_layout(
        height=500,
        title_x=0.5
    )

    st.plotly_chart(
        fig_cmp,
        use_container_width=True
    )

    # KPI SECTION

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "🌲 RF Accuracy",
        f"{rf_acc}%"
    )

    col2.metric(
        "📈 LR Accuracy",
        f"{lr_acc}%"
    )

    col3.metric(
        "🧪 Samples Tested",
        len(y_test)
    )

    col4.metric(
        "🚨 Anomalies",
        len(anomalies)
    )

    st.markdown("---")

    # CLASSIFICATION REPORT

    st.subheader(
        "📋 Classification Report"
    )

    st.dataframe(
        report_df,
        use_container_width=True
    )

    st.markdown("---")

    # CONFUSION MATRIX

    st.subheader(
        "🎯 Confusion Matrix"
    )

    cm = confusion_matrix(
        y_test,
        pred
    )

    fig_cm = px.imshow(

        cm,

        text_auto=True,

        color_continuous_scale="Blues",

        labels=dict(
            x="Predicted",
            y="Actual"
        ),

        title="Confusion Matrix"

    )

    fig_cm.update_layout(
        height=500,
        title_x=0.5
    )

    st.plotly_chart(
        fig_cm,
        use_container_width=True
    )

    st.markdown("---")

    # ROC CURVE

    st.subheader(
        "📈 ROC Curve Analysis"
    )

    fpr, tpr, _ = roc_curve(
        y_test,
        proba
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    auc_col1, auc_col2 = st.columns(2)

    auc_col1.metric(
        "ROC AUC Score",
        f"{roc_auc:.3f}"
    )

    auc_col2.metric(
        "Model",
        "Random Forest"
    )

    roc_fig = go.Figure()

    roc_fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            line=dict(
                width=5,
                color="#00d4ff"
            ),
            name=f"Random Forest (AUC={roc_auc:.2f})"
        )
    )

    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(
                dash="dash",
                color="gray"
            ),
            name="Random Baseline"
        )
    )

    roc_fig.update_layout(

        title="Receiver Operating Characteristic",

        xaxis_title="False Positive Rate",

        yaxis_title="True Positive Rate",

        template="plotly_dark",

        height=600,

        title_x=0.5

    )

    st.plotly_chart(
        roc_fig,
        use_container_width=True
    )

	
# =====================================
# REPORT
# =====================================

elif page == "📋 Report":

    st.success("""

    ✓ Random Forest Anomaly Detection

    ✓ Logistic Regression Comparison

    ✓ Rolling Window Feature Engineering

    ✓ Mean Gait Trajectory Analysis

    ✓ Feature Importance Analysis

    ✓ SHAP Explainability

    ✓ Confusion Matrix

    ✓ ROC Curve

    ✓ Interactive Dashboard

    """)

    st.info("""

    Future Work

    • SMOTE Class Balancing

    • Real-Time Monitoring

    • Deep Learning Models

    • Multi-Subject Analysis

    """)

    csv = anomalies.to_csv(
        index=False
    )

    st.download_button(

        "📥 Download Anomalies CSV",

        csv,

        "anomalies.csv",

        "text/csv"

    )