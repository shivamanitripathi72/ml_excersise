import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

from imblearn.over_sampling import SMOTE


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🩺 Diabetes Risk Prediction")
st.write(
    "Random Forest + SMOTE based Diabetes Risk Classification"
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙️ Settings")

st.sidebar.info(
    """
    Model:
    Random Forest + SMOTE

    Classes:
    Low
    Moderate
    High
    """
)


# ============================================================
# FILE UPLOAD
# ============================================================

st.header("📂 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload your diabetes CSV file",
    type=["csv"]
)

if uploaded_file is None:

    st.info("Please upload your CSV dataset to continue.")

    st.stop()


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(r'C:\Users\dell\OneDrive\Documents\ml_excersise\diabetes_risk_prediction_dataset.csv')

st.success("Dataset loaded successfully!")

st.subheader("📊 Dataset Preview")

st.dataframe(
    df.head(10),
    use_container_width=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Rows",
        df.shape[0]
    )

with col2:
    st.metric(
        "Columns",
        df.shape[1]
    )

with col3:
    st.metric(
        "Missing Values",
        int(df.isnull().sum().sum())
    )

with col4:
    st.metric(
        "Duplicate Rows",
        int(df.duplicated().sum())
    )


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

selected_features = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "Family_History_Diabetes",
    "Age",
    "Hypertension",
    "BMI",
    "Heart_Disease",
    "Physical_Activity_Level",
    "PCOS",
    "Weight_kg",
    "Blood_Pressure_Systolic",
    "Fatty_Liver",
    "Diet_Quality",
    "Stress_Level",
    "Smoking_Status"
]

target = "Diabetes_Risk"

required_columns = selected_features + [target]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if len(missing_columns) > 0:

    st.error(
        "The following required columns are missing:"
    )

    st.write(missing_columns)

    st.stop()


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

st.header("🧹 Data Preprocessing")

# Numerical columns
numerical_cols = [
    "Fasting_Blood_Sugar",
    "HbA1c",
    "Age",
    "BMI",
    "Weight_kg",
    "Blood_Pressure_Systolic"
]

# Categorical columns
categorical_cols = [
    "Family_History_Diabetes",
    "Hypertension",
    "Heart_Disease",
    "Physical_Activity_Level",
    "PCOS",
    "Fatty_Liver",
    "Diet_Quality",
    "Stress_Level",
    "Smoking_Status"
]


# Fill numerical missing values
for col in numerical_cols:

    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

    df[col] = df[col].fillna(
        df[col].median()
    )


# Fill categorical missing values
for col in categorical_cols:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )


# Target missing values
df = df.dropna(
    subset=[target]
)


st.success("Missing value preprocessing completed.")


# ============================================================
# FEATURES AND TARGET
# ============================================================

X = df[selected_features].copy()

y = df[target].copy()


# ============================================================
# TARGET DISTRIBUTION
# ============================================================

st.subheader("🎯 Target Distribution")

target_counts = y.value_counts()

st.dataframe(
    target_counts.rename("Count"),
    use_container_width=True
)

fig, ax = plt.subplots(
    figsize=(7, 4)
)

sns.countplot(
    data=df,
    x=target,
    order=["Low", "Moderate", "High"],
    ax=ax
)

ax.set_title(
    "Diabetes Risk Distribution"
)

ax.set_xlabel(
    "Diabetes Risk"
)

ax.set_ylabel(
    "Count"
)

st.pyplot(fig)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

st.write(
    f"Training samples: **{X_train.shape[0]}**"
)

st.write(
    f"Testing samples: **{X_test.shape[0]}**"
)


# ============================================================
# PREPROCESSOR
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_cols
        ),
        (
            "num",
            "passthrough",
            numerical_cols
        )
    ]
)


# ============================================================
# ENCODING
# ============================================================

X_train_encoded = preprocessor.fit_transform(
    X_train
)

X_test_encoded = preprocessor.transform(
    X_test
)


# ============================================================
# DATAFRAME AFTER ENCODING
# ============================================================

feature_names = (
    preprocessor
    .get_feature_names_out()
)

X_train_encoded_df = pd.DataFrame(
    X_train_encoded,
    columns=feature_names,
    index=X_train.index
)

X_test_encoded_df = pd.DataFrame(
    X_test_encoded,
    columns=feature_names,
    index=X_test.index
)


# ============================================================
# SCALING
# ============================================================

num_encoded_cols = [
    "num__Fasting_Blood_Sugar",
    "num__HbA1c",
    "num__Age",
    "num__BMI",
    "num__Weight_kg",
    "num__Blood_Pressure_Systolic"
]

scaler = StandardScaler()

X_train_encoded_df[
    num_encoded_cols
] = scaler.fit_transform(
    X_train_encoded_df[
        num_encoded_cols
    ]
)

X_test_encoded_df[
    num_encoded_cols
] = scaler.transform(
    X_test_encoded_df[
        num_encoded_cols
    ]
)


# ============================================================
# SMOTE
# ============================================================

st.header("⚖️ Class Balancing using SMOTE")

before_smote = y_train.value_counts()

st.subheader("Before SMOTE")

st.dataframe(
    before_smote.rename("Count"),
    use_container_width=True
)


smote = SMOTE(
    random_state=42
)

X_train_smote, y_train_smote = (
    smote.fit_resample(
        X_train_encoded_df,
        y_train
    )
)


after_smote = y_train_smote.value_counts()

st.subheader("After SMOTE")

st.dataframe(
    after_smote.rename("Count"),
    use_container_width=True
)


# ============================================================
# RANDOM FOREST MODEL
# ============================================================

st.header("🌳 Random Forest Model")

rf_smote = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)


# ============================================================
# TRAIN MODEL
# ============================================================

with st.spinner(
    "Training Random Forest model..."
):

    rf_smote.fit(
        X_train_smote,
        y_train_smote
    )

st.success(
    "Random Forest + SMOTE training completed!"
)


# ============================================================
# PREDICTION
# ============================================================

y_pred_smote = rf_smote.predict(
    X_test_encoded_df
)


# ============================================================
# ACCURACY
# ============================================================

accuracy_smote = accuracy_score(
    y_test,
    y_pred_smote
)

st.header("📈 Model Performance")

st.metric(
    "Random Forest + SMOTE Accuracy",
    f"{accuracy_smote * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

labels = [
    "Low",
    "Moderate",
    "High"
]

report = classification_report(
    y_test,
    y_pred_smote,
    labels=labels,
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

st.subheader(
    "📋 Classification Report"
)

st.dataframe(
    report_df.round(3),
    use_container_width=True
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

st.subheader(
    "🔲 Confusion Matrix"
)

cm_smote = confusion_matrix(
    y_test,
    y_pred_smote,
    labels=labels
)

fig, ax = plt.subplots(
    figsize=(7, 5)
)

sns.heatmap(
    cm_smote,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
    ax=ax
)

ax.set_xlabel(
    "Predicted"
)

ax.set_ylabel(
    "Actual"
)

ax.set_title(
    "Random Forest + SMOTE"
)

st.pyplot(fig)


# ============================================================
# CLASS-WISE METRICS
# ============================================================

st.subheader(
    "📊 Class-wise Performance"
)

metric_col1, metric_col2, metric_col3 = (
    st.columns(3)
)

with metric_col1:

    st.metric(
        "Low Recall",
        f"{report['Low']['recall'] * 100:.2f}%"
    )

    st.metric(
        "Low F1 Score",
        f"{report['Low']['f1-score']:.2f}"
    )


with metric_col2:

    st.metric(
        "Moderate Recall",
        f"{report['Moderate']['recall'] * 100:.2f}%"
    )

    st.metric(
        "Moderate F1 Score",
        f"{report['Moderate']['f1-score']:.2f}"
    )


with metric_col3:

    st.metric(
        "High Recall",
        f"{report['High']['recall'] * 100:.2f}%"
    )

    st.metric(
        "High F1 Score",
        f"{report['High']['f1-score']:.2f}"
    )


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

st.header(
    "⭐ Feature Importance"
)

importance = rf_smote.feature_importances_

importance_df = pd.DataFrame(
    {
        "Feature": X_train_encoded_df.columns,
        "Importance": importance
    }
)

importance_df = (
    importance_df
    .sort_values(
        "Importance",
        ascending=False
    )
)

st.dataframe(
    importance_df.head(15),
    use_container_width=True
)


# ============================================================
# FEATURE IMPORTANCE GRAPH
# ============================================================

fig, ax = plt.subplots(
    figsize=(9, 6)
)

top_features = (
    importance_df
    .head(15)
    .sort_values(
        "Importance"
    )
)

ax.barh(
    top_features["Feature"],
    top_features["Importance"]
)

ax.set_xlabel(
    "Importance"
)

ax.set_ylabel(
    "Feature"
)

ax.set_title(
    "Top 15 Feature Importances"
)

plt.tight_layout()

st.pyplot(fig)


# ============================================================
# INDIVIDUAL PATIENT PREDICTION
# ============================================================

st.divider()

st.header(
    "🧑‍⚕️ Individual Patient Risk Prediction"
)

st.write(
    "Enter patient information below to predict diabetes risk."
)


# ============================================================
# INPUT FORM
# ============================================================

with st.form(
    "prediction_form"
):

    col1, col2, col3 = st.columns(3)


    # --------------------------------------------------------
    # Numerical Inputs
    # --------------------------------------------------------

    with col1:

        age = st.number_input(
            "Age",
            min_value=1.0,
            max_value=120.0,
            value=54.0
        )

        fasting_blood_sugar = st.number_input(
            "Fasting Blood Sugar",
            min_value=0.0,
            value=100.0
        )

        hba1c = st.number_input(
            "HbA1c",
            min_value=0.0,
            value=5.5
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0
        )

        weight = st.number_input(
            "Weight (kg)",
            min_value=1.0,
            value=70.0
        )

        systolic_bp = st.number_input(
            "Blood Pressure Systolic",
            min_value=50.0,
            value=120.0
        )


    # --------------------------------------------------------
    # Categorical Inputs
    # --------------------------------------------------------

    with col2:

        family_history = st.selectbox(
            "Family History Diabetes",
            ["Yes", "No"]
        )

        hypertension = st.selectbox(
            "Hypertension",
            ["Yes", "No"]
        )

        heart_disease = st.selectbox(
            "Heart Disease",
            ["Yes", "No"]
        )

        physical_activity = st.selectbox(
            "Physical Activity Level",
            ["Low", "Moderate", "High"]
        )

        pcos = st.selectbox(
            "PCOS",
            ["Yes", "No"]
        )


    with col3:

        fatty_liver = st.selectbox(
            "Fatty Liver",
            ["Yes", "No"]
        )

        diet_quality = st.selectbox(
            "Diet Quality",
            ["Poor", "Average", "Good"]
        )

        stress_level = st.selectbox(
            "Stress Level",
            ["Low", "Moderate", "High"]
        )

        smoking_status = st.selectbox(
            "Smoking Status",
            ["Never", "Former", "Current"]
        )


    submit = st.form_submit_button(
        "🔍 Predict Diabetes Risk"
    )


# ============================================================
# PREDICT INDIVIDUAL PATIENT
# ============================================================

if submit:

    patient_data = pd.DataFrame(
        {
            "Fasting_Blood_Sugar": [
                fasting_blood_sugar
            ],

            "HbA1c": [
                hba1c
            ],

            "Family_History_Diabetes": [
                family_history
            ],

            "Age": [
                age
            ],

            "Hypertension": [
                hypertension
            ],

            "BMI": [
                bmi
            ],

            "Heart_Disease": [
                heart_disease
            ],

            "Physical_Activity_Level": [
                physical_activity
            ],

            "PCOS": [
                pcos
            ],

            "Weight_kg": [
                weight
            ],

            "Blood_Pressure_Systolic": [
                systolic_bp
            ],

            "Fatty_Liver": [
                fatty_liver
            ],

            "Diet_Quality": [
                diet_quality
            ],

            "Stress_Level": [
                stress_level
            ],

            "Smoking_Status": [
                smoking_status
            ]
        }
    )


    # --------------------------------------------------------
    # Transform Patient Data
    # --------------------------------------------------------

    patient_encoded = (
        preprocessor.transform(
            patient_data
        )
    )

    patient_encoded_df = pd.DataFrame(
        patient_encoded,
        columns=feature_names
    )


    # --------------------------------------------------------
    # Scale Numerical Features
    # --------------------------------------------------------

    patient_encoded_df[
        num_encoded_cols
    ] = scaler.transform(
        patient_encoded_df[
            num_encoded_cols
        ]
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    prediction = rf_smote.predict(
        patient_encoded_df
    )[0]


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    probabilities = (
        rf_smote
        .predict_proba(
            patient_encoded_df
        )[0]
    )

    class_probabilities = dict(
        zip(
            rf_smote.classes_,
            probabilities
        )
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Prediction Result"
    )

    if prediction == "Low":

        st.success(
            "🟢 Predicted Diabetes Risk: LOW"
        )

    elif prediction == "Moderate":

        st.warning(
            "🟠 Predicted Diabetes Risk: MODERATE"
        )

    else:

        st.error(
            "🔴 Predicted Diabetes Risk: HIGH"
        )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )

    probability_col1, probability_col2, probability_col3 = (
        st.columns(3)
    )


    with probability_col1:

        st.metric(
            "Low",
            f"{class_probabilities.get('Low', 0) * 100:.2f}%"
        )


    with probability_col2:

        st.metric(
            "Moderate",
            f"{class_probabilities.get('Moderate', 0) * 100:.2f}%"
        )


    with probability_col3:

        st.metric(
            "High",
            f"{class_probabilities.get('High', 0) * 100:.2f}%"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Diabetes Risk Prediction | Random Forest + SMOTE"
)