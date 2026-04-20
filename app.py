import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from predict import make_prediction


st.set_page_config(
    page_title="Hospital Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

st.markdown("""
<style>
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    color: #1f4e79;
    margin-bottom: 0.2rem;
}
.subtle {
    color: #555;
    font-size: 0.95rem;
}
.risk-box {
    padding: 1rem;
    border-radius: 0.75rem;
    color: white;
    font-weight: 700;
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 1rem;
    font-size: 1.2rem;
}
.high { background-color: #c0392b; }
.moderate { background-color: #d68910; }
.low { background-color: #1e8449; }
</style>
""", unsafe_allow_html=True)


def create_gauge(probability, threshold):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Predicted 30-Day Readmission Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "steps": [
                    {"range": [0, 30], "color": "#d5f5e3"},
                    {"range": [30, 70], "color": "#fcf3cf"},
                    {"range": [70, 100], "color": "#fadbd8"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        )
    )
    fig.update_layout(height=340, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def get_risk_band(probability):
    if probability >= 0.70:
        return "High", "high"
    if probability >= 0.30:
        return "Moderate", "moderate"
    return "Low", "low"


def collect_inputs():
    with st.sidebar:
        st.header("Patient Information")

        age = st.selectbox(
            "Age Group",
            ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
             "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"],
            index=5
        )

        gender = st.selectbox("Gender", ["Male", "Female"])
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other"])

        admission_type_id = st.selectbox("Admission Type", [1, 2, 3, 4])
        discharge_disposition_id = st.selectbox("Discharge Disposition", [1, 2, 3, 6])
        admission_source_id = st.selectbox("Admission Source", [1, 2, 3, 4, 5, 6, 7, 8])

        time_in_hospital = st.slider("Length of Stay (days)", 1, 14, 3)
        num_lab_procedures = st.slider("Number of Lab Procedures", 1, 100, 40)
        num_procedures = st.slider("Number of Procedures", 0, 6, 1)
        num_medications = st.slider("Number of Medications", 1, 50, 10)
        number_outpatient = st.slider("Outpatient Visits (past year)", 0, 15, 0)
        number_emergency = st.slider("Emergency Visits (past year)", 0, 15, 0)
        number_inpatient = st.slider("Prior Inpatient Visits (past year)", 0, 15, 0)
        number_diagnoses = st.slider("Number of Diagnoses", 1, 16, 5)

        max_glu_serum = st.selectbox("Maximum Glucose Serum", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("HbA1c Result", ["None", "Norm", ">7", ">8"])
        change = st.selectbox("Medication Change", ["No", "Ch"])
        diabetesMed = st.selectbox("Diabetes Medication", ["Yes", "No"])

        st.subheader("Diagnosis Codes")
        diag_1 = st.text_input("Primary Diagnosis Code", "250")
        diag_2 = st.text_input("Secondary Diagnosis Code", "250")
        diag_3 = st.text_input("Tertiary Diagnosis Code", "250")

        st.subheader("Selected Medication Status")
        metformin = st.selectbox("Metformin", ["No", "Steady", "Up", "Down"])
        insulin = st.selectbox("Insulin", ["No", "Steady", "Up", "Down"])
        glipizide = st.selectbox("Glipizide", ["No", "Steady", "Up", "Down"])
        glyburide = st.selectbox("Glyburide", ["No", "Steady", "Up", "Down"])
        pioglitazone = st.selectbox("Pioglitazone", ["No", "Steady", "Up", "Down"])
        rosiglitazone = st.selectbox("Rosiglitazone", ["No", "Steady", "Up", "Down"])

        predict_button = st.button("Calculate Risk", type="primary", use_container_width=True)

    return {
        "race": race,
        "gender": gender,
        "age": age,
        "admission_type_id": admission_type_id,
        "discharge_disposition_id": discharge_disposition_id,
        "admission_source_id": admission_source_id,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "diag_1": diag_1,
        "diag_2": diag_2,
        "diag_3": diag_3,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        "metformin": metformin,
        "repaglinide": "No",
        "nateglinide": "No",
        "chlorpropamide": "No",
        "glimepiride": "No",
        "acetohexamide": "No",
        "glipizide": glipizide,
        "glyburide": glyburide,
        "tolbutamide": "No",
        "pioglitazone": pioglitazone,
        "rosiglitazone": rosiglitazone,
        "acarbose": "No",
        "miglitol": "No",
        "troglitazone": "No",
        "tolazamide": "No",
        "examide": "No",
        "citoglipton": "No",
        "insulin": insulin,
        "glyburide-metformin": "No",
        "glipizide-metformin": "No",
        "glimepiride-pioglitazone": "No",
        "metformin-rosiglitazone": "No",
        "metformin-pioglitazone": "No",
        "change": change,
        "diabetesMed": diabetesMed,
    }, predict_button


def main():
    st.markdown('<div class="main-title">Hospital Readmission Risk Predictor</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtle">Educational prototype for predicting 30-day readmission risk in diabetic patients.</div>',
        unsafe_allow_html=True
    )

    with st.expander("How to interpret this tool"):
        st.write(
            """
            The probability shown is the model's estimated chance of readmission within 30 days.
            The binary decision uses the saved tuned threshold from model validation.
            The risk band is a simple visual guide and is separate from the trained threshold.
            """
        )

    inputs, predict_button = collect_inputs()

    if not predict_button:
        st.info("Enter patient details in the sidebar and click **Calculate Risk**.")
        return

    result = make_prediction(inputs)

    probability = result["predicted_probability"]
    threshold = result["threshold"]
    prediction = result["prediction"]
    confidence = result["model_confidence"]
    risk_band, css_class = get_risk_band(probability)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.plotly_chart(create_gauge(probability, threshold), use_container_width=True)
        st.markdown(
            f'<div class="risk-box {css_class}">{risk_band} Risk Band: {probability:.1%}</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted Probability", f"{probability:.1%}")
        m2.metric("Binary Threshold", f"{threshold:.1%}")
        m3.metric("Binary Prediction", "Positive" if prediction == 1 else "Negative")

    with col2:
        st.subheader("Model Output")
        st.write(f"**Selected model:** {result['model_name']}")
        st.write(f"**Model confidence:** {confidence:.1%}")
        st.write(f"**PR-AUC:** {result['metadata']['pr_auc']:.3f}")
        st.write(f"**ROC-AUC:** {result['metadata']['roc_auc']:.3f}")

        st.subheader("Note")
        st.write(
            """
            This app is for educational use only. It supports discussion and planning,
            but should not replace clinical judgement.
            """
        )

    st.markdown("---")
    st.write(
        "This prototype is based on historical data and may not generalise to all hospitals or patient populations."
    )

    summary_df = pd.DataFrame([
        {"field": "Predicted probability", "value": f"{probability:.4f}"},
        {"field": "Model confidence", "value": f"{confidence:.4f}"},
        {"field": "Binary threshold", "value": f"{threshold:.4f}"},
        {"field": "Binary prediction", "value": int(prediction)},
        {"field": "Risk band", "value": risk_band},
        {"field": "Selected model", "value": result["model_name"]},
    ])

    st.download_button(
        "Download prediction summary",
        summary_df.to_csv(index=False),
        file_name="prediction_summary.csv",
        mime="text/csv"
    )


if __name__ == "__main__":
    main()
