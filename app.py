import streamlit as st  # type: ignore
import pandas as pd
import joblib

# ----------------------------
# Page configuration
# ----------------------------
st.set_page_config(
    page_title="Heart Stroke Prediction",
    page_icon="❤️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ----------------------------
# Custom dark theme styling
# ----------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: radial-gradient(circle at top left, #1b1024 0%, #0e0b16 45%, #0a0710 100%);
        color: #f2f0f7;
    }

    /* Hide default Streamlit chrome */
    #MainMenu, footer, header {visibility: hidden;}

    /* Hero header */
    .hero {
        text-align: center;
        padding: 1.4rem 1rem 1.2rem 1rem;
        margin-bottom: 1.2rem;
        border-radius: 18px;
        background: linear-gradient(135deg, rgba(255,45,85,0.18), rgba(122,45,255,0.18));
        border: 1px solid rgba(255,255,255,0.08);
    }
    .hero h1 {
        font-size: 2.1rem;
        margin: 0;
        background: linear-gradient(90deg, #ff5c7a, #b06bff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    .hero p {
        color: #cfc9db;
        margin-top: 0.4rem;
        font-size: 0.95rem;
    }

    /* Section labels */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #ff8fa3;
        margin: 1.2rem 0 0.4rem 0;
        border-left: 4px solid #ff5c7a;
        padding-left: 0.6rem;
    }

    /* Card wrapper for inputs */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 0.4rem 0.6rem;
    }

    /* Inputs */
    .stSlider, .stSelectbox, .stNumberInput {
        margin-bottom: 0.3rem;
    }
    label, .stMarkdown p {
        color: #e8e4f0 !important;
    }

    /* Predict button */
    div.stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff5c7a, #b06bff);
        color: white;
        font-weight: 700;
        font-size: 1.05rem;
        padding: 0.7rem 0;
        border: none;
        border-radius: 12px;
        margin-top: 1.4rem;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
        box-shadow: 0 4px 18px rgba(176,107,255,0.35);
    }
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(255,92,122,0.45);
        color: white;
    }

    /* Result cards */
    .result-card {
        margin-top: 1.4rem;
        padding: 1.3rem;
        border-radius: 16px;
        text-align: center;
        font-size: 1.15rem;
        font-weight: 700;
        border: 1px solid rgba(255,255,255,0.1);
    }
    .result-high {
        background: linear-gradient(135deg, rgba(255,71,87,0.22), rgba(255,71,87,0.05));
        color: #ff8080;
    }
    .result-low {
        background: linear-gradient(135deg, rgba(46,213,115,0.22), rgba(46,213,115,0.05));
        color: #6bffa0;
    }

    .footer-note {
        text-align: center;
        color: #6f6a80;
        font-size: 0.78rem;
        margin-top: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Hero header
# ----------------------------
st.markdown(
    """
    <div class="hero">
        <h1>❤️ Heart Stroke Risk Predictor</h1>
        <p>by JAYA &nbsp;•&nbsp; Enter your health details below to estimate your risk</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------
# Load model artifacts
# ----------------------------
try:
    model = joblib.load("knn_heart_model.pkl")
    scaler = joblib.load("heart_scaler.pkl")
    expected_columns = joblib.load("heart_columns.pkl")
except FileNotFoundError as e:
    st.error(f"⚠️ Could not load model files: {e}")
    st.stop()

# ----------------------------
# Input form
# ----------------------------
with st.form("prediction_form"):

    st.markdown('<div class="section-title">👤 Personal Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 100, 40)
    with col2:
        sex = st.selectbox("Sex", ["M", "F"])

    st.markdown('<div class="section-title">🫀 Heart & Vitals</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
        resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
        cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
    with col4:
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.slider("Max Heart Rate", 60, 220, 150)
        exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    st.markdown('<div class="section-title">📈 ECG Reading</div>', unsafe_allow_html=True)
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)

    submitted = st.form_submit_button("🔍 Predict My Risk")

# ----------------------------
# Prediction
# ----------------------------
if submitted:
    with st.spinner("Analyzing your details..."):
        raw_input = {
            "Age": age,
            "RestingBP": resting_bp,
            "Cholesterol": cholesterol,
            "FastingBS": fasting_bs,
            "MaxHR": max_hr,
            "Oldpeak": oldpeak,
            "Sex_" + sex: 1,
            "ChestPainType_" + chest_pain: 1,
            "RestingECG_" + resting_ecg: 1,
            "ExerciseAngina_" + exercise_angina: 1,
            "ST_Slope_" + st_slope: 1,
        }

        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Reorder columns
        input_df = input_df[expected_columns]

        # Scale the input
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Try to get a probability/confidence if the model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(scaled_input)[0]
            confidence = round(max(proba) * 100, 1)

    if prediction == 1:
        conf_text = f" (confidence: {confidence}%)" if confidence is not None else ""
        st.markdown(
            f'<div class="result-card result-high">⚠️ High Risk of Heart Disease{conf_text}<br>'
            f'<span style="font-size:0.85rem; font-weight:400; color:#e8b4b4;">'
            f'Please consult a cardiologist for a full evaluation.</span></div>',
            unsafe_allow_html=True,
        )
    else:
        conf_text = f" (confidence: {confidence}%)" if confidence is not None else ""
        st.markdown(
            f'<div class="result-card result-low">✅ Low Risk of Heart Disease{conf_text}<br>'
            f'<span style="font-size:0.85rem; font-weight:400; color:#b4e8c8;">'
            f'Keep up the healthy habits and regular checkups.</span></div>',
            unsafe_allow_html=True,
        )

st.markdown(
    '<div class="footer-note">This tool provides an estimate only and is not a substitute for professional medical advice.</div>',
    unsafe_allow_html=True,
)