import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# ── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

# ── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #F0F4FF; }
    .stApp { background-color: #F0F4FF; }

    .hero-box {
        background: linear-gradient(135deg, #4361EE 0%, #3A0CA3 100%);
        padding: 2.5rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
    .hero-box h1 { font-size: 2.2rem; font-weight: 700; margin: 0; }
    .hero-box p  { font-size: 1.05rem; opacity: 0.88; margin-top: 0.5rem; }

    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }
    .metric-card .number { font-size: 2rem; font-weight: 700; color: #4361EE; }
    .metric-card .label  { font-size: 0.85rem; color: #6B7280; margin-top: 2px; }

    .result-pass {
        background: linear-gradient(135deg, #06D6A0, #059669);
        color: white; padding: 1.5rem; border-radius: 14px;
        text-align: center; font-size: 1.4rem; font-weight: 700;
    }
    .result-fail {
        background: linear-gradient(135deg, #EF233C, #B91C1C);
        color: white; padding: 1.5rem; border-radius: 14px;
        text-align: center; font-size: 1.4rem; font-weight: 700;
    }
    .tip-box {
        background: #EEF2FF; border-left: 4px solid #4361EE;
        padding: 0.8rem 1.2rem; border-radius: 0 10px 10px 0;
        margin: 0.4rem 0; font-size: 0.95rem;
    }
    div[data-testid="stSlider"] > div { padding-top: 0.2rem; }
    .section-title {
        font-size: 1.15rem; font-weight: 700;
        color: #1E1E2E; margin: 1.5rem 0 0.8rem;
        border-bottom: 2px solid #4361EE;
        padding-bottom: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open('model.pkl',  'rb') as f: model  = pickle.load(f)
    with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
    return model, scaler

model, scaler = load_model()

# ── HERO ──────────────────────────────────────────────────────
st.markdown("""
<div class="hero-box">
  <h1>🎓 Student Performance Predictor</h1>
  <p>Enter student details below to predict academic outcome using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── TOP METRICS ───────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("87%",   "Model Accuracy"),
    ("0.830", "ROC-AUC Score"),
    ("1,000", "Students Trained On"),
    ("3",     "ML Models Compared"),
]
for col, (num, lbl) in zip([c1,c2,c3,c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="number">{num}</div>
        <div class="label">{lbl}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── TWO COLUMN LAYOUT ─────────────────────────────────────────
left, right = st.columns([1.1, 1], gap="large")

with left:
    st.markdown('<div class="section-title">📋 Student Input Form</div>', unsafe_allow_html=True)

    study_hours   = st.slider("📚 Daily Study Hours",          0.0, 12.0, 5.0, 0.5)
    attendance    = st.slider("🏫 Attendance Percentage (%)",  30,  100,  75)
    prev_score    = st.slider("📝 Previous Exam Score",        20,  100,  60)
    sleep_hours   = st.slider("😴 Sleep Hours per Night",      3.0, 10.0, 7.0, 0.5)
    extra_classes = st.radio("📖 Attends Extra Classes?",
                             ["Yes", "No"], horizontal=True)
    parental_edu  = st.selectbox("👨‍👩‍🎓 Parental Education Level",
                                 ["No Formal Education", "School Level", "Graduate & Above"])

    predict_btn = st.button("🔮 Predict Now", use_container_width=True, type="primary")

with right:
    st.markdown('<div class="section-title">📊 Prediction Result</div>', unsafe_allow_html=True)

    if predict_btn:
        ec  = 1 if extra_classes == "Yes" else 0
        pe  = ["No Formal Education","School Level","Graduate & Above"].index(parental_edu)

        inp = np.array([[study_hours, attendance, prev_score, sleep_hours, ec, pe]])
        inp_sc = scaler.transform(inp)

        pred  = model.predict(inp_sc)[0]
        proba = model.predict_proba(inp_sc)[0]
        conf  = proba[pred] * 100

        if pred == 1:
            st.markdown(f'<div class="result-pass">✅ PASS &nbsp;|&nbsp; Confidence: {conf:.1f}%</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="result-fail">❌ AT RISK &nbsp;|&nbsp; Confidence: {conf:.1f}%</div>',
                        unsafe_allow_html=True)

        # Probability bar
        st.markdown("<br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(5, 1.2))
        ax.barh([''], [proba[0]*100], color='#EF233C', height=0.5, label='Fail Risk')
        ax.barh([''], [proba[1]*100], left=[proba[0]*100],
                color='#06D6A0', height=0.5, label='Pass Chance')
        ax.set_xlim(0, 100); ax.set_xlabel('Probability (%)')
        ax.set_title('Pass vs Fail Probability', fontweight='bold', fontsize=11)
        ax.legend(loc='lower right', fontsize=8)
        fig.patch.set_facecolor('#F0F4FF'); ax.set_facecolor('#F0F4FF')
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # Tips
        st.markdown('<div class="section-title">💡 Improvement Tips</div>', unsafe_allow_html=True)
        tips = []
        if study_hours < 5:  tips.append("📚 Increase daily study time to at least 5 hours")
        if attendance  < 75: tips.append("🏫 Improve attendance — aim for 80%+")
        if prev_score  < 55: tips.append("📝 Focus on weak subjects from previous exams")
        if sleep_hours < 6:  tips.append("😴 Get 7-8 hours of sleep for better retention")
        if ec == 0:           tips.append("📖 Consider joining extra coaching classes")
        if not tips:          tips.append("🌟 Keep it up — you're on the right track!")

        for tip in tips:
            st.markdown(f'<div class="tip-box">{tip}</div>', unsafe_allow_html=True)

    else:
        st.info("👈 Fill in the student details and click **Predict Now**")

        # Show feature importance plot if available
        if os.path.exists('plots/feature_importance.png'):
            st.markdown('<div class="section-title">🔍 What Drives Predictions?</div>',
                        unsafe_allow_html=True)
            st.image('plots/feature_importance.png', use_container_width=True)

# ── MODEL COMPARISON SECTION ─────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">🤖 Model Performance Comparison</div>',
            unsafe_allow_html=True)

mc1, mc2, mc3 = st.columns(3)
model_data = [
    ("Logistic Regression", "86.00%", "0.8297", "82.38%", "#4361EE"),
    ("Random Forest",       "84.00%", "0.8130", "81.88%", "#3A86FF"),
    ("Gradient Boosting ✓", "87.00%", "0.8300", "80.25%", "#06D6A0"),
]
for col, (name, acc, auc, cv, color) in zip([mc1, mc2, mc3], model_data):
    col.markdown(f"""
    <div class="metric-card" style="border-top: 4px solid {color};">
        <div style="font-weight:700; font-size:1rem; margin-bottom:0.6rem">{name}</div>
        <div style="display:flex; justify-content:space-around; margin-top:0.4rem">
            <div><div class="number" style="font-size:1.3rem;color:{color}">{acc}</div>
                 <div class="label">Accuracy</div></div>
            <div><div class="number" style="font-size:1.3rem;color:{color}">{auc}</div>
                 <div class="label">AUC</div></div>
            <div><div class="number" style="font-size:1.3rem;color:{color}">{cv}</div>
                 <div class="label">CV Score</div></div>
        </div>
    </div>""", unsafe_allow_html=True)

if os.path.exists('plots/model_comparison.png'):
    st.image('plots/model_comparison.png', use_container_width=True)

# ── FOOTER ────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<p style='text-align:center; color:#9CA3AF; font-size:0.85rem'>
Built with Scikit-learn + Streamlit &nbsp;|&nbsp;
Dataset: Student Performance (1,000 samples) &nbsp;|&nbsp;
Amazon ML Summer School 2026 Portfolio Project
</p>""", unsafe_allow_html=True)
