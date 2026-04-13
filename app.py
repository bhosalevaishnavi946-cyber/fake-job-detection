import streamlit as st

st.set_page_config(page_title="Home", page_icon="🏠", layout="wide")

# ---------------- HERO SECTION ----------------
st.markdown("<h1 style='text-align:center;'>🔍 Fake Job Detection System</h1>", unsafe_allow_html=True)

st.markdown(
    "<h4 style='text-align:center;'>Protect Yourself from Online Job Scams 🚨</h4>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Analyze job postings instantly using Artificial Intelligence and detect whether they are <b>Real or Fake</b>.</p>",
    unsafe_allow_html=True
)

# Center button
if st.button("Start Detection"):
    st.switch_page("fake_job.py")

# ---------------- FEATURES ----------------
st.markdown("## 🚀 Features")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("⚡ Instant Detection\n\nGet results within seconds")

with col2:
    st.info("🤖 AI Powered\n\nMachine learning model trained on job data")

with col3:
    st.warning("🔒 Safe & Secure\n\nAvoid fake job scams and fraud")

# ---------------- HOW IT WORKS ----------------
st.markdown("## ⚙️ How It Works")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 1️⃣ Enter Job\nPaste job description")

with col2:
    st.markdown("### 2️⃣ Analyze\nClick analyze button")

with col3:
    st.markdown("### 3️⃣ Get Result\nSee Real or Fake prediction")

# ---------------- ALERT ----------------
st.error("🚨 Never pay money for job applications. Always verify before applying!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>🎓 Final Year Project | Fake Job Detection System</p>",
    unsafe_allow_html=True
)
import nltk
nltk.download('stopwords')