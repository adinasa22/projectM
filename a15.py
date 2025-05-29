import streamlit as st
from io import StringIO
import time
import random

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="JD vs CV Compare Tool",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# -----------------------------
# Custom CSS Styling
# -----------------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(145deg, #001f3f, #003366);
        color: white;
    }
    .stApp {
        background-color: #001f3f;
    }
    .stButton>button {
        background-color: #0074D9;
        color: white;
        font-weight: bold;
        padding: 0.6em 2em;
        border-radius: 8px;
    }
    .stTextArea, .stFileUploader, .stTextInput {
        background-color: #f0f0f5 !important;
        color: #001f3f !important;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -----------------------------
# Header
# -----------------------------
st.markdown("<h1 style='text-align: center; color: white;'>JD vs. CV Compare Tool</h1>", unsafe_allow_html=True)
st.markdown("###")

# -----------------------------
# Upload and JD Input
# -----------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload CV")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"], label_visibility="collapsed")

with col2:
    st.subheader("Paste Job Description")
    job_description = st.text_area("Job Description", height=200, label_visibility="collapsed")

# -----------------------------
# Analyze Button
# -----------------------------
analyze = st.button("Analyze")

# -----------------------------
# Match Score & Feedback
# -----------------------------
if analyze:
    with st.spinner("Analyzing CV against JD..."):
        time.sleep(2) # Simulate processing time

        # Dummy match score for UI demo
        match_score = random.randint(60, 95)

        st.markdown("###")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Match")
            st.progress(match_score)
            st.markdown(f"<h2 style='color:white;'>{match_score}% Match</h2>", unsafe_allow_html=True)

        with col2:
            st.subheader("Feedback")
            st.markdown("**Reasons to select the candidate:**")
            st.markdown("- Matches key skills\n- Experience in relevant domain\n- Educational background aligns")

            st.markdown("**Reasons to reject the candidate:**")
            st.markdown("- Lacks required certifications\n- Limited leadership experience")

# -----------------------------
# Footer
# -----------------------------
st.markdown("---")
st.markdown("<p style='text-align: center; color: white;'>🔍 Powered by Local AI | Built with ❤️ using Streamlit</p>", unsafe_allow_html=True)
