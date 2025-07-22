# --- Required Libraries ---
import streamlit as st
from recommender_engine import final_kmeans_recommender
import numpy as np
import pandas as pd
import altair as alt

# --- Page Configuration ---
st.set_page_config(
    page_title="SmartPath Career Recommender",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Font Styling ---
st.markdown("""
    <style>
        html, body, [class*="css"] {
            font-size: 17px !important;
        }
        .stSlider > div > div {
            font-size: 17px !important;
        }
        label, .stSelectbox label, .stMultiSelect label {
            font-size: 17px !important;
            font-weight: 500 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Make the Button Bold & Tab-like ---
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #0e76a8;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.75em 1.5em;
        font-size: 1.1em;
        border: none;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    div.stButton > button:first-child:hover {
        background-color: #095e88;
        transform: scale(1.03);
    }
    </style>
""", unsafe_allow_html=True)

# --- Main Header (Mobile-Responsive) ---
st.markdown("""
    <style>
        @media (max-width: 768px) {
            .header-container h1 {
                font-size: 1.5rem !important;
            }
            .header-container p {
                font-size: 0.95rem !important;
            }
        }
    </style>
    <div class="header-container" style='text-align:center; padding: 1rem; background-color: #003262; border-radius: 10px;'>
        <h1 style='color:white;'>🔍 SmartPath Career Recommender</h1>
        <p style='color:white;'>Welcome to SmartPath, your personalized career assistant based on the RIASEC model, education level and your skills.<br>
        <em>Powered by the RIASEC models and real job market data.</em></p>
    </div>
""", unsafe_allow_html=True)

# --- Input Form ---
with st.form("user_profile_form"):
    st.markdown("""
    ### 🧠 <span style='font-weight:bold; font-size: 1.5rem;'>Enter Your RIASEC Scores (0–7)</span>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        r = st.slider("Realistic (R)", 0.0, 7.0, 4.0)
        i = st.slider("Investigative (I)", 0.0, 7.0, 4.0)
    with col2:
        a = st.slider("Artistic (A)", 0.0, 7.0, 4.0)
        s = st.slider("Social (S)", 0.0, 7.0, 4.0)
    with col3:
        e = st.slider("Enterprising (E)", 0.0, 7.0, 4.0)
        c = st.slider("Conventional (C)", 0.0, 7.0, 4.0)

    st.markdown("""
    ### 🎓 <span style='font-weight:bold; font-size: 1.5rem;'>Highest Education Level</span>
    <span style='font-size: 1rem; color: #444;'>Choose the most accurate description of your completed education.</span>
    """, unsafe_allow_html=True)
    
    edu_level = st.selectbox("Select your highest level of education", [
        "Less than High School", "High School Diploma or Equivalent", "Some College Courses",
        "Associate Degree", "Bachelor's Degree", "Master's Degree",
        "Doctoral or Professional Degree", "Post-Doctoral Training"
    ], index=4)

    st.markdown("""
    ### 🛠️ <span style='font-weight:bold; font-size: 1.5rem;'>Strong Skills (Select up to 10)</span>
    <span style='font-size: 1rem; color: #444;'>Search or scroll to select the skills you're confident in.</span>
    """, unsafe_allow_html=True)

    skill_options = [
        "Data Analysis", "Communication", "Problem Solving", "Project Management", "Creativity",
        "Critical Thinking", "Leadership", "Teamwork", "Technical Writing", "Machine Learning",
        "SQL", "Python", "R", "Tableau", "Excel", "Public Speaking", "Negotiation", "Sales",
        "Graphic Design", "Customer Service", "Financial Literacy", "Coding", "UX/UI Design",
        "Operations", "Time Management", "On-the-Job Training", "Time Management", "Oral Comprehension", "Written Comprehension", "Oral Expression", "Written Expression",
        "Fluency of Ideas", "Originality", "Problem Sensitivity", "Deductive Reasoning",
        "Inductive Reasoning", "Information Ordering", "Category Flexibility", "Mathematical Reasoning",
        "Number Facility", "Memorization", "Speed of Closure", "Flexibility of Closure",
        "Perceptual Speed", "Spatial Orientation", "Visualization", "Selective Attention",
        "Time Sharing", "Arm-Hand Steadiness", "Manual Dexterity", "Finger Dexterity",
        "Control Precision", "Multilimb Coordination", "Response Orientation", "Rate Control",
        "Reaction Time", "Wrist-Finger Speed", "Speed of Limb Movement", "Static Strength",
        "Explosive Strength", "Dynamic Strength", "Trunk Strength", "Stamina",
        "Extent Flexibility", "Dynamic Flexibility", "Gross Body Coordination", "Gross Body Equilibrium",
        "Near Vision", "Far Vision", "Visual Color Discrimination", "Night Vision",
        "Peripheral Vision", "Depth Perception", "Glare Sensitivity", "Hearing Sensitivity",
        "Auditory Attention", "Sound Localization", "Speech Recognition", "Speech Clarity"
    ]
    selected_skills = st.multiselect("Select your top skills", skill_options, max_selections=10)

    submitted = st.form_submit_button("🚀 Get Career Recommendations")

# --- Output Section ---
if submitted:
    st.info("⏳ Generating your personalized career matches...")

    user_profile = {
        'R': r, 'I': i, 'A': a, 'S': s, 'E': e, 'C': c,
        'education_level': edu_level,
        'skills': selected_skills
    }

    try:
        results = final_kmeans_recommender(user_profile)
    except Exception as e:
        st.error("⚠️ An error occurred while generating recommendations.")
        st.exception(e)
    else:
        if results.empty:
            st.warning("No matching jobs found. Try adjusting your input.")
        else:
            st.success("🎯 Recommendations ready!")

            st.markdown("### 📌 Top Job Matches")
            st.dataframe(results, use_container_width=True)

            st.caption("✅ Recommendations are scored based on RIASEC fit, education match, and skill alignment.")

            # --- Clean, Interactive Altair Chart ---
            st.markdown("### 📊 Breakdown of Top 5 Recommendation Scores")
            top5 = results.head(5).copy()
            melted = pd.melt(
                top5,
                id_vars=["Title"],
                value_vars=["User RIASEC Similarity", "Normalized Education Score", "User Skill Similarity"],
                var_name="Metric",
                value_name="Score"
            )

            chart = alt.Chart(melted).mark_bar().encode(
                x=alt.X("Score:Q", stack="zero", title="Score"),
                y=alt.Y("Title:N", sort='-x', title="Job Title"),
                color=alt.Color("Metric:N", scale=alt.Scale(scheme="tableau20")),
                tooltip=["Title", "Metric", "Score"]
            ).properties(
                width="container",
                height=400
            ).configure_axis(
                labelFontSize=14,
                titleFontSize=16
            ).configure_legend(
                labelFontSize=13,
                titleFontSize=14
            )

            st.altair_chart(chart, use_container_width=True)

# --- Footer ---
st.markdown("""
    <style>
        @media (max-width: 768px) {
            .footer { font-size: 0.8rem !important; }
        }
    </style>
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div class="footer" style='text-align: center; font-size: 0.9rem; color: gray;'>
        &copy; 2025 <strong>SmartPath</strong> | Developed by Allan Ofula | Moringa School Capstone Project
    </div>
""", unsafe_allow_html=True)
