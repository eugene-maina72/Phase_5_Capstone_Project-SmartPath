import streamlit as st
import pandas as pd
from recommender import kmeans_svd_recommender
import json
from datetime import datetime

# Page title and markdowns
st.set_page_config(page_title="Smart Career Recommender", layout="centered")
st.title("🎯 SmartPath Career Recommender")

st.markdown("""
Welcome to **SmartPath** — your quirky AI buddy for career matchmaking! 🤖✨

This app uses your **RIASEC** personality traits and **education level** to sniff out job roles you might actually enjoy (yes, even Mondays!).

---

### 🧠 What’s RIASEC?
These are 6 personality dimensions that describe what type of work you're wired for:

- **🔧 R: Realistic** — love building, fixing, or working with your hands (think engineers, carpenters)
- **🔬 I: Investigative** — curious about how the world works (think scientists, analysts)
- **🎨 A: Artistic** — expressive and creative (think designers, writers)
- **🫶 S: Social** — helpful and empathetic (think teachers, counselors)
- **💼 E: Enterprising** — love leading or persuading (think sales, entrepreneurs)
- **📊 C: Conventional** — organized and data-driven (think accountants, admins)

---

### 🎓 Education Level
Measured on a scale from **1 to 12**:

| Level| Description                                 |
|------|---------------------------------------------|
|  1   |   Less than High School                     |
|  2   |   High School Diploma or Equivalent         |                   
|  3   |   Post-Secondary Certificate                |            
|  4   |   Some College Courses                      |      
|  5   |   Associate Degree                          |  
|  6   |   Bachelor's Degree                         |   
|  7   |   Post-Baccalaureate's Degree               |             
|  8   |   Master's Degree                           | 
|  9   |   Post-Master's Certificate                 |           
|  10  |   First Professional Degree                 |           
|  11  |   Doctoral Degree                           | 
|  12  |   Post-Doctoral Training                    |


Now go ahead and tweak the sliders to match **your vibe**:
""")

# Load job data
@st.cache_data
def load_data():
    return pd.read_csv("data/job_profiles_with_skills.csv")

job_data = load_data()

# Input sliders
user_profile = {
    "R": st.slider("🔧 R: Realistic", min_value=0, max_value=7, value=4, step=1),
    "I": st.slider("🔬 I: Investigative", min_value=0, max_value=7, value=4, step=1),
    "A": st.slider("🎨 A: Artistic", min_value=0, max_value=7, value=4, step=1),
    "S": st.slider("🫶 S: Social", min_value=0, max_value=7, value=4, step=1),
    "E": st.slider("💼 E: Enterprising", min_value=0, max_value=7, value=4, step=1),
    "C": st.slider("📊 C: Conventional", min_value=0, max_value=7, value=4, step=1),
    "education_level": st.slider("🎓 Education Level (1 = None, 12 = PhD!)", min_value=0, max_value=12, value=8, step=1)
}

if st.button("🚀 Get My Job Matches"):
    # Log user profile
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, **user_profile}
    with open("user_logs.jsonl", "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")

    # Generate recommendations
    recs = kmeans_svd_recommender(user_profile, job_data)
    st.subheader("🧭 Top Career Suggestions Just for You!")
    st.dataframe(recs[['Title', 'Description', 
                        'Education Category Label',
                        'Preparation Level',
                        'Similarity Score'
                            ]].reset_index(drop=True))

 # Show stats for exploration (mock-up preview)
    try:
        df_logs = pd.read_json("user_logs.jsonl", lines=True)
        st.markdown("### 📊 What Users Are Picking")
        st.bar_chart(df_logs[["R", "I", "A", "S", "E", "C"]].mean())
        st.line_chart(df_logs["education_level"].value_counts().sort_index())
        df_logs["hour"] = pd.to_datetime(df_logs["timestamp"]).dt.hour
        st.markdown("### ⏰ User Activity by Hour")
        st.bar_chart(df_logs["hour"].value_counts().sort_index())

        # Export logs to CSV
        df_logs.to_csv("logs/user_logs.csv", index=False)
        st.success("📁 Logs successfully exported to user_logs.csv")

    except Exception as e:
        st.warning("User stats could not be displayed yet. Logs might be empty or malformed.")