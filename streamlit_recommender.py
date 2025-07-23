import streamlit as st
import pandas as pd
from recommender import  recommend_jobs, plot_riasec_radar
import json
from datetime import datetime
import os
import numpy as np

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
# Loading data and performing sanity checks
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("data/job_profiles_and_abilities.csv")
        required = ['ONET_Code',
            'R', 'I', 'A', 'S', 'E', 'C',
            'Title', 'Description', 'Education Category Label',
            'Education Level', 'Preparation Level', 'Normalized Education Score'
        ]
        # Warn if columns are missing (prevents downstream errors)
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        return df
    except Exception as e:
        st.error(f"Could not load data: {e}")
        return pd.DataFrame()

job_data = load_data()  # Load job data once

# --- Gather User Input via Sliders ---
user_profile = {
    "R": st.slider("🔧 R: Realistic", min_value=0, max_value=7, value=4, step=1),
    "I": st.slider("🔬 I: Investigative", min_value=0, max_value=7, value=4, step=1),
    "A": st.slider("🎨 A: Artistic", min_value=0, max_value=7, value=4, step=1),
    "S": st.slider("🫶 S: Social", min_value=0, max_value=7, value=4, step=1),
    "E": st.slider("💼 E: Enterprising", min_value=0, max_value=7, value=4, step=1),
    "C": st.slider("📊 C: Conventional", min_value=0, max_value=7, value=4, step=1),
    "education_level": st.slider(
        "🎓 Education Level (1 = None, 12 = PhD!)", min_value=0, max_value=12, value=8, step=1
    )
}

# --- Ensure Data Directory and Logging Files Exist ---
DATA_DIR = "logs"
os.makedirs(DATA_DIR, exist_ok=True)
LOG_PATH = os.path.join(DATA_DIR, "user_logs.jsonl")
INTERACTION_PATH = os.path.join(DATA_DIR, "user_interactions.csv")
FEEDBACK_PATH = os.path.join(DATA_DIR, "user_feedback.csv")

# --- Create CSVs with headers if not present ---
for path, columns in [
    (INTERACTION_PATH, ["timestamp", "user_id", "job_id", "job_title"]),
    (FEEDBACK_PATH, ["timestamp", "user_id", "star_rating", "comment"]),
]:
    if not os.path.exists(path):
        pd.DataFrame(columns=columns).to_csv(path, index=False)

# --- Set up session state for persistent UI experience ---
if "recommended_jobs" not in st.session_state:
    st.session_state["recommended_jobs"] = []
if "show_recommendations" not in st.session_state:
    st.session_state["show_recommendations"] = False
if "selected_job_index" not in st.session_state:
    st.session_state["selected_job_index"] = 0

# --- MAIN BUTTON: Get Recommendations ---
if st.button("🚀 Get My Job Matches"):
    # Log the user's input for stats/audit
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = {"timestamp": timestamp, **user_profile}
    with open(LOG_PATH, "a") as log_file:
        log_file.write(json.dumps(log_entry) + "\n")
    # Run the recommendation model
    recs = recommend_jobs(user_profile, job_data)
    # Store in session for persistent user experience
    st.session_state["recommended_jobs"] = recs.to_dict('records')
    st.session_state["show_recommendations"] = True
    st.session_state["selected_job_index"] = 0

# --- SHOW RECOMMENDATIONS AND RIASEC COMPARISON ---
if st.session_state.get("show_recommendations", False) and st.session_state.get("recommended_jobs", []):
    st.subheader("🧭 Top Career Suggestions Just for You!")
    recs_df = pd.DataFrame(st.session_state["recommended_jobs"])
    # Show the top recommendations as a table
    st.dataframe(recs_df[['Title', 'Description', 
                          'Education Category Label',
                          'Preparation Level',
                          'Similarity Score']].reset_index(drop=True))

    # --- Let user pick which job to compare to their profile ---
    job_titles = recs_df['Title'].tolist()
    selected_title = st.selectbox(
        "Select a job to compare:",
        job_titles,
        index=st.session_state.get("selected_job_index", 0),
        key="job_selectbox"
    )
    # Track which job is currently selected
    selected_index = job_titles.index(selected_title)
    st.session_state["selected_job_index"] = selected_index
    selected_job = recs_df.iloc[selected_index]

    # --- Ensure RIASEC columns are numeric ---
    for col in ['R', 'I', 'A', 'S', 'E', 'C']:
        if col in recs_df.columns:
            recs_df[col] = pd.to_numeric(recs_df[col], errors='coerce').fillna(0)

    # --- Gather scores for chart ---
    job_row = job_data[job_data['Title'] == selected_title].iloc[0]
    job_scores = [np.round(float(job_row[l]*7), 1) if l in job_row and pd.notnull(job_row[l]) else 0.0 for l in ['R', 'I', 'A', 'S', 'E', 'C']]
    user_scores = [float(user_profile.get(l, 0)) for l in ['R', 'I', 'A', 'S', 'E', 'C']]

    # DEBUG: See scores in Streamlit sidebar (optional) ---
    #st.sidebar.write("user_scores:", user_scores)
    #st.sidebar.write("job_scores:", job_scores)
    #st.sidebar.write("selected_job:", selected_job)

    # --- Draw radar chart of user vs. job profile ---
    fig = plot_riasec_radar(user_scores, job_scores, job_title=selected_title)
    st.pyplot(fig)

    # --- Allow user to reset UI without reloading ---
    if st.button("🔄 Reset Recommendations"):
        st.session_state["show_recommendations"] = False
        st.session_state["recommended_jobs"] = []
        st.session_state["selected_job_index"] = 0

    # --- Show user stats and trends (optional, for admin/UX) ---
    try:
        df_logs = pd.read_json(LOG_PATH, lines=True)
        st.markdown("### 📊 What Users Are Picking")
        st.bar_chart(df_logs[["R", "I", "A", "S", "E", "C"]].mean())
        st.markdown("### What are other Users Education levels?")
        st.line_chart(df_logs["education_level"].value_counts().sort_index())
        df_logs["hour"] = pd.to_datetime(df_logs["timestamp"]).dt.hour
        st.markdown("### ⏰ User Activity by Hour")
        st.bar_chart(df_logs["hour"].value_counts().sort_index())
        df_logs.to_csv(os.path.join(DATA_DIR, "user_logs.csv"), index=False)
    except Exception as e:
        st.warning("User stats could not be displayed yet. Logs might be empty or malformed.")

    # --- Let user express interest in a job (bookmark) ---
    st.write("Click 'Show Interest' to bookmark jobs. Don’t forget to leave your feedback at the bottom!")
    for idx, row in recs_df.iterrows():
        with st.expander(f"{row['Title']}"):
            st.markdown(f"**Similarity Score:** {row['Similarity Score']:.2f}")
            # Fetch full job row from job_data by unique ID
        if 'ONET_Code' in row and row['ONET_Code'] in job_data['ONET_Code'].values:
            job_full = job_data[job_data['ONET_Code'] == row['ONET_Code']].iloc[0]
        else:
            job_full = job_data[job_data['Title'] == row['Title']].iloc[0]  # fallback
        
        # Display RIASEC from full job row
        riasec_dict = {k: round((float(job_full[k]) * 7), 2) for k in ['R', 'I', 'A', 'S', 'E', 'C']}
        st.markdown("**RIASEC Profile:** " + ", ".join([f" {k}: {v} " for k, v in riasec_dict.items()]))
        if st.button(f"Show Interest in {row['Title']}", key=f"interest_{idx}"):

            # Log the interest
            log_data = {
                "timestamp": datetime.now(),
                "user_id": st.session_state.get("user_id", "anonymous"),
                "job_id": job_full['ONET_Code'],
                "job_title": job_full['Title'],
                
            }
            pd.DataFrame([log_data]).to_csv(INTERACTION_PATH, mode="a", header=False, index=False)
            st.success(f"Interest in '{job_full['Title']}' logged! 👍")

# --- Feedback Section ---
st.header("Rate Your Recommendations")
st.write("Give us your honest feedback – help us make SmartPath smarter!")

rating = st.slider("How would you rate your recommendations?", 1, 5, 3, format="%d ⭐")
feedback_text = st.text_area(
    "Any comments or suggestions? (Optional)", 
    help="Share what you liked, hated, or wish was different!"
)

if st.button("Submit Feedback"):
    feedback_entry = {
        "timestamp": datetime.now(),
        "user_id": st.session_state.get("user_id", "anonymous"),
        "star_rating": rating,
        "comment": feedback_text.strip()
    }
    pd.DataFrame([feedback_entry]).to_csv(FEEDBACK_PATH, mode="a", header=False, index=False)
    st.success("Thanks for your feedback! 🙏 Your insights make SmartPath even smarter.")

st.caption("All feedback is logged anonymously and will be used to improve future recommendations and model performance.")

# --- Get admin password securely from Streamlit secrets ---
ADMIN_PASSWORD = st.secrets["admin"]["password"]

with st.sidebar:
    st.markdown("### Admin Log Download")
    admin_pw = st.text_input("Enter admin password", type="password")
    if admin_pw == ADMIN_PASSWORD:
        log_files = os.listdir("logs")
        for fname in log_files:
            fpath = os.path.join("logs", fname)
            if os.path.isfile(fpath):
                with open(fpath, "rb") as f:
                    st.download_button(
                        f"Download {fname}",
                        f,
                        file_name=fname
                    )
        st.success("Logs ready for download.")
    elif admin_pw:
        st.error("Incorrect password.")