import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Load cleaned job profiles data
job_profiles_clean = pd.read_csv("data/job_profiles_clean.csv")

# Load the trained KMeans model (if needed)
with open("models/kmeans_model.pkl", "rb") as f:
    kmeans_model = pickle.load(f)

# Main recommendation function
def final_kmeans_recommender(user_profile):
    global job_profiles_clean  # <- Optional but makes it clear you're using the loaded one

    # Copy data to avoid modifying original
    job_df = job_profiles_clean.copy()

    # --- RIASEC similarity ---
    riasec_cols = ['R', 'I', 'A', 'S', 'E', 'C']
    user_vector = np.array([
        user_profile['R'], user_profile['I'], user_profile['A'],
        user_profile['S'], user_profile['E'], user_profile['C']
    ]).reshape(1, -1)

    job_vectors = job_df[riasec_cols].values
    job_df['User RIASEC Similarity'] = cosine_similarity(user_vector, job_vectors)[0]

    # --- Education normalization ---
    edu_mapping = {
        "Less than High School": 1,
        "High School Diploma or Equivalent": 2,
        "Some College Courses": 3,
        "Associate Degree": 4,
        "Bachelor's Degree": 5,
        "Master's Degree": 6,
        "Doctoral or Professional Degree": 7,
        "Post-Doctoral Training": 8
    }
    user_edu_score = edu_mapping.get(user_profile['education_level'], 1)
    max_edu_score = 8
    job_df['Normalized Education Score'] = job_df['Education Category'].apply(lambda x: min(x / max_edu_score, 1.0))

    # --- Skills ---
    skill_cols = [col for col in job_df.columns if col.startswith("Skill List_")]
    user_skills = user_profile.get('skills', [])
    user_skill_vector = np.zeros((1, len(skill_cols)))

    for i, skill in enumerate(skill_cols):
        skill_name = skill.replace("Skill List_", "").lower()
        if any(skill_name in s.lower() for s in user_skills):
            user_skill_vector[0][i] = 1

    job_skill_matrix = job_df[skill_cols].fillna(0).values
    job_df['User Skill Similarity'] = cosine_similarity(user_skill_vector, job_skill_matrix)[0]

    # --- Final Score ---
    job_df['Hybrid Recommendation Score'] = (
        job_df['User RIASEC Similarity'] +
        job_df['Normalized Education Score'] +
        job_df['User Skill Similarity']
    )

    # --- Top 10 matches ---
    top_matches = job_df.sort_values('Hybrid Recommendation Score', ascending=False).head(10)

    return top_matches[[
        'Title', 'Description', 'Education Level', 'Preparation Level',
        'Education Category Label', 'Hybrid Recommendation Score',
        'User RIASEC Similarity', 'Normalized Education Score', 'User Skill Similarity'
    ]]
