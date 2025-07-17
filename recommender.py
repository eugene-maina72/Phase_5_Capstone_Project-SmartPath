import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

job_profiles_and_abilities = pd.read_csv('data/job_profiles_and_abilities.csv')


def recommend_jobs(user_profile, job_profiles_and_abilities=job_profiles_and_abilities, top_n=5):
  
    """
    Recommends top N jobs based on user's RIASEC scores and education level.
    Filters jobs based on user's selected skills (matched against entries in Skill_1 to Skill_25),
    but does not use skills for cosine similarity.

    Args:
        user_profile (dict): Dictionary with keys 'R', 'I', 'A', 'S', 'E', 'C', 'education_level', and 'skills'.
        job_profiles_and_abilities (pd.DataFrame): Job dataset containing RIASEC, education, and skill columns.
        top_n (int, optional): Number of job recommendations to return. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with top N recommended jobs sorted by similarity score.

    """

    # Extract RIASEC columns
    riasec_cols = ['R', 'I', 'A', 'S', 'E', 'C']


    # Extract skill columns from Skill_1 to Skill_25
    skill_features = [f'Skill_{i}' for i in range(1, 25)]

    # Filter jobs by matching any user skill in any Skill_# column
    if 'skills' in user_profile is not None and user_profile['skills']:
        def has_matching_skill(row):
            return any(skill in row[skill_features].values for skill in user_profile['skills'])
        job_profiles_and_abilities = job_profiles_and_abilities[job_profiles_and_abilities.apply(has_matching_skill, axis=1)]

    # If no jobs remain after filtering, return empty DataFrame with a message column
    if job_profiles_and_abilities.empty:
        return pd.DataFrame({'Message': ["No matching jobs found based on selected skills."]})

    # Normalize RIASEC
    user_riasec = np.array([
        user_profile['R'], user_profile['I'], user_profile['A'],
        user_profile['S'], user_profile['E'], user_profile['C']
    ]).reshape(1, -1)
    user_riasec_normalized = user_riasec/ 7.0

    

    # Normalize education
    user_education_normalized = np.array([user_profile['education_level'] / 12.0]).reshape(1, -1)

    # Combine user vector (only RIASEC + Education)
    weighted_user_vector = np.hstack([
        user_riasec_normalized * 0.7,
        user_education_normalized * 0.3,
        
    ])

    # Prepare job matrix
    job_riasec = job_profiles_and_abilities[riasec_cols].values
    job_education = job_profiles_and_abilities[['Normalized Education Score']].values
    

    weighted_job_matrix = np.hstack([
        job_riasec * 0.7,
        job_education * 0.3,
    ])

    # Compute similarity
    similarity_scores = cosine_similarity(weighted_user_vector, weighted_job_matrix).flatten()
    job_profiles_and_abilities['Similarity Score'] = similarity_scores



    
    # Sorting by similarity score and selecting top N jobs
    recommended_jobs = job_profiles_and_abilities.sort_values(by='Similarity Score', ascending=False).head(top_n)

    # Selecting relevant columns to return
    recommended_jobs = recommended_jobs[['Title', 'Description', 
                                         'Education Category Label',
                                         'Education Level', 
                                          'Preparation Level',
                                            'Similarity Score',
                                            'Normalized Education Score']].style.background_gradient(cmap='YlGn')
    
    # Returning the recommended jobs DataFrame
    
    return recommended_jobs