import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity



job_profiles_with_skills = pd.read_csv('data/job_profiles_with_skills.csv')


def recommend_jobs(user_profile, job_profiles_and_abilities, top_n=5):

    job_profiles_and_abilities = pd.read_csv('data/job_profiles_and_abilities.csv')
    
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
                                            'Normalized Education Score']]
    
    # Returning the recommended jobs DataFrame
    
    return recommended_jobs


def kmeans_svd_recommender(user_profile: dict, job_profile_with_skills=job_profiles_with_skills, n_clusters = 2, svd_components = 15, top_n: int = 5):

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    """
    Recommends top N jobs based on user's RIASEC scores and education level and skills required.
    Takes in a Dataframe, reduces its dimensionality using Truncated SVD to retain 20 of the most important components.
    Clusters the data using K-Means Clustering then calculates cosine similarity of a user profile and the reduced Dataframe RIASEC + Education level
    but does not use skills for cosine similarity.

    Args:
        user_profile (dict): Dictionary with keys 'R', 'I', 'A', 'S', 'E', 'C', 'education_level', and 'skills'.
        job_profiles_and_abilities (pd.DataFrame): Job dataset containing RIASEC, education, and skill columns.
        n_clusters (int) : The number of clusters to be instatiated in the KMeans.Optimized at 3(default)
        svd_components (int): The number of features to be retained after in the truncated SVD algorithm
        top_n (int, optional): Number of job recommendations to return. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with top N recommended jobs sorted by similarity score.

    """
    # Extracting the RIASEC, Normalized Education score and the skill columns for machine learning
    features = ['R', 'I', 'A', 'S', 'E', 'C', 'Normalized Education Score'] + job_profile_with_skills.columns[27:-1].tolist()
    X = job_profile_with_skills[features].fillna(0).values
    
    # Reducing the number of features to improve generalization
    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_reduced = svd.fit_transform(X)

    #instatiating and fitting the model
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_reduced)
    job_profile_with_skills['Cluster'] = labels #Creating a new column for the cluster assignments

    user_vector_raw = np.array([
        user_profile['R']/7, user_profile['I']/7, user_profile['A']/7,
        user_profile['S']/7, user_profile['E']/7, user_profile['C']/7,
        user_profile['education_level']/12
    ] + [0] * (X.shape[1] - 7)).reshape(1, -1)
    user_vector_reduced = svd.transform(user_vector_raw)    # Creating a new user profile similar to the TruncatedSVD array

    # Predicting the User's Cluster assignment and Calculating cosine similarity to determine best job match
    user_cluster = model.predict(user_vector_reduced)[0]
    svd_cluster_jobs = job_profile_with_skills[job_profile_with_skills['Cluster'] == user_cluster].copy()
    sim_scores = cosine_similarity(user_vector_reduced, svd.transform(svd_cluster_jobs[features].fillna(0).values)).flatten()
    svd_cluster_jobs['Similarity Score'] = sim_scores
    
    # Sort values based on cosine similarity and displaying top 5 matches
    svd_top_jobs =svd_cluster_jobs.sort_values(by='Similarity Score', ascending=False).head(top_n)
    return svd_top_jobs[['Title', 'Description', 
                                         'Education Category Label',
                                         'Education Level', 
                                          'Preparation Level',
                                            'Similarity Score',
                                            'Normalized Education Score']]


def kmeans_pca_recommender(user_profile: dict, job_profile_with_skills=job_profiles_with_skills, n_clusters = 2, pca_components = 13, top_n: int = 5):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    """
    Recommends top N jobs based on user's RIASEC scores and education level and skills required.
    Takes in a Dataframe, reduces its dimensionality using PCA to retain 13 of the most important components.
    Clusters the data using K-Means Clustering then calculates cosine similarity of a user profile and the reduced Dataframe RIASEC + Education level
    but does not use skills for cosine similarity.

    Args:
        user_profile (dict): Dictionary with keys 'R', 'I', 'A', 'S', 'E', 'C', 'education_level', and 'skills'.
        job_profiles_and_abilities (pd.DataFrame): Job dataset containing RIASEC, education, and skill columns.
        n_clusters (int) : The number of clusters to be instatiated in the KMeans.Optimized at 3(default)
        svd_components (int): The number of features to be retained after in the truncated SVD algorithm
        top_n (int, optional): Number of job recommendations to return. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with top N recommended jobs sorted by similarity score.

    """
    # Extracting the RIASEC, Normalized Education score and the skill columns for machine learning
    features = ['R', 'I', 'A', 'S', 'E', 'C', 'Normalized Education Score'] + job_profile_with_skills.columns[26:-1].tolist()
    X = job_profile_with_skills[features].fillna(0).values
    
    # Reducing the number of features to improve generalization
    pca= PCA(n_components=pca_components, random_state=42)
    X_pca = pca.fit_transform(X)

    #instatiating and fitting the model
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(X_pca)
    job_profile_with_skills['Cluster'] = labels #Creating a new column for the cluster assignments

    user_vector_raw = np.array([
        user_profile['R']/7, user_profile['I']/7, user_profile['A']/7,
        user_profile['S']/7, user_profile['E']/7, user_profile['C']/7,
        user_profile['education_level']/12
    ] + [0] * (X.shape[1] - 7)).reshape(1, -1)
    user_vector_reduced = pca.transform(user_vector_raw)    # Creating a new user profile similar to the TruncatedSVD array

    # Predicting the User's Cluster assignment and Calculating cosine similarity to determine best job match
    user_cluster = model.predict(user_vector_reduced)[0]
    pca_cluster_jobs = job_profile_with_skills[job_profile_with_skills['Cluster'] == user_cluster].copy()
    sim_scores = cosine_similarity(user_vector_reduced, pca.transform(pca_cluster_jobs[features].fillna(0).values)).flatten()
    pca_cluster_jobs['Similarity Score'] = sim_scores
    
    # Sort values based on cosine similarity and displaying top 5 matches
    pca_top_jobs =pca_cluster_jobs.sort_values(by='Similarity Score', ascending=False).head(top_n)
    return pca_top_jobs[['Title', 'Description', 
                                         'Education Category Label',
                                         'Education Level', 
                                          'Preparation Level',
                                            'Similarity Score',
                                            'Normalized Education Score']]

def optimised_agglomerative_recommender(user_profile: dict, job_profile_with_skills=job_profiles_with_skills, n_clusters=2, svd_components=15, top_n: int = 5):
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    from sklearn.cluster import AgglomerativeClustering
    
    """
    Recommends top N jobs based on user's RIASEC scores and education level and skills required.
    Takes in a DataFrame, reduces its dimensionality using PCA to retain key components,
    clusters the data using Agglomerative Clustering, then calculates cosine similarity between a user profile
    and the reduced DataFrame (RIASEC + Education level).

    Args:
        user_profile (dict): Dictionary with keys 'R', 'I', 'A', 'S', 'E', 'C', 'education_level', and 'skills'.
        job_profile_with_skills (pd.DataFrame): Job dataset containing RIASEC, education, and skill columns.
        n_clusters (int): The number of clusters for Agglomerative Clustering. Defaults to 10.
        pca_components (int): Number of features retained after PCA.
        top_n (int, optional): Number of job recommendations to return. Defaults to 5.

    Returns:
        pd.DataFrame: DataFrame with top N recommended jobs sorted by similarity score.
    """

    # Defining features for machine learning
    features = ['R', 'I', 'A', 'S', 'E', 'C', 'Normalized Education Score'] + job_profile_with_skills.columns[26:-1].tolist()
    X = job_profile_with_skills[features].fillna(0).values

    svd = TruncatedSVD(n_components=svd_components, random_state=42)
    X_svd = svd.fit_transform(X)

    model = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    labels = model.fit_predict(X_svd)
    job_profile_with_skills['Cluster'] = labels

    user_vector_raw = np.array([
        user_profile['R']/7, user_profile['I']/7, user_profile['A']/7,
        user_profile['S']/7, user_profile['E']/7, user_profile['C']/7,
        user_profile['education_level']/12
    ] + [0] * (X.shape[1] - 7)).reshape(1, -1)
    user_vector_reduced = svd.transform(user_vector_raw)

    user_cluster = labels[np.argmax(cosine_similarity(X_svd, user_vector_reduced))]  # find closest cluster
    cluster_jobs = job_profile_with_skills[job_profile_with_skills['Cluster'] == user_cluster].copy()
    sim_scores = cosine_similarity(user_vector_reduced, svd.transform(cluster_jobs[features].fillna(0).values)).flatten()
    cluster_jobs['Similarity Score'] = sim_scores

    top_jobs = cluster_jobs.sort_values(by='Similarity Score', ascending=False).head(top_n)
    return top_jobs[[
        'Title', 'Description', 'Education Category Label',
        'Education Level', 'Preparation Level',
        'Similarity Score', 'Normalized Education Score']]