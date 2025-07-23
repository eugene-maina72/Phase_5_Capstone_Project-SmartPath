# Phase_5_Capstone_Project-SmartPath

<img src = images\smart_path.png alt = 'Smart Path' width= 300>

 **SMART PATH** : Personalized Career Recommendation System.

*****

## Introduction / Background

In today’s global economy, the job market is evolving rapidly due to technological advancements, automation, and shifting skill demands. Meanwhile, students and job seekers—especially youth in underserved areas—struggle to make informed career decisions due to limited access to personalized, up-to-date guidance.
Traditional career counseling services are often generic, inaccessible, or disconnected from real-world occupational data. As a result, many individuals pursue careers misaligned with their potential, leading to underemployment, poor job satisfaction, and wasted educational resources.
SmartPath aims to address this gap by leveraging data from the O*NET occupational database to offer intelligent, personalized career recommendations based on a user's skills, interests, and education level.

## Project Goal / Objective

* SmartPath aims to build a personalized, data-driven career recommendation system that:

 - Aligns individual users with career paths based on their unique skills, interests, and educational background.
 - Recommends alternative or adjacent careers that fit their profile.
 - Provides insights into job requirements, skill gaps, and occupational attributes.
 - Empowers users to make confident, informed career decisions using structured occupational data.

## Data Sources

-**O*NET 29.3 Database**.Contains detailed information on occupations, including required skills, knowledge areas, work context, interests (RIASEC), education levels, and more.
-There are several datasets like Education, training and experience, interests, related occupations, skills, tools used, technology skills, work styles, work values and occupation data

*****

## **SMARTPATH APP**
[SmartPath](https://smartpath.streamlit.app/)

## Table of Contents

- [About SmartPath](#about-smartpath)
- [How It Works](#how-it-works)
- [Usage](#usage)
- [Running the App](#running-the-app)
- [Feedback & Contribution](#feedback--contribution)
- [Acknowledgements](#acknowledgements)

*****

## About SmartPath

SmartPath is built as a capstone project to:

- Match users to jobs using the RIASEC personality model (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)
- Leverage users' education levels to further personalize recommendations
- Show not just *what* jobs match, but *why* (including interactive RIASEC radar charts)
- Log user choices and feedback for ongoing improvement

*****

## How It Works

1. **User Inputs**: Users set their RIASEC scores (sliders, 0-7 each) and education level (1-12 scale).
2. **Recommendation Engine**: The app matches users to jobs using:
   - Cosine similarity between user/job RIASEC vectors
   - Education level normalization
   - Advanced clustering algorithms (PCA, SVD, KMeans, Agglomerative) in `recommender.py`
3. **Results**: Top job matches are shown, along with similarity scores, education requirements, and descriptions.
4. **RIASEC Comparison**: Users can select any suggested job to compare their own RIASEC radar chart with that job's profile.
5. **Bookmarking/Feedback**: Users can bookmark jobs ("Show Interest") and rate their recommendations. All interactions are logged anonymously.

*****

## Usage

- Run the app locally or deploy to [Streamlit Cloud](https://streamlit.io/cloud).
- Set your RIASEC profile and education level.
- Click **Get My Job Matches**.
- Browse and compare jobs; bookmark those that interest you.
- View your activity, stats, and previously bookmarked jobs.
- Leave feedback to help us improve SmartPath.

*****

## Running the App

### **Locally**

1. Clone this repo:

   ```bash
   git clone https://github.com/eugene-maina72/Phase_5_Capstone_Project-SmartPath.git
   cd Phase_5_Capstone_Project-SmartPath
    ```

2. Install Dependencies

 * Choose one of the following methods, based on your environment:

   - Using pip (recommended for most users and Streamlit Cloud):
     

    ```bash

    pip install -r requirements.txt
    ```

   - Using conda (if you prefer Anaconda/Miniconda):

    ```bash

    conda env create -f environment.yml
    conda activate smartpath-env
    ```

3. Run the Streamlit App

* Launch the app from your project directory:

   ```bash

   streamlit run streamlit_recommender.py
   ```
### On the cloud

* You can access the app [here](https://smartpath.streamlit.app/) 

## Feedback & Contribution

- Open issues or pull requests for bug reports and feature ideas.

- Contributions are welcome—help make SmartPath smarter for everyone!

## Acknowledgements

We want to express our sincere gratitude to:

- Moringa School – for providing the learning foundation and project framework.
- O*NET (Occupational Information Network) – for the rich job dataset that powers this recommendation engine.
- Career Development Theorists – especially John Holland, for the RIASEC model.
- Our instructors, Mildred Jepkosgei, Antonny Muiko and Brian Chacha from Moringa School, for mentorship and support in fostering innovative talent.
- This work reflects a growing commitment to applying data science in empowering youth, career clarity, and digital transformation in Africa.

### Authors

- Rachael Nyawira

  - Kenya | Data Scientist | Passionate about using data to transform lives.
     - Email | GitHub | LinkedIn

- Beryl Okelo

    - Kenya | Data Scientist | Passionate about using data to transform lives
        - Email | GitHub | LinkedIn

- Beth Nyambura

    - Kenya | Data Scientist | Passionate about using data to transform lives
        - Email | GitHub | LinkedIn

- Allan Ofula

    - Kenya | Data Science Associate | Youth Advocate | Developer of SmartPath | Passionate about using data to transform lives
        - [Email](mailto:ofulaallan@gmail.com) | GitHub | LinkedIn

- Eugene Maina

    - Kenya | Data Scientist | Developer of SmartPath | Passionate about using data to transform lives.
        - [Email](mailto:eugenemaina72@gmail.com) | [GitHub](https://github.com/eugene-maina72) | LinkedIn

#### Mentors, Reviewers, and Supporters:

- Brian Chacha
- Mildred Jepkosgei
- Antonny Muiko

*“Empowering smart paths, one data insight at a time.”*

```markdown
The contents of the repo are:
.
├── streamlit_recommender.py      # Main Streamlit UI
├── recommender.py                # Recommender engine, clustering, radar chart
├── data/
│   └── job_profiles_and_abilities.csv
├──images
├── requirements.txt
├── Non-Technical_Presentation.pdf
├── SmartPath.ipynb               #Jupyter Notebook
├── logs/                         # User logs, feedback, bookmarks (created automatically)
│   ├── user_logs.jsonl
│   ├── user_interactions.csv
│   └── user_feedback.csv
└── README.md

The dependencies are:
            -Numpy
            -Pandas
            -sklearn
            -matplotlib
            -seaborn
            -streamlit
            -hdbscan
```
