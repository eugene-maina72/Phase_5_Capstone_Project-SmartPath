def get_user_profile():

    """    
    Function to collect user profile information including RIASEC scores, education level, and skills.

    Prompts the user to input their RIASEC scores, highest education level, and up to 3 skills they consider strong.
        1. Collects RIASEC scores from the user.
        2. Prompts the user to select their highest education level from a predefined list.
        3. Allows the user to select up to 3 skills they consider their strengths.
        4. Returns a dictionary containing the user's RIASEC scores, education level, and selected skills.
    
    Returns: A dictionary with keys 'RIASEC', 'Education Level', and 'Skills' containing the user's inputs.
    
    """
    # Prompt the user to input their RIASEC scores (Realistic, Investigative, Artistic, Social, Enterprising, Conventional)
    print("Enter your RIASEC Scores (scale of 0–7, separated by commas):")
    print("Format: R, I, A, S, E, C")
    r_i_a_s_e_c = input("Enter scores: ").strip().split(',')

    # Converting the input scores to float; fallback to neutral values if input is invalid
    try:
        r, i, a, s, e, c = [float(score.strip()) for score in r_i_a_s_e_c]
    except ValueError:
        print("Invalid RIASEC input. Using default neutral scores.")
        r, i, a, s, e, c = [4, 4, 4, 4, 4, 4]  ## Default scores if input is invalid

    ## Prompt user to select their highest education level from a list of 12 categories
     
    ## Collect user input for education level
    edu_input = input("Enter number (1–12): ").strip()

    education_map = {
        1: "Less than High School",
        2: "High School Diploma or Equivalent",
        3: "Post-Secondary Certificate",
        4: "Some College Courses",
        5: "Associate Degree",
        6: "Bachelor's Degree",
        7: "Post-Baccalaureate's Degree",
        8: "Master's Degree",
        9: "Post-Master's Certificate",
        10: "First Professional Degree",
        11: "Doctoral Degree",
        12: "Post-Doctoral Training"
    }

    print("\n Select your highest education level:")
    print("1. Less than High School")
    print("2. High School Diplolma or Equivalent")
    print("3. Post-Secondary Certificate")
    print("4. Some College Courses")
    print("5. Associate Degree")
    print("6. Bachelor's Degree")
    print("7. Post-Baccalaureate's Degree")
    print("8. Masters Degree")
    print("9. Post-Master's Certificate")
    print("10. First Professional Degree")
    print("11. Doctoral Degree")
    print("12. Post-Doctoral Training")

   
    ## Try converting education level to integer; default to "Bachelor's Degree" if input is invalid
    try:
        education_level = int(edu_input) <= 12 and int(edu_input) >= 1
    except ValueError:
        education_level = 6  ## Default to Bachelor's Degree
    
    edu_level = print(f'{education_map.get(education_level, "Bachelor's Degree")}')

    ## Present a list of skills and prompt the user to select up to 3 they consider their strengths
    print("\n Select up to 3 skills you consider strong (or press Enter to skip):")
    
    skill_map = {
        "1"  : 'Oral Comprehension',
        "2"  : 'Written Comprehension',
        "3"  : 'Oral Expression',
        "4"  : 'Written Expression',
        "5"  : 'Fluency of Ideas',
        "6"  : 'Originality',
        "7"  : 'Problem Sensitivity',
        "8"  : 'Deductive Reasoning',
        "9"  : 'Inductive Reasoning',
        "10" : 'Information Ordering',
        "11" : 'Category Flexibility',
        "12" : 'Mathematical Reasoning',
        "13" : 'Number Facility',
        "14" : 'Memorization',
        "15" : 'Speed of Closure',
        "16" : 'Flexibility of Closure',
        "17" : 'Perceptual Speed',
        "18" : 'Spatial Orientation',
        "19" : 'Visualization',
        "20" : 'Selective Attention',
        "21" : 'Time Sharing',
        "22" : 'Arm-Hand Steadiness',
        "23" : 'Manual Dexterity',
        "24" : 'Finger Dexterity',
        "25" : 'Control Precision',
        "26" : 'Multilimb Coordination',
        "27" : 'Response Orientation',
        "28" : 'Rate Control',
        "29" : 'Reaction Time',
        "30" : 'Wrist-Finger Speed',
        "31" : 'Speed of Limb Movement',
        "32" : 'Static Strength',
        "33" : 'Explosive Strength',
        "34" : 'Dynamic Strength',
        "35" : 'Trunk Strength',
        "36" : 'Stamina',
        "37" : 'Extent Flexibility',
        "38" : 'Dynamic Flexibility',
        "39" : 'Gross Body Coordination',
        "40" : 'Gross Body Equilibrium',
        "41" : 'Near Vision',
        "42" : 'Far Vision',
        "43" : 'Visual Color Discrimination',
        "44" : 'Night Vision',
        "45" : 'Peripheral Vision',
        "46" : 'Depth Perception',
        "47" : 'Glare Sensitivity',
        "48" : 'Hearing Sensitivity',
        "49" : 'Auditory Attention',
        "50" : 'Sound Localization',
        "51" : 'Speech Recognition',
        "52" : 'Speech Clarity'
    }

    ## Print skill options to the user
    for k, v in skill_map.items():
        print(f"{k}. {v}")
    
    ## Collect and parse the user's skill choices
    skill_input = input("Enter skill numbers separated by commas (e.g., 1,3,5): ").strip()

    ## Converting selected skill numbers to skill names, filtering only valid inputs
    user_skills = []
    if skill_input:
        user_skills = [skill_map[num.strip()] for num in skill_input.split(',') if num.strip() in skill_map]

    ## Final message before returning the user profile
    print("\n Thank you. Generating personalized recommendations...\n")

    user_profile = {
        'R': r, 'I': i, 'A': a, 'S': s, 'E': e, 'C': c,
        'education_level': education_level ,
        'skills': user_skills
                    }
    ## Return a dictionary with user profile data: RIASEC scores, education level, and skills
    return user_profile , edu_level