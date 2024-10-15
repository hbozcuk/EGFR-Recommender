# -*- coding: utf-8 -*-
"""
Created on Tue Sep 24 15:39:19 2024

@author: hbozcuk
"""

# streamlit_app.py

import streamlit as st
import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from gymnasium import spaces
from stable_baselines3.dqn.policies import DQNPolicy
import os

# Get the port from the environment variable (default to 8501 if not set)
port = int(os.environ.get('PORT', 8501))

# Set the Streamlit server port and address
st.set_page_config(layout="wide")  # Optional: Set the page layout

# Define observation_space and action_space
number_of_features = 10  # Adjust based on your actual number of features
observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(number_of_features,), dtype=np.float32)
action_space = spaces.Discrete(4)

def dummy_schedule(_):
    return 1e-4  # Learning rate schedule isn't used during inference

# Recreate the policy network
policy = DQNPolicy(
    observation_space=observation_space,
    action_space=action_space,
    lr_schedule=dummy_schedule,
    net_arch=[64, 64]  # Use the same architecture as during training
)

# Load the policy parameters
policy.load_state_dict(torch.load("dqn_policy.pth", map_location=torch.device('cpu')))

# Load the saved scaler using joblib
scaler = joblib.load("scaler.pkl")  # Ensure scaler.pkl is in the same directory

# Boltzmann action selection function with tau=8
def boltzmann_action_selection(q_values, tau=8.0):
    # Ensure no NaN values in Q-values
    if np.isnan(q_values).any():
        raise ValueError("Q-values contain NaN. Check the model outputs.")
    
    # Subtract the max Q-value for numerical stability
    q_values_stable = q_values - np.max(q_values)
    
    # Apply Boltzmann exploration (softmax with temperature tau)
    exp_q_values = np.exp(q_values_stable / tau)
    probabilities = exp_q_values / np.sum(exp_q_values)

    # Select an action based on the probabilities
    return np.random.choice(range(len(q_values)), p=probabilities)

with st.sidebar:
    add_subheader = st.subheader("Reinforcement Learning based advisory system to guide initial EGFR TKI treatment for EGFR mutant, TKI naive, NSCLC cases.")
    add_text = st.write("This is an AI application using Reinforcement Learning to guide treatment in patients with advanced, EGFR mutant NSCLC. Developed as an experimental tool for medical oncologists by Hakan Şat Bozcuk, MD, using data from 300+ EGFR mutant advanced NSCLC patients. This app aims to maximize progression-free survival with initial TKI usage, in TKI naive patients.")
    from PIL import Image
    img = Image.open("image.jpg")
    st.image(img, width=300, caption="AI recommending treatment for NSCLC (Image by DALL-E)")

# Streamlit web interface
st.title("EGFR Mutant NSCLC Treatment Advisory System")

# Collect patient features via Streamlit input widgets
gender = st.selectbox('Gender', ('Female', 'Male'))  # Gender: Male or Female
ecog_cat = st.selectbox('ECOG score', ('0 or 1', '2 to 4'))  # ECOG PS from 0 to 4
mutation_status = st.selectbox('EGFR mutation status', ('Exon 19 mutation only', 'Other EGFR mutations'))  # Mutation status: exon 19 versus others
number_mets = st.selectbox('Number of metastases', ('1 to 3', '4 or more'))  # Number of metastases
bone_or_liver_met = st.selectbox('Bone or liver metastases', ('Absent', 'Present'))  # Presence of bone or liver metastases
brain_met = st.selectbox('Brain metastases', ('Absent', 'Present'))  # Presence of brain metastases
comorbidity = st.selectbox('Comorbidities', ('Absent', 'Present'))  # Presence of comorbidities
smoking_status = st.selectbox('Smoking status', ('Never smoker', 'Ever smoker'))  # Smoking history
previous_treatment = st.selectbox('Has this patient received prior chemotherapy?', ('Yes', 'No'))  # Any treatment before?

age = st.slider('Age', 18, 90, 60)  # Age: slider from 18 to 90
log_age = np.log(age)  # Apply log transformation to age

# Input for neutrophil and lymphocyte counts
neutrophil = st.text_input("Neutrophil count (x1000/mm3)")  # Neutrophil count
lymphocyte = st.text_input("Lymphocyte count (x1000/mm3)")  # Lymphocyte count

# Only perform validation if both inputs are not empty
if neutrophil and lymphocyte:
    try:
        neutrophil = float(neutrophil)
        lymphocyte = float(lymphocyte)

        # Ensure lymphocyte is not zero to avoid division by zero error
        if lymphocyte > 0:
            log_nlr = np.log(neutrophil / lymphocyte)  # Logarithmic transformation
        else:
            st.error("Lymphocyte count must be greater than 0.")
            log_nlr = None
    except ValueError:
        st.error("Please enter valid numeric values for neutrophil and lymphocyte counts.")
        log_nlr = None
else:
    log_nlr = None  # Avoids any warning if input is not provided yet

# Process categorical inputs into numerical values (assuming 0/1 encoding)
gender_value = 1 if gender == 'Male' else 0
ecog_cat_value = 1 if ecog_cat == '2 to 4' else 0
mutation_status_value = 1 if mutation_status == 'Other EGFR mutations' else 0
number_mets_value = 1 if number_mets == '4 or more' else 0
bone_or_liver_met_value = 1 if bone_or_liver_met == 'Present' else 0
brain_met_value = 1 if brain_met == 'Present' else 0
comorbidity_value = 1 if comorbidity == 'Present' else 0
smoking_status_value = 1 if smoking_status == 'Ever smoker' else 0
previous_treatment_value = 1 if previous_treatment == 'Yes' else 0

# Create the input features array for the model
patient_features = [log_age, log_nlr, gender_value, ecog_cat_value, mutation_status_value, number_mets_value,
                    bone_or_liver_met_value, brain_met_value, comorbidity_value, smoking_status_value]

# Normalize features using the loaded scaler
scaled_features = scaler.transform([patient_features])

# Define a function to recommend treatments based on patient features and previous treatment status
def recommend_treatment(patient_features, previous_treatment_value):
    # Convert the patient features to a PyTorch tensor
    state_tensor = torch.tensor(patient_features, dtype=torch.float32).unsqueeze(0)
    
    # Get the Q-values for the patient features (state)
    q_values = policy.q_net(state_tensor).cpu().detach().numpy().flatten()
    
    # If previous treatment exists, modify rewards for actions 2 and 3
    if previous_treatment_value == 1:
        q_values[2] -= 20  # Penalize action 2 slightly
        q_values[3] += 20  # Strongly favor action 3
        valid_actions = [2, 3]
        q_values = q_values[valid_actions]  # Select only valid actions (2, 3)
        
        # Use Boltzmann exploration for valid actions
        selected_action_idx = boltzmann_action_selection(q_values, tau=8)  # Index of valid action (0 or 1)
        
        # Map the selected index back to the original action (2 or 3)
        recommended_action = valid_actions[selected_action_idx]
        
        # The second-best action is simply the other valid action
        second_best_action = valid_actions[1 - selected_action_idx]  # Pick the other action
    else:
        # If previous treatment = 0, favor action 1 more and penalize action 0
        if previous_treatment_value == 0:
            q_values[0] -= 20  # Penalize action 0
            q_values[1] += 15  # Favor action 1
            q_values[2] += 8   # Slightly favor action 2
            q_values[3] += 20  # Strongly favor action 3
        
        # Use Boltzmann exploration with tau=8 for action selection among all actions (0, 1, 2, 3)
        recommended_action = boltzmann_action_selection(q_values, tau=8)
        recommended_action = int(recommended_action)  # Convert to Python int
        # Get the second-best action
        top_two_indices = np.argsort(q_values)[-2:]  # Top 2 Q-values
        second_best_action = top_two_indices[0] if top_two_indices[1] == recommended_action else top_two_indices[1]

    return recommended_action, second_best_action

# Predict the recommended treatment when the user clicks the button
if st.button('Get Treatment Recommendation', key='recommend_button'):
    recommended_action, second_best_action = recommend_treatment(scaled_features.flatten(), previous_treatment_value)
    
    # Display the best treatment recommendation
    if recommended_action == 0:
        st.success("Reinforcement Learning Recommended Treatment (Best): 1st line, 1st generation TKI")
    elif recommended_action == 1:
        st.success("Reinforcement Learning Recommended Treatment (Best): 1st line, 2nd or higher generation TKI")
    elif recommended_action == 2:
        st.success("Reinforcement Learning Recommended Treatment (Best): 2nd or later Line, 1st generation TKI")
    elif recommended_action == 3:
        st.success("Reinforcement Learning Recommended Treatment (Best): 2nd or later line, 2nd or higher generation TKI")
    
    # Display the second-best treatment recommendation
    if second_best_action == 0:
        st.info("Reinforcement Learning Recommended Treatment (Second Best): 1st line, 1st generation TKI")
    elif second_best_action == 1:
        st.info("Reinforcement Learning Recommended Treatment (Second Best): 1st line, 2nd or higher generation TKI")
    elif second_best_action == 2:
        st.info("Reinforcement Learning Recommended Treatment (Second Best): 2nd or later Line, 1st generation TKI")
    elif second_best_action == 3:
        st.info("Reinforcement Learning Recommended Treatment (Second Best): 2nd or later line, 2nd or higher generation TKI")
