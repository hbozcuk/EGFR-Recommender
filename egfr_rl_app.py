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

with st.sidebar:
    add_subheader = st.subheader("Reinforcement Learning based recommender system to get EGFR TKI treatment feedback for EGFR mutant NSCLC cases.")
    add_text = st.write("This is an AI application using Reinforcement Learning to guide treatment in patients with advanced, EGFR mutant NSCLC. Developed as an experimental tool for medical oncologists by Hakan Şat Bozcuk, MD.")
    from PIL import Image
    img = Image.open("image.jpg")
    st.image(img, width=300, caption="AI recommending treatment for NSCLC (Image by DALL-E)")

# Define a function to recommend a treatment based on patient features
def recommend_treatment(patient_features):
    # Convert the patient features to a PyTorch tensor
    state_tensor = torch.tensor(patient_features, dtype=torch.float32).unsqueeze(0)
    
    # Get the Q-values for the patient features (state)
    q_values = policy.q_net(state_tensor)
    
    # Select the action with the highest Q-value
    recommended_action = torch.argmax(q_values).item()
    
    return recommended_action

# Streamlit web interface
st.title("EGFR Mutant NSCLC Treatment Advisory System")

# Collect patient features via Streamlit input widgets
gender = st.selectbox('Gender', ('Female', 'Male'))  # Gender: Male or Female
ecog_cat = st.selectbox('ECOG score', ('0 or 1', '2 to 4'))  # ECOG PS from 0 to 4
mutation_status = st.selectbox('EGFR mutation status', ('Exon 19 mutation only', 'Other EGFR mutations'))  # Mutation status: exon 19 versus others
number_mets = st.selectbox('Number of metastases', ('1 to 3', '4 or more')) #Number of metastatic 
bone_or_liver_met = st.selectbox('Bone or liver metastases', ('Absent', 'Present')) #Presence of bone or liver metastases
brain_met = st.selectbox('Brain metastases', ('Absent', 'Present')) #Presence of brain metastases
comorbidity = st.selectbox('Comorbidities', ('Absent', 'Present')) #Presence of comorbidities
smoking_status = st.selectbox('Smoking status', ('Never smoker', 'Ever smoker')) #Smoking history

age = st.slider('Age', 18, 90, 60)  # Age: slider from 18 to 100
log_age = np.log(age) # Apply log transformation to age

neutrophil = st.text_input("Neutrophil count (x1000/mm3)") #Neutrophil count
lymphocyte = st.text_input("Lymphocyte count (x1000/mm3)") #Lymphocyte count
try:
    # Convert to float and calculate log_nlr
    neutrophil = float(neutrophil)
    lymphocyte = float(lymphocyte)
    log_nlr = np.log(neutrophil / lymphocyte)  # Logarithmic transformation
except ValueError:
    st.error("Please enter valid numeric values for neutrophil and lymphocyte counts.")
    log_nlr = None

line_of_treatment = st.selectbox('Line of Treatment', ('1st Line', '2nd Line or Higher'))  # Line of Treatment

# Process categorical inputs into numerical values (assuming 0/1 encoding)
gender_value = 1 if gender == 'Male' else 0
ecog_cat_value = 1 if ecog_cat == '2 to 4' else 0
mutation_status_value = 1 if mutation_status == 'Other EGFR mutations' else 0
number_mets_value = 1 if number_mets == '4 or more' else 0
bone_or_liver_met_value = 1 if bone_or_liver_met == 'Present' else 0    
brain_met_value = 1 if brain_met == 'Present' else 0  
comorbidity_value = 1 if comorbidity == 'Present' else 0 
smoking_status_value = 1 if smoking_status == 'Ever smoker' else 0 

# Create the input features array for the model
patient_features = [log_age, log_nlr, gender_value, ecog_cat_value, mutation_status_value, number_mets_value,
                    bone_or_liver_met_value, brain_met_value, comorbidity_value, smoking_status_value  ]

# Normalize features using the loaded scaler
scaled_features = scaler.transform([patient_features])

# Predict the recommended treatment when the user clicks the button
if st.button('Get Treatment Recommendation'):
    recommended_action = recommend_treatment(scaled_features.flatten())
    
    # Display the treatment recommendation
    if recommended_action == 0:
        st.success("Recommended Treatment: 1st line, 1st generation TKI")
    elif recommended_action == 1:
        st.success("Recommended Treatment: 1st line, 2nd or higher generation TKI")
    elif recommended_action == 2:
        st.success("Recommended Treatment: 2nd or later Line, 1st generation TKI")
    elif recommended_action == 3:
        st.success("Recommended Treatment: 2nd or later line, 2nd or higher generation TKI")
    else:
        st.error("No valid recommendation found.")
