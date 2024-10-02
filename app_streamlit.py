import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the model
model = joblib.load('house_model.joblib')

# Title of the app
st.title("California Housing Price Predictor")

# Sidebar for user input
st.sidebar.header("User Input")
median_income = st.sidebar.number_input("Median Income ($)", min_value=0.0, max_value=15.0, value=3.0, step=0.01)
house_age = st.sidebar.number_input("House Age (years)", min_value=1, max_value=52, value=20)
average_rooms = st.sidebar.number_input("Average Rooms", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
average_bedrooms = st.sidebar.number_input("Average Bedrooms", min_value=1.0, max_value=10.0, value=2.0, step=0.1)

# Create a DataFrame with the input data
input_data = pd.DataFrame({
    'MedInc': [median_income],
    'HouseAge': [house_age],
    'AveRooms': [average_rooms],
    'AveBedrooms': [average_bedrooms]
})

# Scale the input data
scaler = StandardScaler()
training_data = pd.DataFrame({
    'MedInc': np.random.rand(1000) * 15,    
    'HouseAge': np.random.rand(1000) * 52,
    'AveRooms': np.random.rand(1000) * 10,
    'AveBedrooms': np.random.rand(1000) * 10
})
scaler.fit(training_data)  
input_scaled = scaler.transform(input_data)

# Make prediction
predicted_price = model.predict(input_scaled)

# Display the predicted price
st.write(f"Predicted House Price: ${predicted_price[0]:.2f}")
