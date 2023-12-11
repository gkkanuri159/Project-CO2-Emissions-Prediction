#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 17:36:50 2023

@author: gitakanuri
"""

# app.py

import streamlit as st
import joblib
import pandas as pd

# Load the pre-trained model
model = joblib.load('co2_emit_pred_model.pkl')

# Function to preprocess user input
def preprocess_input(input_df):
    input_df['cylinders'] = input_df['cylinders'].astype('category')
    return input_df

# Function to predict CO2 emissions
def predict_co2_emissions(features):
    input_data = pd.DataFrame(features, index=[0])
    preprocessed_input = preprocess_input(input_data)
    prediction = model.predict(preprocessed_input)[0]
    return prediction

# Loading the dataset
df = pd.read_csv('co2_emissions.csv', sep=';')

# Streamlit UI
st.markdown(
    """
    <div style="font-family: Times New Roman; font-size:50px;">
        CO2 Emissions Prediction
    </div>
    """, 
    unsafe_allow_html=True
)
# User input form
st.sidebar.header('User Input')
make = st.sidebar.selectbox('Make', df['make'].unique())
model_name = st.sidebar.selectbox('Model', df['model'].unique())
vehicle_class = st.sidebar.selectbox('Vehicle Class', df['vehicle_class'].unique())
engine_size = st.sidebar.slider('Engine Size', min_value=float(df['engine_size'].min()), max_value=float(df['engine_size'].max()), step=0.1)
cylinders = st.sidebar.slider('Cylinders', min_value=float(df['cylinders'].min()), max_value=float(df['cylinders'].max()), step=1.0)
transmission = st.sidebar.selectbox('Transmission', df['transmission'].unique())
fuel_type = st.sidebar.selectbox('Fuel Type', df['fuel_type'].unique())
fuel_consumption_city = st.sidebar.slider('Fuel Consumption City (l/100km)', min_value=float(df['fuel_consumption_city'].min()), max_value=float(df['fuel_consumption_city'].max()), step=0.1)
fuel_consumption_hwy = st.sidebar.slider('Fuel Consumption Hwy (l/100km)', min_value=float(df['fuel_consumption_hwy'].min()), max_value=float(df['fuel_consumption_hwy'].max()), step=0.1)
fuel_consumption_comb_l100km = st.sidebar.slider('Fuel Consumption Comb (l/100km)', min_value=float(df['fuel_consumption_comb(l/100km)'].min()), max_value=float(df['fuel_consumption_comb(l/100km)'].max()), step=0.1)
fuel_consumption_comb_mpg = st.sidebar.slider('Fuel Consumption Comb (mpg)', min_value=float(df['fuel_consumption_comb(mpg)'].min()), max_value=float(df['fuel_consumption_comb(mpg)'].max()), step=1.0)

# User input features
user_input = {
    'make': make,
    'model': model_name,
    'vehicle_class': vehicle_class,
    'engine_size': engine_size,
    'cylinders': cylinders,
    'transmission': transmission,
    'fuel_type': fuel_type,
    'fuel_consumption_city': fuel_consumption_city,
    'fuel_consumption_hwy': fuel_consumption_hwy,
    'fuel_consumption_comb(l/100km)': fuel_consumption_comb_l100km,
    'fuel_consumption_comb(mpg)': fuel_consumption_comb_mpg
}

# Prediction
if st.sidebar.button('Predict CO2 Emissions'):
    prediction = predict_co2_emissions(user_input)
    st.success(f'Predicted CO2 Emissions is: {prediction:.2f} grams/KM')

