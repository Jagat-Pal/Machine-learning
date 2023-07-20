import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import math

# Load the dataset
df = pd.read_csv("placement.csv")

# Set up the sidebar
st.sidebar.title("Student CGPA to Package Prediction")
cgpa_input = st.sidebar.number_input("Enter CGPA:", min_value=0.0, max_value=10.0, step=0.01)

# Find independent and dependent feature
x = df.iloc[:, 0].values.reshape(-1, 1)  # independent feature (CGPA)
y = df.iloc[:, 1].values  # dependent feature (Package)

# Split the dataset into train and test sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

# Create and train the Linear Regression model
regression = LinearRegression()
regression.fit(x_train, y_train)

# Predict the package based on user input CGPA
predicted_package = regression.predict([[cgpa_input]])

# Display the predicted package
st.title("Student Package Prediction")
st.write(f"Predicted Package (LPA): {predicted_package[0]:.2f} LPA")
