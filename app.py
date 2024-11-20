import streamlit as st
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

st.title('Thyroid Disease Prediction')

# Load the model
@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgb_model.pkl')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load and preprocess data
@st.cache_data
def load_and_preprocess_data():
    try:
        # Load dataset
        df = pd.read_csv('hypothyroid.csv')
        
        # Remove rows with "?" values
        df = df[(df != "?").all(axis=1)]
        
        # Map target variable
        df['binaryClass'] = df['binaryClass'].map({'P': 1, 'N': 0})
        
        # Drop the target variable to get features
        X = df.drop('binaryClass', axis=1)
        
        # Encode categorical columns
        for col in X.select_dtypes(include='object').columns:
            X[col] = LabelEncoder().fit_transform(X[col])
        
        # Convert to DataFrame after encoding
        X = pd.DataFrame(X)
        
        # Impute missing values
        imputer = SimpleImputer(strategy='most_frequent')
        X = imputer.fit_transform(X)
        
        # Standardize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        return X
    except Exception as e:
        st.error(f"Error preprocessing data: {e}")
        return None

def main():
    # Load model
    model = load_model()
    
    if model is None:
        st.error("Could not load the model. Please check if the model file exists.")
        return
    
    # Load and preprocess data
    X = load_and_preprocess_data()
    
    if X is None:
        st.error("Could not load or preprocess the data. Please check if the dataset file exists.")
        return
    
    # Add a button to make predictions
    if st.button('Make Predictions'):
        # Make predictions
        predictions = model.predict(X[:10])  # Predict first 10 rows
        
        # Convert predictions to class labels
        predicted_classes = ['Positive (P)' if p == 1 else 'Negative (N)' for p in predictions]
        
        # Display results
        st.subheader('Predictions for first 10 samples:')
        for i, pred in enumerate(predicted_classes):
            st.write(f"Sample {i+1}: {pred}")

if __name__ == '__main__':
    main()
