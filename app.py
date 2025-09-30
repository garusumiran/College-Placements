import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import joblib # Import joblib to potentially save/load the scaler and kmeans models

# Define the path to the saved model
MODEL_PATH = 'best_deep_learning_model.h5'

# Define numerical features for scaling and KMeans
NUMERICAL_FEATURES = ['IQ', 'Prev_Sem_Result', 'CGPA', 'Academic_Performance', 'Extra_Curricular_Score', 'Communication_Skills', 'Projects_Completed']

# Define categorical feature (for now, only 'Internship_Experience' is treated as such)
CATEGORICAL_FEATURES = ['Internship_Experience']


@st.cache_resource
def load_model(model_path):
    """Loads the trained deep learning model."""
    try:
        model = tf.keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_resource
def load_and_fit_preprocessing_models(data):
    """Loads or fits the StandardScaler and KMeans models."""
    try:
        # Fit StandardScaler on the numerical features of the provided data
        scaler = StandardScaler()
        scaler.fit(data[NUMERICAL_FEATURES])

        # Fit KMeans on the data (excluding the target if present) to get cluster labels
        # Assuming 'Placement' is the target and 'College_ID' was dropped
        features_for_kmeans = data.drop(columns=['Placement'] if 'Placement' in data.columns else [], errors='ignore')
        features_for_kmeans = features_for_kmeans.select_dtypes(include=np.number) # Use only numerical features for KMeans
        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        kmeans.fit(features_for_kmeans)

        return scaler, kmeans
    except Exception as e:
        st.error(f"Error fitting preprocessing models: {e}")
        return None, None


def preprocess_input(input_data, scaler, kmeans):
    """Preprocesses the raw user input."""
    try:
        # Convert input to DataFrame if it's a dictionary
        if isinstance(input_data, dict):
            input_df = pd.DataFrame([input_data])
        elif isinstance(input_data, pd.DataFrame):
            input_df = input_data
        else:
            raise ValueError("Input data must be a dictionary or pandas DataFrame")

        # Ensure columns are in the correct order and format as during training
        # Recreate the structure of the data used for scaling and KMeans fitting
        # We need the original df to fit the scaler and kmeans correctly
        # For the app, we will refit them on a representative dataset or the full dataset if available.
        # For this example, we will assume we refit on the data available in the notebook state (df before splitting).

        # Scale numerical features
        input_numerical = input_df[NUMERICAL_FEATURES]
        input_numerical_scaled = scaler.transform(input_numerical)

        # Get categorical feature (assuming it's already encoded as 0 or 1)
        input_categorical = input_df[CATEGORICAL_FEATURES].values

        # Predict cluster label
        # Need to prepare data for KMeans prediction - use only the features KMeans was fitted on
        input_for_kmeans = input_df.select_dtypes(include=np.number) # Use only numerical features for KMeans prediction
        cluster_label = kmeans.predict(input_for_kmeans).reshape(-1, 1)

        # Concatenate scaled numerical, categorical, and cluster label features
        processed_input = np.hstack((input_numerical_scaled, input_categorical, cluster_label))

        return processed_input
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None


# Load the model
model = load_model(MODEL_PATH)

# Load and fit the scaler and kmeans models (using the full original dataframe for fitting)
# Note: In a real deployment, you would save and load these fitted objects.
# For this exercise, we're refitting them on the 'df' available in the session state.
if 'df' in locals():
    scaler, kmeans = load_and_fit_preprocessing_models(df)
else:
    st.error("Original dataframe 'df' not found. Cannot fit scaler and KMeans.")
    scaler, kmeans = None, None


st.title("College Placement Prediction")
st.write("Enter the student's details to predict placement.")

# Create input widgets
iq = st.number_input("IQ", min_value=40, max_value=160, value=100)
prev_sem_result = st.number_input("Previous Semester Result (out of 10)", min_value=0.0, max_value=10.0, value=7.5, format="%.2f")
cgpa = st.number_input("CGPA (out of 10)", min_value=0.0, max_value=10.0, value=7.5, format="%.2f")
academic_performance = st.slider("Academic Performance (1-10)", min_value=1, max_value=10, value=6)
internship_experience = st.selectbox("Internship Experience", options=["No", "Yes"], index=0)
extra_curricular_score = st.slider("Extra Curricular Score (0-10)", min_value=0, max_value=10, value=5)
communication_skills = st.slider("Communication Skills (1-10)", min_value=1, max_value=10, value=6)
projects_completed = st.slider("Projects Completed (0-5)", min_value=0, max_value=5, value=3)

# Map categorical input to numerical
internship_experience_encoded = 1 if internship_experience == "Yes" else 0

# Create a button to predict
if st.button("Predict Placement"):
    if model is not None and scaler is not None and kmeans is not None:
        # Collect input into a dictionary or DataFrame
        input_data = {
            'IQ': iq,
            'Prev_Sem_Result': prev_sem_result,
            'CGPA': cgpa,
            'Academic_Performance': academic_performance,
            'Internship_Experience': internship_experience_encoded,
            'Extra_Curricular_Score': extra_curricular_score,
            'Communication_Skills': communication_skills,
            'Projects_Completed': projects_completed
        }

        # Preprocess the input
        processed_input = preprocess_input(input_data, scaler, kmeans)

        if processed_input is not None:
            # Make prediction
            prediction_prob = model.predict(processed_input)
            prediction = (prediction_prob > 0.5).astype("int32")[0][0]

            # Display result
            st.header("Prediction Result:")
            if prediction == 1:
                st.success("Predicted Placement: Yes")
            else:
                st.error("Predicted Placement: No")
            st.write(f"Prediction Probability (Placement=Yes): {prediction_prob[0][0]:.4f}")
    else:
        st.warning("Model or preprocessing components not loaded correctly.")
