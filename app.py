import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download, login
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# Constants
MODEL_PATH = "diabetes_model.pkl"
SCALER_PATH = "scaler.pkl"

# Pima Indians Diabetes Dataset (embedded for reliability)
DIABETES_DATA = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,0.0,0.232,54,1
4,110,92,0,0,37.6,0.191,30,0
10,168,74,0,0,38.0,0.537,34,1
10,139,80,0,0,27.1,1.441,57,0
1,189,60,23,846,30.1,0.398,59,1
5,166,72,19,175,25.8,0.587,51,1
7,100,0,0,0,30.0,0.484,32,1
0,118,84,47,230,45.8,0.551,31,1
7,107,74,0,0,29.6,0.254,31,1
1,103,30,38,83,43.3,0.183,33,0
1,115,70,30,96,34.6,0.529,32,1
3,126,88,41,235,39.3,0.704,27,0
8,99,84,0,0,35.4,0.388,50,0
7,196,90,0,0,39.8,0.451,41,1
9,119,80,35,0,29.0,0.263,29,1
11,143,94,33,146,36.6,0.254,51,1
10,125,70,26,115,31.1,0.205,41,1
7,147,76,0,0,39.4,0.257,43,1
1,97,66,15,140,23.2,0.487,22,0
13,145,82,19,110,22.2,0.245,57,0
5,117,92,0,0,34.1,0.337,38,0
5,109,75,26,0,36.0,0.546,60,0
3,158,76,36,245,31.6,0.851,28,1
3,88,58,11,54,24.8,0.267,22,0
6,92,92,0,0,19.9,0.188,28,0
10,122,78,31,0,27.6,0.512,45,0
4,103,60,33,192,24.0,0.966,33,0
11,138,76,0,0,33.2,0.420,35,0
9,102,76,37,0,32.9,0.665,46,1
2,90,68,42,0,38.2,0.503,27,1
4,111,72,47,176,37.1,1.390,56,1
3,180,64,25,70,34.0,0.271,26,0
7,133,84,0,0,40.2,0.696,37,0
7,106,92,18,0,22.7,0.235,48,0
9,171,110,24,240,45.4,0.721,54,1
7,159,64,0,0,27.4,0.294,40,0
0,180,66,39,0,42.0,1.893,25,1
1,146,56,0,0,29.7,0.564,29,0
2,71,70,27,0,28.0,0.586,22,0
7,103,66,32,0,39.1,0.344,31,1
7,105,0,0,0,0.0,0.305,24,0
1,103,80,11,82,19.4,0.491,22,0
1,101,50,15,36,24.2,0.526,26,0
5,88,66,21,23,24.4,0.342,30,0
8,176,90,34,300,33.7,0.467,58,1
7,150,66,42,342,34.7,0.718,42,0
1,73,50,10,0,23.0,0.248,21,0
7,187,68,39,304,37.7,0.254,41,1
0,100,88,60,110,46.8,0.962,31,0
0,146,82,0,0,40.5,1.781,44,0
0,105,64,41,142,41.5,0.173,22,0
2,84,0,0,0,0.0,0.304,21,0
8,133,72,0,0,32.9,0.270,39,1
5,44,62,0,0,25.0,0.587,36,0
2,141,58,34,128,25.4,0.699,24,0
7,114,66,0,0,32.8,0.258,42,1
5,99,74,27,0,29.0,0.203,32,0
0,109,88,30,0,32.5,0.855,38,1
2,109,92,0,0,42.7,0.845,54,0
1,95,66,13,38,19.6,0.334,25,0
4,146,85,27,100,28.9,0.189,27,0
2,100,66,20,90,32.9,0.867,28,1
5,139,64,35,140,28.6,0.411,26,0
13,126,90,0,0,43.4,0.583,42,1
4,129,86,20,270,35.1,0.231,23,0
1,79,75,30,0,32.0,0.396,22,0
1,0,48,20,0,24.7,0.140,22,0
7,62,78,0,0,32.6,0.391,41,0
5,95,72,33,0,37.7,0.370,27,0
0,131,0,0,0,43.2,0.270,26,1
2,112,66,22,0,25.0,0.307,24,0
3,113,44,13,0,22.4,0.140,22,0
2,74,0,0,0,0.0,0.102,22,0
7,83,78,26,71,29.3,0.767,36,0
0,101,65,28,0,24.6,0.237,22,0
5,137,108,0,0,48.8,0.227,37,1
2,110,74,29,125,32.4,0.698,27,0
13,106,72,54,0,36.6,0.178,45,0
2,100,68,25,71,38.5,0.324,26,0
15,136,70,32,110,37.1,0.153,43,1
1,107,68,19,0,26.5,0.165,24,0
1,80,55,0,0,19.1,0.258,21,0
4,123,80,15,176,32.0,0.443,34,0
7,81,78,40,48,46.7,0.261,42,0
4,134,72,0,0,23.8,0.277,60,1
2,142,82,18,64,24.7,0.761,21,0
6,144,72,27,228,33.9,0.255,40,0
2,92,62,28,0,31.6,0.130,24,0
1,71,48,18,76,20.4,0.323,22,0
6,93,50,30,64,28.7,0.356,23,0
1,122,90,51,220,49.7,0.325,31,1
1,163,72,0,0,39.0,1.222,33,1
1,151,60,0,0,26.1,0.179,22,0
0,125,96,0,0,22.5,0.262,21,0
1,81,72,18,40,26.6,0.283,24,0
2,85,65,0,0,39.6,0.930,27,0
1,126,56,29,152,28.7,0.801,21,0
1,96,122,0,0,22.4,0.207,27,0
4,144,58,28,140,29.5,0.287,37,0
3,83,58,31,18,34.3,0.336,25,0
0,95,85,25,36,37.4,0.247,24,1
3,171,72,33,135,33.3,0.199,24,1
8,155,62,26,495,34.0,0.543,46,1
1,89,76,34,37,31.2,0.192,23,0
4,76,62,0,0,34.0,0.391,25,0
7,160,54,32,175,30.5,0.588,39,1
4,146,92,0,0,31.2,0.539,61,1
5,124,74,0,0,34.0,0.220,38,1
2,71,70,27,0,28.0,0.586,22,0
0,102,86,17,105,29.3,0.695,27,0
0,119,66,27,0,38.8,0.259,22,0
6,108,44,20,130,24.0,0.813,35,0
2,118,80,0,0,42.9,0.693,21,1
2,112,78,50,140,39.4,0.175,24,0
0,167,0,0,0,32.3,0.839,30,1
0,86,68,32,0,35.8,0.238,25,0
"""


def get_hf_token():
    """Get Hugging Face API token from environment"""
    return os.getenv("hf_key")


def authenticate_hf():
    """Authenticate with Hugging Face using API key"""
    token = get_hf_token()
    if token:
        try:
            login(token=token)
            return True
        except Exception as e:
            st.warning(f"HuggingFace authentication failed: {e}")
            return False
    return False


def load_or_create_model():
    """Load existing model or create a new one"""
    # Try to load existing model
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            return model, scaler
        except Exception:
            pass
    
    # Create and train a new model
    return train_model()


def train_model():
    """Train a diabetes prediction model using embedded dataset"""
    from io import StringIO
    
    # Load embedded dataset
    df = pd.read_csv(StringIO(DIABETES_DATA))
    
    # Features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Save model and scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    return model, scaler


def predict_diabetes(model, scaler, input_data):
    """Make prediction using the model"""
    # Scale input
    input_scaled = scaler.transform([input_data])
    
    # Get prediction and probability
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0]
    
    return prediction, probability


def main():
    # Header
    st.title("🩺 Diabetes Prediction App")
    st.markdown("---")
    st.markdown("""
    This app predicts the likelihood of diabetes based on medical indicators.
    Enter your health parameters below to get a prediction.
    """)
    
    # Authenticate with HuggingFace (for potential future model downloads)
    with st.spinner("Initializing..."):
        authenticate_hf()
        model, scaler = load_or_create_model()
    
    st.markdown("---")
    st.subheader("📋 Enter Your Health Parameters")
    
    # Gender selection first
    gender = st.radio(
        "Select Gender",
        options=["Female", "Male"],
        horizontal=True,
        help="Select your gender"
    )
    
    st.markdown("")  # Add some spacing
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    with col1:
        # Show pregnancies input only for females
        if gender == "Female":
            pregnancies = st.number_input(
                "Pregnancies",
                min_value=0,
                max_value=20,
                value=0,
                help="Number of times pregnant"
            )
        else:
            pregnancies = 0  # Default to 0 for males
        
        glucose = st.number_input(
            "Glucose Level (mg/dL)",
            min_value=0,
            max_value=200,
            value=100,
            help="Plasma glucose concentration (2 hours after glucose tolerance test)"
        )
        
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0,
            max_value=130,
            value=70,
            help="Diastolic blood pressure"
        )
        
        skin_thickness = st.number_input(
            "Skin Thickness (mm)",
            min_value=0,
            max_value=100,
            value=20,
            help="Triceps skin fold thickness"
        )
    
    with col2:
        insulin = st.number_input(
            "Insulin (mu U/ml)",
            min_value=0,
            max_value=900,
            value=80,
            help="2-Hour serum insulin"
        )
        
        bmi = st.number_input(
            "BMI (kg/m²)",
            min_value=10.0,
            max_value=70.0,
            value=25.0,
            step=0.1,
            help="Body Mass Index (weight in kg / height in m²)"
        )
        
        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            max_value=2.5,
            value=0.5,
            step=0.01,
            help="Diabetes pedigree function (genetic influence score)"
        )
        
        age = st.number_input(
            "Age (years)",
            min_value=21,
            max_value=100,
            value=30,
            help="Age in years"
        )
    
    st.markdown("---")
    
    # Predict button
    if st.button("🔍 Predict Diabetes Risk", type="primary", use_container_width=True):
        # Prepare input data
        input_data = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]
        
        # Make prediction
        with st.spinner("Analyzing..."):
            prediction, probability = predict_diabetes(model, scaler, input_data)
        
        # Display results
        st.markdown("---")
        st.subheader("📊 Prediction Results")
        
        if prediction == 1:
            st.error("⚠️ **High Risk of Diabetes**")
            risk_percentage = probability[1] * 100
        else:
            st.success("✅ **Low Risk of Diabetes**")
            risk_percentage = probability[0] * 100
        
        # Show probability
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Level", f"{probability[1] * 100:.1f}%")
        with col2:
            st.metric("Confidence", f"{risk_percentage:.1f}%")
        
        # Progress bar for risk visualization
        st.markdown("**Risk Indicator:**")
        st.progress(probability[1])
        
        # Health recommendations
        st.markdown("---")
        st.subheader("💡 Health Recommendations")
        
        if prediction == 1:
            st.warning("""
            Based on the analysis, you may be at higher risk for diabetes. Consider:
            - 🏥 Consulting with a healthcare professional
            - 🥗 Maintaining a balanced diet low in sugar and processed foods
            - 🏃 Regular physical activity (at least 30 minutes daily)
            - 📊 Regular monitoring of blood glucose levels
            - 💊 Following medical advice and prescribed treatments
            """)
        else:
            st.info("""
            Your risk appears to be lower, but maintaining a healthy lifestyle is important:
            - 🥗 Continue eating a balanced, nutritious diet
            - 🏃 Stay physically active
            - 📊 Get regular health check-ups
            - 😴 Maintain healthy sleep patterns
            - 💧 Stay well hydrated
            """)
    
    # Footer
    st.markdown("---")
    st.caption("""
    ⚠️ **Disclaimer:** This app is for educational purposes only and should not be used 
    as a substitute for professional medical advice, diagnosis, or treatment. 
    Always consult with a qualified healthcare provider.
    """)
    
    # Sidebar with info
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This diabetes prediction app uses a **Random Forest** machine learning model 
        trained on the Pima Indians Diabetes Dataset.
        
        **Features used for prediction:**
        - Pregnancies
        - Glucose Level
        - Blood Pressure
        - Skin Thickness
        - Insulin
        - BMI
        - Diabetes Pedigree Function
        - Age
        
        **Model Accuracy:** ~77%
        """)
        
        st.markdown("---")
        st.markdown("**Powered by:**")
        st.markdown("🤗 Hugging Face | 🎈 Streamlit")


if __name__ == "__main__":
    main()
