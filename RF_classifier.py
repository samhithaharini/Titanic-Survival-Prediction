import streamlit as st
import pandas as pd
import joblib
import numpy as np


try:
    model = joblib.load('random_forest_model.pkl')
    le_feature = joblib.load('label_encoder_feature.pkl')
    
    
    FEATURE_COLUMNS = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    
    st.sidebar.success("Model and Encoder loaded successfully!")
except FileNotFoundError:
    st.error("Error: Could not find model files. Please run the Joblib code first.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred during loading: {e}")
    st.stop()


st.title("Titanic Survival Prediction")

with st.form("prediction_form"):
    st.header("Passenger Information")

   
    pclass = st.selectbox(
        "Passenger Class (Pclass)",
        options=[1, 2, 3],
        index=2,
        help="1 = 1st (Upper), 2 = 2nd (Middle), 3 = 3rd (Lower)"
    )

   
    sex = st.selectbox(
        "Sex",
        options=['male', 'female'],
        index=0 
    )

  
    age = st.slider(
        "Age",
        min_value=0.42, 
        max_value=80.0, 
        value=28.0, 
        step=1.0
    )

   
    sibsp = st.number_input(
        "Number of Siblings/Spouses Aboard (SibSp)",
        min_value=0, 
        max_value=8, 
        value=0, 
        step=1
    )

   
    parch = st.number_input(
        "Number of Parents/Children Aboard (Parch)",
        min_value=0, 
        max_value=6, 
        value=0, 
        step=1
    )

   
    fare = st.number_input(
        "Fare (Ticket Price)",
        min_value=0.0, 
        value=15.0, 
        step=0.01
    )

    submitted = st.form_submit_button("Predict Survival")


if submitted:
    try:
        
        feature_encoded = le_feature.transform([sex])[0]
        
        
        input_data = pd.DataFrame(
            [[pclass, feature_encoded, age, sibsp, parch, fare]],
            columns=FEATURE_COLUMNS
        )

        
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        st.subheader("Prediction Result")
        
        if prediction[0] == 1:
            st.success(f"**Prediction: SURVIVED!** ")
            survival_status = "Survived"
        else:
            st.error(f"**Prediction: DID NOT SURVIVE!** ")
            survival_status = "Did Not Survive"
            
        st.write(f"The model predicts the passenger would **{survival_status}**.")
        
        
        st.info(f"""
            **Confidence Scores:**
            * Did Not Survive (0): {prediction_proba[0][0]*100:.2f}%
            * Survived (1): {prediction_proba[0][1]*100:.2f}%
        """)

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")