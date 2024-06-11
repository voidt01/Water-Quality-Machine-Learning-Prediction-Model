import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
scaler = joblib.load('scaler.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
knn_model = joblib.load('models/knn_model.pkl')
lr_model = joblib.load('models/lr_model.pkl')
svc_model = joblib.load('models/svc_model.pkl')
dt_model = joblib.load('models/dt_model.pkl')
rf_model = joblib.load('models/rf_model.pkl')
voting_model = joblib.load('models/voting_clf_model.pkl')

def main():
    st.title('Water Quality Classification App')
    
    # create a number_input for these column ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity'
    
    # Create ph input with 0 as the minimum and 14 as the max value
    model_selection = st.selectbox('Select Model', ['XGBoost', 'KNN', 'Logistic Regression', 'SVC','Decision Tree', 'Random Forest', 'Voting Classifier'])

    if model_selection == 'XGBoost':
        model = xgb_model
    elif model_selection == 'KNN':
        model = knn_model
    elif model_selection == 'Logistic Regression':
        model = lr_model
    elif model_selection == 'SVC':
        model = svc_model
    elif model_selection == 'Decision Tree':
        model = dt_model
    elif model_selection == 'Random Forest':
        model = rf_model
    elif model_selection == 'Voting Classifier':
        model = voting_model

    ph = st.number_input('pH (0-14): ', min_value=0.0, max_value=14.0)
    Hardness = st.number_input('Hardness (mg/L): ', min_value=0.0)
    Solids = st.number_input('Solids (ppm): ', min_value=0.0)
    Chloramines = st.number_input('Chloramines (ppm): ', min_value=0.0)
    Sulfate = st.number_input('Sulfate (mg/L): ', min_value=0.0)
    Conductivity = st.number_input('Conductivity (μS/cm): ', min_value=0.0)
    Organic_carbon = st.number_input('Organic_carbon (ppm): ', min_value=0.0)
    Trihalomethanes = st.number_input('Trihalomethanes (μg/L): ', min_value=0.0)
    Turbidity = st.number_input('Turbidity (NTU): ', min_value=0.0)
    
    if st.button('Check Water Quality'):
        # make prediction
        
        input = np.array([[ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity]])
        
        scaled_input = scaler.transform(input)
        
        prediction = model.predict(scaled_input)
        
        if prediction == 1:
            st.success('The water is safe to drink')
        elif prediction == 0:
            st.warning('The water is not safe to drink')

if __name__ == '__main__':
    main()
