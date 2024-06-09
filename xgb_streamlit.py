import streamlit as st
import joblib
import numpy as np

# Load the machine learning model
model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

def main():
    st.title('Water Quality Classification App')
    
    # create a number_input for these column ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 'Conductivity','Organic_carbon', 'Trihalomethanes', 'Turbidity'
    
    # Create ph input with 0 as the minimum and 14 as the max value
    ph = st.number_input('pH (0-14): ', min_value=0.0, max_value=14.0)
    Hardness = st.number_input('Hardness: ')
    Solids = st.number_input('Solids: ')
    Chloramines = st.number_input('Chloramines: ')
    Sulfate = st.number_input('Sulfate: ')
    Conductivity = st.number_input('Conductivity: ')
    Organic_carbon = st.number_input('Organic_carbon: ')
    Trihalomethanes = st.number_input('Trihalomethanes: ')
    Turbidity = st.number_input('Turbidity: ')
    
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
