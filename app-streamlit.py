import numpy as np
import pickle
import streamlit as st

model = pickle.load(open('model.pkl', 'rb'))

def welcome():
    return "Welcome All"

def predict(X):
    final_features = [np.array(X)] # storing input values as numpy array
    prediction = model.predict(final_features) # predicting
    return prediction[0]

def main():
    st.title("Crop Recommendation")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Crop Recommender ML App </h2>
    </div>
    """


    st.markdown(html_temp,unsafe_allow_html=True)
    nitrogen = st.number_input("Nitrogen(ratio in soil)", min_value=0.0, step=0.01,max_value=100.0,format="%.2f")
    phosphorous = st.number_input("Phosphorous(ratio in soil)", min_value=0.0,  step=0.01,max_value=100.0, format="%.2f")
    potassium = st.number_input("Potassium(ratio in soil)", min_value=0.0,  step=0.01,max_value=100.0, format="%.2f")
    temperature = st.number_input("Temperature(°C)", min_value=-50.0, max_value=60.00,value=0.0,step=0.01,format="%.2f")
    relative_humidity = st.number_input("Relative Humidity(in %)", min_value=0.0, max_value=100.00, step=0.01,format="%.2f")
    ph = st.number_input("PH Value", min_value=0.0, max_value=14.00,  step=0.01,format="%.2f")
    rainfall = st.number_input("Rainfall(in mm)", min_value=0.0, step=0.01,format="%.2f")
    result=""
    r = [nitrogen,phosphorous,potassium,temperature,relative_humidity,ph,rainfall]
    X =  [float(x) for x in r]
    if st.button("Predict"):
        result=predict(X)  
    st.success('Best Crop to grow is {}'.format(result))
    if st.button("About"):
        st.text("Build by Naimish Sharma")
        st.text("Email: naimish.s2017@gmail.com")



if __name__ == "__main__": # runs complete flask
    main()
    
    