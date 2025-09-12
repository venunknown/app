import joblib
import streamlit as st
import pandas as pd
import numpy as np

model=joblib.load("model.joblib")
df=pd.read_csv('dataset1.csv')
x=df.drop("diseases",axis=1)
all_symptoms={col:0 for col in x.columns}
list_symptoms=[col for col in x.columns]
st.markdown('''
<style>
            .stApp{
            background-color:rgb(133,133,133)
            } </style>
            ''',
            unsafe_allow_html=True
            )

st.title("AI Telemedicine Project")
symptoms = st.multiselect("**Please enter your symptoms below:**", list_symptoms)

if st.button("Submit"):
    for symptom in symptoms:
        all_symptoms[symptom]=1

    st.write("Your predicted disease is:")
    
    st.write(model.predict(pd.DataFrame([all_symptoms]))[0])