# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:58:27 2025

@author:Muskan Gupta
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 17:02:27 2025

@author: Muskan Gupta
"""
import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('C:/Users/Sarvesh/OneDrive/Desktop/Diabetes Prediction/trained_model.sav','rb'))

def diabetes_prediction(input_data):

# changing the input_data to numpy array

    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

#standardize the input data
#std_data = scaler.transform(input_data_reshaped)
#print(std_data)


    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return("The person is not Diabetic")
    else:
        return("The person is Diabetic")
    
def main():
    
    #giving a title
    st.title('Diabetes Prediction Web App')
    
    # getting the input data from user
    
    pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunctions = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    # code for Prediction 
    diagnosis = ''
    
    # creating a button for prediction
    
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunctions,Age])
        
    st.success(diagnosis)
    
    
    
if __name__ == '__main__':
    main()  