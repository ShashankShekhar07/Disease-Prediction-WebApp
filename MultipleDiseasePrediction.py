import numpy as np
import pickle
import streamlit as st


diabetes_model = pickle.load(open('C:/Users/shash/Desktop/Data Science/savedmodels/diabetes_model.sav', 'rb'))
heart_disease_model = pickle.load(open('C:/Users/shash/Desktop/Data Science/savedmodels/heart_disease_model.sav', 'rb'))
parkinson_model = pickle.load(open('C:/Users/shash/Desktop/Data Science/savedmodels/parkinsons_model.sav', 'rb'))
breastcancer_model = pickle.load(open('C:/Users/shash/Desktop/Data Science/savedmodels/BreastCancerPrediction.sav', 'rb'))

disease_selection = st.selectbox("Select Disease Prediction", ["Diabetes", "Heart Disease", "Parkinson's Disease","Breast Cancer"])

def predict_diabetes(input_data):
    """Predicts diabetes based on user input."""
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = diabetes_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'


def predict_heart_disease(input_data):
    """Predicts heart disease based on user input."""
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = heart_disease_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return 'The Person does not have a Heart Disease'
    else:
        return 'The Person has Heart Disease'


def predict_parkinsons(input_data):
    """Predicts Parkinson's disease based on user input."""
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = parkinson_model.predict(input_data_reshaped)
    print(prediction)

    if prediction[0] == 0:
        return "The Person does not have Parkinson's Disease"
    else:
        return "The Person has Parkinson's"

def predict_breastcancer(input_data):
    """Predicts diabetes based on user input."""
    input_data_as_numpy_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = breastcancer_model.predict(input_data_reshaped)

    if prediction == 4:
        return 'The person has a high risk of Breast Cancer'
    else:
        return 'The person has a low risk of Breast Cancer'

if disease_selection == "Diabetes":
    st.title('Diabetes Prediction Web App')

    pregnancies = st.text_input("Number of Pregnancies")
    glucose = st.text_input("Glucose Level")
    blood_pressure = st.text_input("Blood Pressure value")
    skin_thickness = st.text_input("Skin Thickness value")
    insulin = st.text_input("Insulin Level")
    bmi = st.text_input("BMI level")
    diabetes_pedigree_function = st.text_input("Diabetes Pedigree Function value")
    age = st.text_input("Age of the Person")

    diagnosis = ''

    if st.button('Diabetes Test Result'):
        try:
            user_input = [float(x) for x in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
            diagnosis = predict_diabetes(user_input)
        except ValueError:
            diagnosis = "Invalid input. Please enter only numbers."
        st.success(diagnosis)

elif disease_selection == 'Heart Disease':
    st.title('Heart Disease Prediction Web App')

    Age = st.text_input("Age")
    sex = st.text_input("sex")
    ChestPaintypes = st.text_input("Chest Pain types")
    RestingBloodPressure = st.text_input("Resting Blood Pressure")
    SerumCholestoral = st.text_input("Serum Cholestoral in mg/dl")
    FastingBloodSugar = st.text_input("Fasting Blood Sugar > 120 mg/dl")
    RestingElectrocardiographic = st.text_input("Resting Electrocardiographic results")
    MaximumHeartRate = st.text_input("Maximum Heart Rate achieved")
    Angina = st.text_input("Exercise Induced Angina")
    STdepression= st.text_input("ST depression induced by exercise")
    Slope = st.text_input("Slope of the peak exercise ST segment")
    Majorvessels = st.text_input("Major vessels colored by flourosopy")
    thal = st.text_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    diagnosis = ''

    if st.button('Heart Disease Test Result'):
        try:
            user_input = [float(x) for x in [Age, sex, ChestPaintypes, RestingBloodPressure, SerumCholestoral, FastingBloodSugar, RestingElectrocardiographic, MaximumHeartRate,Angina,STdepression,Slope,Majorvessels,thal]]
            diagnosis = predict_heart_disease(user_input)
        except ValueError:
            diagnosis = "Invalid input. Please enter only numbers."
        st.success(diagnosis)

elif disease_selection == "Parkinson's Disease":
    st.title("Parkinson's Disease Web App")
    
    fo = st.text_input('MDVP:Fo(Hz)')
    fhi = st.text_input('MDVP:Fhi(Hz)')
    flo = st.text_input('MDVP:Flo(Hz)')
    Jitter_percent = st.text_input('MDVP:Jitter(%)')
    Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
    RAP = st.text_input('MDVP:RAP')
    PPQ = st.text_input('MDVP:PPQ')
    DDP = st.text_input('Jitter:DDP')
    Shimmer = st.text_input('MDVP:Shimmer')
    Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
    APQ3 = st.text_input('Shimmer:APQ3')
    APQ5 = st.text_input('Shimmer:APQ5')
    APQ = st.text_input('MDVP:APQ')
    DDA = st.text_input('Shimmer:DDA')
    NHR = st.text_input('NHR')
    HNR = st.text_input('HNR')
    RPDE = st.text_input('RPDE')
    DFA = st.text_input('DFA')
    spread1 = st.text_input('spread1')
    spread2 = st.text_input('spread2')
    D2 = st.text_input('D2')
    PPE = st.text_input('PPE')


    diagnosis = ''

    if st.button("Parkinson's Disease Test Result"):
        try:
            user_input = [float(x) for x in [fo, fhi, flo, Jitter_percent, Jitter_Abs,
                      RAP, PPQ, DDP,Shimmer, Shimmer_dB, APQ3, APQ5,
                      APQ, DDA, NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE]]
            diagnosis = predict_parkinsons(user_input)
        except ValueError:
            diagnosis = "Invalid input. Please enter only numbers."
        st.success(diagnosis)    

elif disease_selection == 'Breast Cancer':
    st.title('Breast Cancer Prediction Web App')

    
    clump_thickness = st.text_input("clump_thickness")
    uniform_cell_size = st.text_input("uniform_cell_size")
    uniform_cell_shape = st.text_input("uniform_cell_shape")
    marginal_adhesion = st.text_input("marginal_adhesion")
    single_epithelial_size = st.text_input("single_epithelial_size")
    bare_nuclei = st.text_input("bare_nuclei")
    bland_chromatin = st.text_input("bland_chromatin")
    normal_nucleoli = st.text_input("normal_nucleoli")
    mitoses = st.text_input("mitoses")

    diagnosis = ''

    if st.button('Breast Cancer Test Result'):
        try:
            # Convert inputs to floats
            user_input = [float(x) for x in [clump_thickness, uniform_cell_size, uniform_cell_shape, marginal_adhesion, single_epithelial_size, bare_nuclei, bland_chromatin, normal_nucleoli,mitoses]]
            diagnosis = predict_breastcancer(user_input)
        except ValueError:
            diagnosis = "Invalid input. Please enter only numbers."
        st.success(diagnosis)

else:
    st.title("Please Enter Valid selector")


