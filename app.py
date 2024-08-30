import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the trained model
with open('rf_rusm.pkl', 'rb') as file:
    rf_rusm = pickle.load(file)

# Title of the Streamlit app
st.title("Census Income Prediction App")

# Categorical columns
cat_col = ['workclass', 'marital.status', 'occupation', 'relationship', 'race', 'sex', 'native.country']

# User input for features
age = st.number_input('Age', min_value=17, max_value=90, value=75)
workclass = st.selectbox('Workclass', ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'])
education_num = st.number_input('Education Number', min_value=1, max_value=16, value=16)
marital_status = st.selectbox('Marital Status', ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'])
occupation = st.selectbox('Occupation', ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'])
relationship = st.selectbox('Relationship', ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'])
race = st.selectbox('Race', ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'])
sex = st.selectbox('Sex', ['Male', 'Female'])
capital_gain = st.number_input('Capital Gain', min_value=0, max_value=100000, value=120)
capital_loss = st.number_input('Capital Loss', min_value=0, max_value=4500, value=95)
hours_per_week = st.number_input('Hours per Week', min_value=0, max_value=100, value=94)
native_country = st.selectbox('Native Country', ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands'])

# Create a DataFrame from user input
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass],
    'education.num': [education_num],
    'marital.status': [marital_status],
    'occupation': [occupation],
    'relationship': [relationship],
    'race': [race],
    'sex': [sex],
    'capital.gain': [capital_gain],
    'capital.loss': [capital_loss],
    'hours.per.week': [hours_per_week],
    'native.country': [native_country]
})

# Feature preprocessing (Label Encoding)
def preprocess_input(df):
    for col in cat_col:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
    return df

# Apply preprocessing to the input data
input_data = preprocess_input(input_data)

# Make prediction
if st.button('Predict'):
    prediction = rf_rusm.predict(input_data)
    st.write(f'The predicted income class is: {"<=50K" if prediction == 0 else ">50K"}')

