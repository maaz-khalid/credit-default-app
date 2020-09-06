import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier


st.set_option('deprecation.showfileUploaderEncoding', False)
st.write("""
# **CREDIT DEFAULT PREDICTION**
_Makes predictions on which customers are likely to **default** in the following month & whether the account should be considered for **credit counseling** or not._
""")
st.write('---')

st.sidebar.header('User Input Features')

# Collects user input features into dataframe
def user_input_features():
    education = st.sidebar.selectbox('EDUCATION',('School Graduate','University','High School','Others'))
    marriage = st.sidebar.selectbox('MARRIAGE',('Married','Single','Others'))
    age = st.sidebar.slider('AGE', 16,100,25)
    limit = st.sidebar.slider('BALANCE LIMIT', 10000,1000000,100000)
    pay_1 = st.sidebar.slider('PREVIOUS MONTH PAYMENT', -2,8,0)
    bill_1 = st.sidebar.slider('BILL AMOUNT 1', -200000.00,1000000.00,0.00)
    bill_2 = st.sidebar.slider('BILL AMOUNT 2', -200000.00,1000000.00,0.00)
    bill_3 = st.sidebar.slider('BILL AMOUNT 3', -200000.00,1000000.00,0.00)
    bill_4 = st.sidebar.slider('BILL AMOUNT 4', -200000.00,1000000.00,0.00)
    bill_5 = st.sidebar.slider('BILL AMOUNT 5', -200000.00,1000000.00,0.00)
    bill_6 = st.sidebar.slider('BILL AMOUNT 6', -200000.00,1000000.00,0.00)
    pay_amt_1 = st.sidebar.slider('PAY AMOUNT 1', 0.00,1000000.00,10000.00)
    pay_amt_2 = st.sidebar.slider('PAY AMOUNT 2', 0.00,1000000.00,10000.00)
    pay_amt_3 = st.sidebar.slider('PAY AMOUNT 3', 0.00,1000000.00,10000.00)
    pay_amt_4 = st.sidebar.slider('PAY AMOUNT 4', 0.00,1000000.00,10000.00)
    pay_amt_5 = st.sidebar.slider('PAY AMOUNT 5', 0.00,1000000.00,10000.00)
    pay_amt_6 = st.sidebar.slider('PAY AMOUNT 6', 0.00,1000000.00,10000.00)
    
    if education =="School Graduate":
        educ = 1
    elif education == "University":
        educ =2
    elif education == "High School":
        educ = 3
    else:
        educ = 4
    if marriage == "Married":
        marry =1
    elif marriage == "Single":
        marry =2
    else:
        marry = 3
    data = {'LIMIT_BAL': limit,
            'EDUCATION': educ,
            'MARRIAGE': marry,
            'AGE': age,
            'PAY_1': pay_1,
            'BILL_AMT1': bill_1,
            'BILL_AMT2': bill_2,
            'BILL_AMT3': bill_3,
            'BILL_AMT4': bill_4,
            'BILL_AMT5': bill_5,
            'BILL_AMT6': bill_6,
            'PAY_AMT1': pay_amt_1,
            'PAY_AMT2': pay_amt_2,
            'PAY_AMT3': pay_amt_3,
            'PAY_AMT4': pay_amt_4,
            'PAY_AMT5': pay_amt_5,
            'PAY_AMT6': pay_amt_6
            }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
credit = pd.read_csv('cleaned_appdata.csv')
credit_use = credit.drop(columns=['DEFAULT'])
df = pd.concat([input_df,credit_use],axis=0)

df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')
st.write(df)
st.write('---')
# Reads in saved classification model
load_rf = joblib.load('credit.pkl')

# Apply model to make predictions
prediction = load_rf.predict(df)
prediction_proba = load_rf.predict_proba(df)


st.subheader('Prediction')
status = np.array(['Not Default','Default'])
st.write(status[prediction])
st.write('---')
st.subheader('Prediction Probability')
st.write(prediction_proba)
st.write('---')
if all(prediction_proba[0]>=0.25):
    if all(prediction==0):
        st.subheader('_THE ACCOUNT SHOULD BE CONSIDERED FOR **CREDIT COUNSELING**_')
