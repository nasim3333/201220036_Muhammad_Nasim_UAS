import streamlit as st
import pandas as pd 
from PIL import Image 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
from sklearn.preprocessing import LabelEncoder

import seaborn as sns 
import pickle 

#import model 
xgb = pickle.load(open('XGB.pkl', 'rb'))


#load dataset
data = pd.read_csv('Stroke Dataset .csv')
#data = data.drop(data.columns[0],axis=1)

st.title('Aplikasi Stroke')

html_layout1 = """
<br>
<div style="background-color:red ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Diabetes Checkup</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['XGB','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah dataset PIMA Indian</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr = ProfileReport(data, explorative=True, config_file="")

    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#Handling missing value dengan mean data
import math
data['bmi'].fillna(math.floor(data['bmi'].mean()),inplace=True)
data=data.drop(['id'],axis=1)

le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['ever_married'] = le.fit_transform(data['ever_married'])
data['work_type'] = le.fit_transform(data['work_type'])
data['Residence_type'] = le.fit_transform(data['Residence_type'])
data['smoking_status'] = le.fit_transform(data['smoking_status'])

#train test split
X = data.drop('stroke', axis=1)  # Fitur
y = data['stroke']  # Variabel target
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    gender = st.sidebar.slider('gender', 0, 1, 1)
    age = st.sidebar.slider('usia',0,20,1)
    hyper = st.sidebar.slider('Darah Tinggi',0,1,0)
    heart= st.sidebar.slider('Sakit Jantung',0,1,0)
    married = st.sidebar.slider('Status NIkah',0,1,0)
    work = st.sidebar.slider('Kerja',0,3,1)
    residen = st.sidebar.slider('Tempat Tinggal',0,2,2)
    gula = st.sidebar.slider('Gula darah', 22.05,280.0,20.0)
    bmi = st.sidebar.slider('bmi',24,88,24)
    smoke=st.sidebar.slider('Type perokok',0,3,1)
    user_report_data = {
        'gender': gender,
        'age': age,
        'hypertension':hyper,
        'heart_disease':heart,
        'ever_married':married,
        'work_type':work,
        'Residence_type':residen,
        'avg_glucose_level':gula,
        'bmi':bmi,
        'smoking_status':smoke
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Pasion
user_data = user_report()
st.subheader('Data Pasien')
st.write(user_data)

user_result = xgb.predict(user_data)
xgb_score = accuracy_score(y_test,xgb.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Kamu Aman'
else:
    output ='Kamu terkena Stroke'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(xgb_score*100)+'%')


user_result = xgb.predict(user_data)
xgb_score = accuracy_score(y_test,xgb.predict(X_test))



