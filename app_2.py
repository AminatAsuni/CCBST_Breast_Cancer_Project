import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

scaler_model = pickle.load(open('models\model_scaler.pkl','rb'))
logistic_model = pickle.load(open('models\model_lr.pkl','rb'))
dtc_model = pickle.load(open('models\model_dtc.pkl','rb'))

#load_data
bc = load_breast_cancer()
bc_df = pd.DataFrame(bc.data,columns=bc.feature_names)


st.title("This is a web app to predict whether a person is malignant or Benign")
models = {
    "Logistic Regression":logistic_model, "Decision Tree": dtc_model,
     }

#user select the model
selected_model= st.selectbox("Select a model",list(models.keys()))

final_model= models[selected_model]


input_data ={}

for col in bc_df.columns:
    input_data[col] = st.slider(col, min_value= bc_df[col].min(),max_value=bc_df[col].max())

#convert dict to df
input_df = pd.DataFrame([input_data])

st.write(input_df)

input_df_scaled = scaler_model.transform(input_df)

if st.button("Predict"):
    predicted = final_model.predict(input_df_scaled)[0]
    predicted_prob = final_model.predict_proba(input_df_scaled)[0]

#display prediction
    if predicted == 0:
        st.write("The person is Malignant")
    else:
        st.write("The person is Benign")


