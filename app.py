import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

st.title("Breast Cancer Prediction App")
st.write("Hello world please use App tp predict Cancer Status")

#load the data set
bc = load_breast_cancer()
df = pd.DataFrame(bc.data, columns=bc.feature_names)
df['target'] = bc.target


X=df.drop('target',axis=1) #Input Features
y=df['target'] #target Features

st.subheader("Data Overview of the first 10 rows")
st.dataframe(df.head(10))

X_train, X_test, y_train, y_test =train_test_split(X, y, test_size =0.2,random_state=42)

#scaling tha data 
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# create a decision tree model
dtc=DecisionTreeClassifier() #initialize model
dtc.fit(X_train,y_train) #fit the model

# make predictions
y_pred = dtc.predict(X_test_sc)

# evaluate the model
Ac = accuracy_score(y_test,y_pred)
Ps = precision_score(y_test,y_pred)
Rs = recall_score(y_test,y_pred)
F1 = f1_score(y_test,y_pred)
Cm= confusion_matrix(y_test,y_pred)


st.subheader("Model Performance Metrics")
st.write(f"Accuracy Score: {Ac}")
st.write(f"Precison Score: {Ps}")
st.write(f"Recall Score: {Rs}")
st.write(f"f1 Score: {F1}")
st.write(f"Confusion Matrix: {Cm}")

st.subheader("Predict Cancer Likelihood for New Data")
user_data = []
for feature in X.columns:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_data.append(value)

if st.button("Predict"):
    user_data = scaler.transform([user_data])
    prediction = dtc.predict(user_data)
    st.write("Prediction:", "Malignant" if prediction[0] == 0 else "Benign")
    




