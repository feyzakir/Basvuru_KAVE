import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, r2_score,accuracy_score
from sklearn.datasets import load_iris

df=load_iris()

X=df.data
y=df.target

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

model=RandomForestClassifier(n_estimators=200,random_state=101)
model.fit(X_train,y_train)

y_pred=model.predict(X_test)

model.score(X_test,y_test)

# mse = mean_squared_error(y_test,y_pred)
# acc=accuracy_score(y_test,y_pred)
# "MSE : % f" %(mse)
# "ACC : % f" %(acc)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# mse = mean_squared_error(y_test,y_pred)
# acc=accuracy_score(y_test,y_pred)
# "MSE : % f" %(mse)
# "ACC : % f" %(acc)

# "Confusion Matrix:",confusion_matrix(y_test, y_pred)
# "Classification Report:",classification_report(y_test, y_pred, target_names=df.target_names)

#Doğruluk oranının 1.00 olması beni şüphelendirdi o yüzden manuel olarak ilk 5 değerin girdi ve çıktılarını kontrol edeceğim.

# for i in range(10):
#     X_sample = X_test[i].reshape(1, -1)
#     y_true = y_test[i]
#     y_pred = model.predict(X_sample)
#     print(f"\nÖrnek {i+1}:")
#     print("X_test:", X_sample)
#     print("Gerçek değer (y_true):", y_true, "->", df.target_names[y_true])
#     print("Tahmin edilen değer (y_pred):", y_pred[0], "->", df.target_names[y_pred[0]])

import streamlit as st
st.markdown("""
    <style>
    body {
        background-color: #2E2E2E;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color: #2E2E2E;
    }
    .stTextInput input {
        background-color: white;
        color: black;
    }
    .stButton button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
    }
    .stMarkdown h1 {
        text-align: center;
        color: white;
    }
    .answer-box {
        background-color: white;
        color: black;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
    }
    .question-text {
        color: white;
        font-weight: bold;
        font-size: 24px;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1>Iris Tahmin</h1>", unsafe_allow_html=True)

image_path = "https://miro.medium.com/v2/resize:fit:1400/1*ZK9_HrpP_lhSzTq9xVJUQw.png"
st.image(image_path, use_container_width=True) 

model=model

sepal_length=st.number_input("Sepal Length")
sepal_width=st.number_input("Sepal Width")
petal_length=st.number_input("Petal Length")
petal_width=st.number_input("Petal Width")

class_names = ["Setosa", "Versicolor", "Virginica"]

if st.button("Tahmin Yap"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]
    st.write("Tahmin Edilen Sınıf:", prediction)
    st.success(f"Tahmin Edilen Tür: {class_names[prediction]}")
    