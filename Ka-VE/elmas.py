import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("diamonds.csv")
df = df.drop(columns=["Unnamed: 0"], errors="ignore") 
encoder = LabelEncoder()
categorical_columns = ["cut", "color", "clarity"]
for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])
y = df["price"]
X = df.drop("price", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("MEA:", mean_absolute_error(y_pred, y_test))
print("MSE:", mean_squared_error(y_pred, y_test))
print("RMSE:", np.sqrt(mean_squared_error(y_pred, y_test)))
print("R2 Score:", r2_score(y_test, y_pred))
# -------------------Streamlit kısmı---------------------
st.title("Elmas Tahmin Uygulaması")
st.markdown("""
    <style>
    body {
        background-color:rgba(255, 255, 255, 0.73);
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .stApp {
        background-color:rgba(252, 252, 252, 0.66);
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

st.image("https://www.labmedya.com/uploads/haberler/elmas.jpg", use_container_width=True)
carat = st.slider("Carat", min_value=0.0, max_value=5.0, value=2.5, step=0.1)
cut_dic = {
    'Fair': 0,
    'Good': 1,
    'Ideal': 2,
    'Premium': 3,
    'Very Good': 4
}
cut = cut_dic[st.sidebar.selectbox("Cut", cut_dic.keys())]

color_dic = {
    'D': 0,
    'E': 1,
    'F': 2,
    'G': 3,
    'H': 4,
    'I': 5,
    'J': 6
}
color = color_dic[st.sidebar.selectbox("Color", color_dic.keys())]

clarity_dic = {
    'I1': 0,
    'SI2': 1,
    'SI1': 2,
    'VS2': 3,
    'VS1': 4,
    'VVS2': 5,
    'VVS1': 6,
    'IF': 7
}
clarity = clarity_dic[st.sidebar.selectbox("Clarity", clarity_dic.keys())]

table = st.sidebar.slider(label="Table", min_value=40.0, max_value=100.0, value=70.0, step=0.1)
depth = st.sidebar.slider(label="Depth", min_value=30.0, max_value=80.0, value=40.0, step=0.1)
x = st.sidebar.slider(label="X", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
y = st.sidebar.slider(label="Y", min_value=0.0, max_value=30.0, value=30.0, step=0.1)
z = st.sidebar.slider(label="Z", min_value=2.0, max_value=20.0, value=20.0, step=0.1)
features = {
    "carat": carat,
    "cut": cut,
    "color": color,
    "clarity": clarity,
    "depth": depth,
    "table":table,
    "x": x,
    "y": y,
    "z": z
}
features_df = pd.DataFrame([features])

st.table(features_df)
if st.button("Fiyat Tahmin Et"):
    pred = model.predict(features_df)
    st.write(f"Tahmin edilen değer: {pred[0]}")
