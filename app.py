import streamlit as st
import pickle
import numpy as np
import sklearn as sk

st.set_page_config(page_title="Customer Segment Predictor", layout="centered")

st.title("Mall Customer Segmentation App")
st.write("enter you annual and monthly score")

try:
    with open('customer_model.pkl','rb') as f:
        model =pickle.load(f)
except FileNotFoundError:
    st.error("customer_model.pkl")

income =st.slider("(Annual Income)", 10,150,70)
spending = st.slider("Spending Score (1-100)",1,100,50)

if st.button("Predict My Cluster"):
    user_data = np.array([[income,spending]])
    Prediction = model.predict(user_data)
    cluster_id = Prediction[0]
    clusters = {
        0: "High Income, Low Spending(Target for Prediction)",
        1: "Average Income, Average Spending (Standard Customer)",
        2: "High Income, High Spending (VIP Customer/Diamond)",
        3: "Low Income, High Spending (Careless Spender)",
        4: "low Income, Low Spending (Sensible/Budget Shopper)"
    }

    st.success(f"Aap Cluster {cluster_id} mein hain!")
    st.info(f"That Means: {clusters[cluster_id]}")