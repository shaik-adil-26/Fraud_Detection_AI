import os
import pandas as pd
import numpy as np
import joblib
import streamlit as st

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


MODEL_FILE = "fraud_kmeans_model.pkl"
SCALER_FILE = "scaler.pkl"
DATA_FILE = "fraud_dataset.csv"
CLEAN_FILE = "fraud_dataset_cleaned.csv"


def generate_raw_dataset(n_rows=300):
    np.random.seed(42)

   
    amount_normal = np.random.normal(loc=500, scale=200, size=n_rows).clip(50, 1500)
    items_normal = np.random.randint(1, 6, size=n_rows)
    distance_normal = np.random.normal(loc=3, scale=2, size=n_rows).clip(0.5, 15)

    
    outliers = 15
    amount_outliers = np.random.uniform(5000, 15000, size=outliers)
    items_outliers = np.random.randint(10, 25, size=outliers)
    distance_outliers = np.random.uniform(20, 60, size=outliers)

    amount = np.concatenate([amount_normal, amount_outliers])
    items = np.concatenate([items_normal, items_outliers])
    distance = np.concatenate([distance_normal, distance_outliers])

    df = pd.DataFrame({
        "TransactionID": range(1, len(amount) + 1),
        "Amount": amount.round(2),
        "Items": items,
        "Distance_km": distance.round(2)
    })

   
    for col in ["Amount", "Items", "Distance_km"]:
        missing_index = np.random.choice(df.index, size=8, replace=False)
        df.loc[missing_index, col] = np.nan

    df = pd.concat([df, df.sample(5, random_state=42)], ignore_index=True)

    df.to_csv(DATA_FILE, index=False)
    return df

def clean_dataset(df):
    df = df.copy()

    
    df = df.drop_duplicates()

  
    for col in ["Amount", "Items", "Distance_km"]:
        df[col] = df[col].fillna(df[col].median())

    df["Items"] = df["Items"].astype(int)

    df.to_csv(CLEAN_FILE, index=False)
    return df


def train_and_save_model(df, k=3):
    X = df[["Amount", "Items", "Distance_km"]]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    joblib.dump(kmeans, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)

    return kmeans, scaler



def load_or_train_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(SCALER_FILE):
        kmeans = joblib.load(MODEL_FILE)
        scaler = joblib.load(SCALER_FILE)
        return kmeans, scaler, "Loaded existing model files."

   
    raw_df = generate_raw_dataset()
    clean_df = clean_dataset(raw_df)
    kmeans, scaler = train_and_save_model(clean_df)

    return kmeans, scaler, "Trained new model and saved model files."


st.set_page_config(page_title="Fraud Detection AI", page_icon="🛡️", layout="centered")

st.title("🛡️ Fraud Detection AI (K-Means Clustering)")
st.write("Industry AI Pipeline: **Raw Data → Cleaning → Model → Deployment UI**")

kmeans, scaler, status = load_or_train_model()
st.info(status)


with st.expander("📂 View Raw Dataset Preview"):
    if os.path.exists(DATA_FILE):
        df_raw = pd.read_csv(DATA_FILE)
        st.write(df_raw.head(10))
        st.write("Raw Dataset Shape:", df_raw.shape)

with st.expander("🧹 View Cleaned Dataset Preview"):
    if os.path.exists(CLEAN_FILE):
        df_clean = pd.read_csv(CLEAN_FILE)
        st.write(df_clean.head(10))
        st.write("Cleaned Dataset Shape:", df_clean.shape)

st.subheader("Enter Transaction Details")

amount = st.number_input("Transaction Amount", min_value=0.0, value=500.0, step=50.0)
items = st.number_input("Number of Items", min_value=1, value=2, step=1)
distance = st.number_input("Delivery Distance (km)", min_value=0.0, value=3.0, step=0.5)

if st.button("Predict Risk"):
    new_data = np.array([[amount, items, distance]])
    new_scaled = scaler.transform(new_data)

    predicted_cluster = int(kmeans.predict(new_scaled)[0])

   
    cluster_centers = kmeans.cluster_centers_
    suspicious_cluster = int(np.argmax(cluster_centers[:, 0]))

    risk = "Suspicious" if predicted_cluster == suspicious_cluster else "Normal"

    st.write("---")
    st.write("### Prediction Result")
    st.write("Predicted Cluster:", predicted_cluster)

    if risk == "Suspicious":
        st.error("⚠️ Risk: Suspicious Transaction")
    else:
        st.success("✅ Risk: Normal Transaction")

st.write("---")
st.caption("Prediction done successfully.")
