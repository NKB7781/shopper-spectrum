import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import gdown
import os

# ----------------------------------------
# Load Dataset from Google Drive
# ----------------------------------------
file_id = "1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-"
csv_file = "online_retail.csv"

if not os.path.exists(csv_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, csv_file, quiet=False)

df = pd.read_csv(csv_file, encoding='ISO-8859-1')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df.dropna(subset=['CustomerID'], inplace=True)
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# ----------------------------------------
# RFM Feature Engineering
# ----------------------------------------
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
})
rfm.columns = ['Recency', 'Frequency', 'Monetary']
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)

# Clustering
kmeans = KMeans(n_clusters=4, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

cluster_map = {
    0: 'High-Value',
    1: 'Regular',
    2: 'Occasional',
    3: 'At-Risk'
}
rfm['Segment'] = rfm['Cluster'].map(cluster_map)

# ----------------------------------------
# Product Similarity Matrix
# ----------------------------------------
pivot = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
sim_matrix = pd.DataFrame(cosine_similarity(pivot.T), index=pivot.columns, columns=pivot.columns)

# ----------------------------------------
# Streamlit Interface
# ----------------------------------------
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation")

tab1, tab2 = st.tabs(["Product Recommendation", "Customer Segmentation"])

with tab1:
    st.header("🌍 Product Recommendation")
    product_name = st.text_input("Enter a Product StockCode (e.g. 85123A):")
    if st.button("Get Recommendations"):
        if product_name in sim_matrix.columns:
            sim_scores = sim_matrix[product_name].sort_values(ascending=False)[1:6]
            st.write("### Recommended Products (StockCodes):")
            for i, prod in enumerate(sim_scores.index, 1):
                st.write(f"{i}. {prod}")
        else:
            st.error("Product not found in dataset.")

with tab2:
    st.header("🔍 Customer Segmentation")
    rec = st.number_input("Recency (days since last purchase):", min_value=0)
    freq = st.number_input("Frequency (number of purchases):", min_value=0)
    mon = st.number_input("Monetary (total spend):", min_value=0.0)

    if st.button("Predict Cluster"):
        input_scaled = scaler.transform([[rec, freq, mon]])
        cluster = kmeans.predict(input_scaled)[0]
        label = cluster_map.get(cluster, "Unknown")
        st.success(f"Predicted Customer Segment: {label}")
