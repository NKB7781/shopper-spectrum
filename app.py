import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import gdown
import os

# ----------------------------------------
# Download & Load Dataset from Google Drive
# ----------------------------------------
file_id = "1rzRwxm_CJxcRzfoo9Ix37A2JTlMummY-"
csv_file = "online_retail.csv"

if not os.path.exists(csv_file):
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, csv_file, quiet=False)

@st.cache_data
def load_data():
    df = pd.read_csv(csv_file, encoding='ISO-8859-1')
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df.dropna(subset=['CustomerID'], inplace=True)
    df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
    df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
    df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
    return df

df = load_data()

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

# Determine Optimal Clusters with Silhouette Score
scores = []
for k in range(2, 10):
    model = KMeans(n_clusters=k, random_state=42)
    model.fit(rfm_scaled)
    score = silhouette_score(rfm_scaled, model.labels_)
    scores.append(score)

best_k = np.argmax(scores) + 2
kmeans = KMeans(n_clusters=best_k, random_state=42)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

# Labeling clusters manually (optional tweak)
cluster_map = {i: f"Segment {i}" for i in range(best_k)}
rfm['Segment'] = rfm['Cluster'].map(cluster_map)

# ----------------------------------------
# Product Similarity Matrix
# ----------------------------------------
pivot = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', fill_value=0)
sim_matrix = pd.DataFrame(cosine_similarity(pivot.T), index=pivot.columns, columns=pivot.columns)

# Prepare product description map
desc_map = df.drop_duplicates(subset='StockCode')[['StockCode', 'Description']].set_index('StockCode')['Description'].fillna("No Description")

# ----------------------------------------
# Streamlit Interface
# ----------------------------------------
st.set_page_config(page_title="Shopper Spectrum", layout="wide")
st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation")

# Tabs for interaction
tab1, tab2, tab3 = st.tabs(["📦 Product Recommendation", "📊 Customer Segmentation", "📈 EDA"])

with tab1:
    st.header("🌍 Product Recommendation")
    product_name = st.text_input("Enter a Product StockCode (e.g. 85123A):")
    if st.button("Get Recommendations"):
        if product_name in sim_matrix.columns:
            sim_scores = sim_matrix[product_name].sort_values(ascending=False)[1:6]
            st.write("### Recommended Products:")
            for i, prod in enumerate(sim_scores.index, 1):
                desc = desc_map.get(prod, "No Description")
                st.markdown(f"**{i}. {prod}** — {desc}")
        else:
            st.error("Product not found in dataset.")

with tab2:
    st.header("🔍 Customer Segmentation")
    rec = st.number_input("Recency (days since last purchase):", min_value=0)
    freq = st.number_input("Frequency (number of purchases):", min_value=0)
    mon = st.number_input("Monetary (total spend):", min_value=0.0)

    if st.button("Predict Segment"):
        input_scaled = scaler.transform([[rec, freq, mon]])
        cluster = kmeans.predict(input_scaled)[0]
        label = cluster_map.get(cluster, "Unknown")
        st.success(f"Predicted Customer Segment: {label}")

with tab3:
    st.header("📊 Exploratory Data Analysis")

    st.subheader("Top 10 Selling Products")
    top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
    fig1, ax1 = plt.subplots()
    sns.barplot(x=top_products.values, y=top_products.index, ax=ax1)
    ax1.set_xlabel("Total Quantity Sold")
    ax1.set_ylabel("Product")
    st.pyplot(fig1)

    st.subheader("Sales by Country")
    top_countries = df.groupby('Country')['TotalPrice'].sum().sort_values(ascending=False).head(10)
    fig2, ax2 = plt.subplots()
    sns.barplot(x=top_countries.values, y=top_countries.index, ax=ax2)
    ax2.set_xlabel("Total Sales")
    ax2.set_ylabel("Country")
    st.pyplot(fig2)

    st.subheader("📌 Optimal Clusters (Silhouette Scores)")
    fig3, ax3 = plt.subplots()
    ax3.plot(range(2, 10), scores, marker='o')
    ax3.set_title("Silhouette Score vs Number of Clusters")
    ax3.set_xlabel("k")
    ax3.set_ylabel("Silhouette Score")
    st.pyplot(fig3)

