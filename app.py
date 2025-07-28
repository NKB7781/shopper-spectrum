import streamlit as st
def load_data():
    rfm = pd.read_csv("rfm_segments.csv", index_col=0)
    sim_matrix = pd.read_csv("product_similarity.csv", index_col=0)
    return rfm, sim_matrix

rfm_data, similarity_data = load_data()

st.title("🛒 Shopper Spectrum")
st.subheader("Customer Segmentation & Product Recommendation")

# Tabs
tab1, tab2 = st.tabs(["Product Recommendation", "Customer Segmentation"])

with tab1:
    st.header("🌍 Product Recommendation")
    product_name = st.text_input("Enter a Product StockCode (e.g. 85123A):")
    if st.button("Get Recommendations"):
        if product_name in similarity_data.columns:
            sim_scores = similarity_data[product_name].sort_values(ascending=False)[1:6]
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