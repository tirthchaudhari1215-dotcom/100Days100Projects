import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
import matplotlib.pyplot as plt
import joblib


# --- 1. Data Loading and Clustering Function ---

@st.cache_resource
def load_and_cluster_data(csv_path='Mall_Customers.csv'):
    """Loads data, determines optimal K, trains KMeans, and returns model components."""
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        st.error(f"Error: '{csv_path}' not found. Please download the dataset.")
        return None, None, None

    # **CRITICAL FIX:** Corrected the column name from 'Spending Score (1-1-100)'
    # to the widely used 'Spending Score (1-100)' to resolve KeyError.
    X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

    # Determine Optimal K using the Elbow Method
    # We use manual WCSS calculation as the yellowbrick import caused the distutils error.
    wcss = []  # Within-Cluster Sums of Squares
    max_k = 11

    for i in range(1, max_k):
        # Setting n_init=10 explicitly to silence a scikit-learn warning
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Optimal K is usually where the bend (elbow) is, often 5 for this dataset.
    optimal_k = 5

    # Train the final K-Means model with the optimal K
    final_model = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    clusters = final_model.fit_predict(X)

    # Add cluster labels to the DataFrame
    X['Cluster'] = clusters

    # Save Model for future use
    joblib.dump(final_model, 'mall_kmeans_model.pkl')

    return X, wcss, optimal_k


# Load clustered data and WCSS results
clustered_df, wcss_results, optimal_k = load_and_cluster_data()

# --- 2. Streamlit Application UI ---

st.set_page_config(page_title="Day 3: Customer Segmentation")

st.title("🛍️ Day 3: Mall Customer Segmentation (K-Means)")
st.subheader("Unsupervised Learning for Market Analysis")
st.markdown("This project segments customers based on their Annual Income and Spending Score.")

if clustered_df is not None:
    # --- 3. Elbow Method Plot ---
    st.write("### 📐 1. Determining the Optimal Number of Clusters (K)")
    st.info(
        f"Using the Elbow Method, the optimal number of clusters (K) is determined to be **{optimal_k}** for this dataset.")

    # Create the Elbow Plot using Matplotlib
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, 11), wcss_results, marker='o')
    ax.set_title('Elbow Method')
    ax.set_xlabel('Number of Clusters (K)')
    ax.set_ylabel('WCSS (Inertia)')
    ax.axvline(x=optimal_k, color='r', linestyle='--', label=f'Optimal K={optimal_k}')
    ax.legend()
    st.pyplot(fig)  # Display the Matplotlib plot

    # --- 4. Cluster Visualization ---
    st.write("### 📊 2. K-Means Clustering Results")
    st.markdown("The scatter plot below shows the 5 distinct customer segments found by the model.")

    # Use Plotly for interactive visualization
    fig_scatter = px.scatter(
        clustered_df,
        x='Annual Income (k$)',
        y='Spending Score (1-100)',  # Using the correct column name here too
        color='Cluster',
        hover_data=['Annual Income (k$)', 'Spending Score (1-100)'],
        title="Customer Segments (K=5)",
        color_continuous_scale=px.colors.qualitative.Plotly
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")
    st.write("### 💡 Segment Interpretation:")
    st.write(
        """
        **Segment 0 (Frugal):** Low Income, Low Spenders\n
        **Segment 1 (Average):** Average Income, Average Spenders\n
        **Segment 2 (Target/VIP):** High Income, High Spenders\n
        **Segment 3 (Careful):** Low Income, High Spenders (Perhaps promotional shoppers)\n
        **Segment 4 (Conservative):** High Income, Low Spenders (Likely saving or spending elsewhere)
        """
    )