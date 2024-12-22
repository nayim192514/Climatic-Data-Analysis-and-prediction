import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Load the dataset
file_path = r"E:\Machine learning\Ml class_cse\Climate data_jashore_2019-2021.csv"
climate_data = pd.read_csv(file_path)

# Selecting features for clustering and ensuring no missing values
clustering_features = climate_data[['Temp_MAX', 'Temp_MIN', 'RH']].dropna()

# Determining the optimal number of clusters using the elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(clustering_features)
    inertia.append(kmeans.inertia_)

# Plotting the elbow curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), inertia, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# Applying KMeans with an optimal number of clusters (e.g., 3)
optimal_clusters = 3
kmeans_model = KMeans(n_clusters=optimal_clusters, random_state=42)
cluster_labels = kmeans_model.fit_predict(clustering_features)

# Adding cluster labels to the data
climate_data['Cluster'] = cluster_labels

# Displaying the first few rows with cluster labels
print(climate_data[['Temp_MAX', 'Temp_MIN', 'RH', 'Cluster']].head())

# Calculating the silhouette score as the accuracy metric
silhouette_avg = silhouette_score(clustering_features, cluster_labels)
print(f"Silhouette Score for {optimal_clusters} clusters: {silhouette_avg:.4f}")

# Visualizing the clusters
plt.figure(figsize=(10, 6))
for cluster in range(optimal_clusters):
    cluster_data = clustering_features[cluster_labels == cluster]
    plt.scatter(cluster_data['Temp_MIN'], cluster_data['Temp_MAX'], label=f'Cluster {cluster}', alpha=0.7)

# Marking centroids
centroids = kmeans_model.cluster_centers_
plt.scatter(centroids[:, 1], centroids[:, 0], c='black', marker='X', s=200, label='Centroids')

plt.title('KMeans Clustering: Temp_MIN vs Temp_MAX')
plt.xlabel('Temp_MIN')
plt.ylabel('Temp_MAX')
plt.legend()
plt.grid(True)
plt.show()



