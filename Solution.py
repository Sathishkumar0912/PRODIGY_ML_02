import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Set the dataset path here
dataset_path = r"C:\Users\sathi\OneDrive\Desktop\WORKSPACE\Prodigy\TASK-2\archive (1)\Mall_Customers.csv"  # <-- Replace this with your actual dataset path

# Check if the dataset exists
if not os.path.exists(dataset_path):
    print(f"Error: The file '{dataset_path}' does not exist.")
    exit(1)

# Load dataset
df = pd.read_csv(dataset_path)

# Convert 'CustomerID' to integer if needed (assuming it's numeric in text format)
df['CustomerID'] = df['CustomerID'].astype(int)

# Replace 'Gender' column values if they are encoded as integers (assuming 0 for Female, 1 for Male)
df['Gender'] = df['Gender'].replace({0: 'Female', 1: 'Male'})

# Select features for clustering (excluding Spending Score since it's empty)
X = df[['Age', 'Annual Income (k$)']]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine optimal number of clusters using Elbow Method
inertia = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot Elbow curve
plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method to Determine Optimal K')
plt.show()

# Apply K-Means with optimal K (assume K=3 based on the elbow method)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Visualizing Clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df['Age'], y=df['Annual Income (k$)'], hue=df['Cluster'], palette='viridis', s=100)
plt.xlabel('Age')
plt.ylabel('Annual Income (k$)')
plt.title('Customer Segments')
plt.legend(title='Cluster')
plt.show()

# Save clustered data
output_path = os.path.join(os.path.dirname(dataset_path), "Mall_Customers_Clustered.csv")
df.to_csv(output_path, index=False)
print(f"Clustered data saved at: {output_path}")
