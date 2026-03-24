from sklearn.metrics import silhouette_score
import numpy as np

def evaluate_clustering(data, labels):
    score = silhouette_score(data, labels)

    print("\nClustering Evaluation Results")
    print("---------------------------------")
    print("Silhouette Score:", round(score, 4))

    unique, counts = np.unique(labels, return_counts=True)

    print("\nCluster Distribution:")
    for cluster, count in zip(unique, counts):
        print(f"Cluster {cluster} -> {count} customers")

    return score
from data_preprocessing import load_and_preprocess
from sklearn.cluster import KMeans
from src.evaluation import evaluate_clustering

scaled_data, df = load_and_preprocess("churn-bigml.csv")
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled_data)
evaluate_clustering(scaled_data, labels)
