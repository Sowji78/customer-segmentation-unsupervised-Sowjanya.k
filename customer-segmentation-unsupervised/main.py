import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def main():

    print("Loading dataset...")
    file_path = "customer-segmentation-unsupervised/data/churn-bigml.csv"

    df = pd.read_csv(file_path)

    print("Dataset loaded successfully!")
    print("Shape of dataset:", df.shape)
    print("\nPreprocessing data...")
    df_numeric = df.select_dtypes(include=['int64', 'float64'])
    if "Churn" in df_numeric.columns:
        df_numeric = df_numeric.drop("Churn", axis=1)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    print("Preprocessing completed!")
    print("\nTraining KMeans model...")

    n_clusters = 3
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)

    print("Clustering completed!")
    score = silhouette_score(scaled_data, labels)

    print("\n===== RESULTS =====")
    print("Number of Clusters:", n_clusters)
    print("Silhouette Score:", round(score, 4))
    print("\nSaving results...")

    df["Cluster"] = labels

    os.makedirs("results", exist_ok=True)

    df.to_csv("results/clustered_output.csv", index=False)

    print("Results saved to: results/clustered_output.csv")
    print("\nProgram completed successfully!")


if __name__ == "__main__":
    main()
