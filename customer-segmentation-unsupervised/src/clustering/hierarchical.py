import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler


def apply_hierarchical(data, n_clusters=3):
    model = AgglomerativeClustering(n_clusters=n_clusters)
    labels = model.fit_predict(data)

    print("\nHierarchical Clustering Results")
    print("---------------------------------")
    print("Number of clusters:", n_clusters)

    print("\nCluster Distribution:")
    unique_labels = set(labels)
    for label in sorted(unique_labels):
        count = list(labels).count(label)
        print(f"Cluster {label} -> {count} customers")

    return labels

if __name__ == "__main__":

    df = pd.read_csv("customer-segmentation-unsupervised/data/churn-bigml.csv")
    df_numeric = df.select_dtypes(include=["int64", "float64"])
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)
    apply_hierarchical(scaled_data, n_clusters=3)
