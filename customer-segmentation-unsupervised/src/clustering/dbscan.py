import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


def apply_dbscan(data, eps=2.0, min_samples=5):

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(data)

    unique_labels = set(labels)
    n_clusters = len(unique_labels - {-1})
    n_noise = list(labels).count(-1)

    print("\nDBSCAN Clustering Results")
    print("---------------------------------")
    print("Number of clusters:", n_clusters)
    print("Number of noise points:", n_noise)

    print("\nCluster Distribution:")
    for label in sorted(unique_labels):
        count = list(labels).count(label)
        print(f"Cluster {label} -> {count} customers")

    return labels



if __name__ == "__main__":

    df = pd.read_csv("customer-segmentation-unsupervised/data/churn-bigml.csv")

    df_numeric = df.select_dtypes(include=["int64", "float64"])

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    apply_dbscan(scaled_data, eps=2.0, min_samples=5)
