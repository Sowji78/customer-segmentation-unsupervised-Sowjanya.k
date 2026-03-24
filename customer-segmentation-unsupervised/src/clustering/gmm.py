
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def apply_gmm(data, n_components=3, random_state=42):
    gmm = GaussianMixture(
        n_components=n_components,
        random_state=random_state
    )

    labels = gmm.fit_predict(data)

    print("\nGMM Clustering Results")
    print("---------------------------------")
    print("Number of clusters:", n_components)

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

    apply_gmm(scaled_data, n_components=3)
