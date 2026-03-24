import pandas as pd

def dataset_summary(df):
    print("\nDataset Summary")
    print("----------------------")
    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns)
    print("\nFirst 5 Rows:")
    print(df.head())


def check_missing_values(df):
    print("\nMissing Values")
    print("----------------------")
    missing = df.isnull().sum()
    print(missing)
    return missing


def save_cluster_results(df, labels, filename="clustered_output.csv"):
    df["Cluster"] = labels
    df.to_csv(filename, index=False)

    print("\nClustered data saved successfully as:", filename)
from utils import dataset_summary, check_missing_values, save_cluster_results
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from utils import dataset_summary, check_missing_values, save_cluster_results
df = pd.read_csv("customer-segmentation-unsupervised/data/churn-bigml.csv")
dataset_summary(df)
check_missing_values(df)
df_numeric = df.select_dtypes(include=['int64', 'float64'])
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_numeric)
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(scaled_data)
save_cluster_results(df, labels)
