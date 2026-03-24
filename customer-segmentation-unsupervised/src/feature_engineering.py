import pandas as pd

def perform_feature_engineering(df):
    """
    Perform simple feature engineering on churn dataset
    """

    print("\nPerforming Feature Engineering...")
    print("-------------------------------------")
    df["Total_minutes"] = (
        df["Total day minutes"] +
        df["Total eve minutes"] +
        df["Total night minutes"] +
        df["Total intl minutes"]
    )
    df["Total_calls"] = (
        df["Total day calls"] +
        df["Total eve calls"] +
        df["Total night calls"] +
        df["Total intl calls"]
    )
    df["Total_charge"] = (
        df["Total day charge"] +
        df["Total eve charge"] +
        df["Total night charge"] +
        df["Total intl charge"]
    )
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype("category").cat.codes

    print("New Features Created Successfully!")
    print("\nUpdated Columns:")
    print(df.columns)

    print("\nUpdated Dataset Shape:", df.shape)

    return df
from feature_engineering import perform_feature_engineering

df = pd.read_csv("customer-segmentation-unsupervised/data/churn-bigml.csv")

df = perform_feature_engineering(df)
