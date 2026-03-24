import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)

    print("Original Dataset Shape:", df.shape)
    print("\nFirst 5 Rows:")
    print(df.head())
    df_numeric = df.select_dtypes(include=['int64', 'float64'])

    print("\nNumeric Columns:")
    print(df_numeric.columns)
    if 'Churn' in df_numeric.columns:
        df_numeric = df_numeric.drop('Churn', axis=1)
        print("\nRemoved 'Churn' column")

    print("\nMissing Values:")
    print(df_numeric.isnull().sum())

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_numeric)

    print("\nData Successfully Scaled!")
    print("Scaled Data Shape:", scaled_data.shape)

    return scaled_data, df_numeric
from data_preprocessing import load_and_preprocess

data, df = load_and_preprocess("customer-segmentation-unsupervised/data/churn-bigml.csv")
