import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def audit_data(df):
    print("Basic Info")
    print(df.info())

    print("\nMissing Values")
    print(df.isnull().sum())

    print("\nSummary Stats")
    print(df.describe())

    print("\nUnique Values")
    print(df.nunique())

def main():
    df = load_data("data/raw/test.csv")
    audit_data(df)

if __name__ == "__main__":
    main()
