import pandas as pd

def load_data(path):
    return pd.read_csv(path)

def audit_data(df):
    print("=== BASIC INFO ===")
    print(df.info())

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== SUMMARY STATS ===")
    print(df.describe())

    print("\n=== UNIQUE VALUES ===")
    print(df.nunique())

def main():
    df = load_data("data/raw/train.csv")
    audit_data(df)

if __name__ == "__main__":
    main()
