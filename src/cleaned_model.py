import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

#pull data from input filepath
def load_data(path):
    return pd.read_csv(path)

#Dataframe cleaning process
def cleaning(df):
    # Drop columns we decided are not useful for baseline modeling
    df = df.drop(columns=["Name", "Ticket", "Cabin"], errors="ignore")

    # Fill a few missing values in the simplest reasonable way
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    #create family size predictor collapsed column
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df = df.drop(columns=["SibSp", "Parch"])
    
    #isolate solo travelers
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    return df


def encode_features(df):
    # Convert categorical columns into numeric columns
    df = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)
    return df


def split_features_target(df):
    # Removing ID here instead of in cleaning process to keep an identifier in the data frame
    # Removing PassengerID from prediction process due to cardinality
    X = df.drop(columns=["Survived", "PassengerId"])
    y = df["Survived"]
    return X, y



def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    print("Initial Model Results")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")

    print("\nClassification Report")
    print(classification_report(y_test, y_pred))


def main():
    df = load_data("data/raw/train.csv")
    df = cleaning(df)
    df = encode_features(df)

    X, y = split_features_target(df)
    #validation
    print("Using features:", X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=217,
        stratify=y
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
    
