import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, log_loss
from sklearn.calibration import CalibratedClassifierCV
#pull data from input filepath
def load_data(path):
    return pd.read_csv(path)

#Dataframe cleaning process
def cleaning(df):
    # Drop columns we decided are not useful for baseline modeling
    df = df.drop(columns=["Ticket", "Cabin", "Name"], errors="ignore")

    # Fill a few missing values
    df["Age"] = df["Age"].fillna(df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    
    #create family size predictor collapsed column
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    
    #isolate solo travelers
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    #group by age
    df["AgeGroup"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, 100])

    df = df.drop(columns=["SibSp", "Parch"])
    return df



def encode_features(df):
    # Convert categorical columns into numeric columns
    df = pd.get_dummies(df, columns=["Sex", "Embarked","AgeGroup"], drop_first=True)
    return df


def split_features_target(df):
    # Removing ID here instead of in cleaning process to keep an identifier in the data frame
    # Removing PassengerID from prediction process due to cardinality
    X = df.drop(columns=["Survived", "PassengerId"])
    y = df["Survived"]
    return X, y



def train_model(X_train, y_train):
    base_model = RandomForestClassifier(random_state=217)

    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test):
    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities for log loss
    y_probs = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    loss = log_loss(y_test, y_probs)

    # Output
    print("=== Model Results ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Log Loss:  {loss:.4f}")

    print("\n=== Classification Report ===")
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
    
