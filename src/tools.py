import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV

THRESHOLD = 0.45
KNOWN_TITLES = ["Mr", "Mrs", "Miss", "Master", "Rare"]

RF_PARAMS = dict(
    max_depth=12,
    max_features="sqrt",
    min_samples_leaf=2,
    n_estimators=200,
    random_state=217,
)


def load_data(path):
    return pd.read_csv(path)


def clean_data(df, age_median=None, embarked_mode=None):
    df = df.copy()

    # Fill Data
    df["Age"] = df["Age"].fillna(age_median if age_median is not None else df["Age"].median())
    df["Embarked"] = df["Embarked"].fillna(embarked_mode if embarked_mode is not None else df["Embarked"].mode()[0])

    # Title, collapse unknown titles to Rare
    df["Title"] = df["Name"].str.extract(r", (\w+)\.")
    df["Title"] = df["Title"].apply(lambda x: x if x in KNOWN_TITLES else "Rare")

    # Family features
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)

    # Age groups
    df["AgeGroup"] = pd.cut(
        df["Age"],
        bins=[0, 12, 18, 35, 60, 120],
        labels=["child", "teen", "adult", "middle", "senior"],
    )

    # Pclass/Sex interaction
    df["Sex_num"] = (df["Sex"] == "male").astype(int)
    df["Pclass_Sex"] = df["Pclass"] * df["Sex_num"]

    df = df.drop(columns=["Name", "Cabin", "SibSp", "Parch", "Ticket"], errors="ignore")

    return df


def encode_features(df):
    df = pd.get_dummies(
        df,
        columns=["Sex", "Embarked", "Title", "AgeGroup"],
        drop_first=True,
    )
    return df


def split_features_target(df):
    X = df.drop(columns=["Survived", "PassengerId"])
    y = df["Survived"]
    return X, y


def train_model(X_train, y_train):
    base_model = RandomForestClassifier(**RF_PARAMS)
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=5)
    model.fit(X_train, y_train)
    return model


def get_training_stats(df):
    age_median = df["Age"].median()
    embarked_mode = df["Embarked"].mode()[0]
    return age_median, embarked_mode


def prepare_combined(train_df, test_df):

    age_median, embarked_mode = get_training_stats(train_df)

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["_is_train"] = 1
    test_df["_is_train"] = 0

    combined = pd.concat([train_df, test_df], ignore_index=True)
    combined = clean_data(combined, age_median, embarked_mode)
    combined = encode_features(combined)

    train_out = combined[combined["_is_train"] == 1].drop(columns=["_is_train"])
    test_out  = combined[combined["_is_train"] == 0].drop(columns=["_is_train"])

    return train_out, test_out
