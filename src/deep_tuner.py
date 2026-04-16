import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from tools import load_data, clean_data, encode_features, split_features_target, train_model, get_training_stats


def evaluate_thresholds(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]

    print("| Threshold | Accuracy | Precision | Recall | Loss   | Log Loss |")
    print("|-----------|----------|-----------|--------|--------|----------|")

    for t in np.arange(0.25, 0.501, 0.025):
        y_pred = (y_probs > t).astype(int)

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred)
        loss      = 1 - accuracy
        logloss   = log_loss(y_test, y_probs)

        print(f"| {t:.3f}     | {accuracy:.4f}   | {precision:.4f}    | {recall:.4f} | {loss:.4f} | {logloss:.4f}   |")


def main():
    df = load_data("data/raw/train.csv")

    age_median, embarked_mode = get_training_stats(df)
    df = clean_data(df, age_median, embarked_mode)
    df = encode_features(df)

    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=217,
        stratify=y,
    )

    # Threshold tuning
    model = train_model(X_train, y_train)
    evaluate_thresholds(model, X_test, y_test)

    # Hyperparameter grid search
    param_grid = {
        "n_estimators": [200, 300],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_leaf": [2, 3],
        "max_features": ["sqrt"],
    }
    grid = GridSearchCV(
        RandomForestClassifier(random_state=217),
        param_grid, cv=10, scoring="accuracy", n_jobs=-1
    )
    grid.fit(X_train, y_train)
    print(grid.best_params_, grid.best_score_)


if __name__ == "__main__":
    main()
