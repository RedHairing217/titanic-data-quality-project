import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, log_loss

from tools import load_data, clean_data, encode_features, split_features_target, train_model, get_training_stats, THRESHOLD


def evaluate_model(model, X_test, y_test, threshold=THRESHOLD):
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs > threshold).astype(int)

    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    loss      = 1 - accuracy
    logloss   = log_loss(y_test, y_probs)

    print("=== MODEL RESULTS ===")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Loss:      {loss:.4f}")
    print(f"Log Loss:  {logloss:.4f}")
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred))


def main():
    df = load_data("data/raw/train.csv")

    age_median, embarked_mode = get_training_stats(df)
    df = clean_data(df, age_median, embarked_mode)
    df = encode_features(df)

    X, y = split_features_target(df)
    print("Using features:", list(X.columns))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=217,
        stratify=y,
    )

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)


if __name__ == "__main__":
    main()
