import pandas as pd

from tools import load_data, train_model, split_features_target, prepare_combined, THRESHOLD


def predict_new_data(model, X_new, threshold=THRESHOLD):
    y_probs = model.predict_proba(X_new)[:, 1]
    return (y_probs > threshold).astype(int)


def save_predictions(passenger_ids, predictions, output_path):
    results = pd.DataFrame({
        "PassengerId": passenger_ids,
        "Survived": predictions,
    })
    results.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    train_df = load_data("data/raw/train.csv")
    test_df  = load_data("data/raw/test.csv")

    train, test = prepare_combined(train_df, test_df)

    X_train, y_train = split_features_target(train)[0], split_features_target(train)[1]
    X_test = test.drop(columns=["Survived", "PassengerId"], errors="ignore")
    passenger_ids = test_df["PassengerId"]

    model = train_model(X_train, y_train)
    predictions = predict_new_data(model, X_test)
    save_predictions(passenger_ids, predictions, "outputs/test_predictions.csv")


if __name__ == "__main__":
    main()
