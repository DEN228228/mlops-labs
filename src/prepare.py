import pandas as pd
import os
import argparse
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Підготовка та розділення даних")
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.04,
        help="Частка тренувальної вибірки (від 0.0 до 1.0, за замовчуванням 0.8)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.01,
        help="Частка тестової вибірки (від 0.0 до 1.0, за замовчуванням 0.2)",
    )
    args = parser.parse_args()

    if args.train_size + args.test_size > 1.0:
        raise ValueError(
            "Помилка: Сума train_size та test_size не може перевищувати 1.0"
        )

    df = pd.read_csv("data/raw/cleaned_house_rent_data_cropped.csv")

    categorical_cols = df.select_dtypes(include=["object"]).columns
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df.drop(columns=["Rent"])
    y = df["Rent"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=args.train_size, test_size=args.test_size, random_state=42
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    processed_path = "data/processed/"
    os.makedirs(processed_path, exist_ok=True)

    train_file = "train.csv"
    test_file = "test.csv"

    train_df.to_csv(os.path.join(processed_path, train_file), index=False)
    test_df.to_csv(os.path.join(processed_path, test_file), index=False)

    print(
        f"Тренувальний датасет збережено у: {os.path.join(processed_path, train_file)}"
    )
    print(f"Тестовий датасет збережено у: {os.path.join(processed_path, test_file)}")


if __name__ == "__main__":
    main()
