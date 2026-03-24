import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import argparse
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

parser = argparse.ArgumentParser(
    description="Тренування Decision Tree з кастомними параметрами"
)
parser.add_argument(
    "--max_depth", type=int, default=5, help="Максимальна глибина дерева (max_depth)"
)
parser.add_argument(
    "--min_samples_split",
    type=int,
    default=100,
    help="Мінімальна кількість зразків для поділу (min_samples_split)",
)

# Зчитуємо аргументи
args = parser.parse_args()

# 1. Завантаження ПІДГОТОВЛЕНИХ даних
print("Завантаження тренувальної та тестової вибірок...")
processed_path = "data/processed/"
train_df = pd.read_csv(f"{processed_path}train.csv")
test_df = pd.read_csv(f"{processed_path}test.csv")

# Розділення на X (ознаки) та y (цільова змінна)
X_train = train_df.drop(columns=["Rent"])
y_train = train_df["Rent"]

X_test = test_df.drop(columns=["Rent"])
y_test = test_df["Rent"]

# 2. Ініціалізація MLflow
experiment_name = "House-Rent_Decision-Tree_v1"
mlflow.set_experiment(experiment_name)

params = {
    "max_depth": args.max_depth,
    "min_samples_split": args.min_samples_split,
    "random_state": 42,
}

current_run_name = f"DT_depth-{params['max_depth']}_split-{params['min_samples_split']}"
print("Запуск MLflow та тренування моделі...")

with mlflow.start_run(run_name=f"TEST:{current_run_name}"):

    mlflow.set_tags(
        {
            "author": "Denys Svintsylo",
            "model_type": "Decision Tree Regressor",
        }
    )

    # Ініціалізація та тренування моделі
    dt_model = DecisionTreeRegressor(**params)
    dt_model.fit(X_train, y_train)

    y_train_pred = dt_model.predict(X_train)  # Прогноз для тренувальної
    y_test_pred = dt_model.predict(X_test)  # Прогноз для тестової

    # Розрахунок метрик якості
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_r2 = r2_score(y_train, y_train_pred)

    # Для тестової вибірки
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # --- ЛОГУВАННЯ В MLFLOW ---

    # Логуємо гіперпараметри
    mlflow.log_params(params)

    # Логуємо метрики
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_r2", train_r2)

    mlflow.log_metric("test_rmse", test_rmse)
    mlflow.log_metric("test_mae", test_mae)
    mlflow.log_metric("test_r2", test_r2)

    # Логуємо саму модель
    mlflow.sklearn.log_model(dt_model, "decision_tree_model")

    # Створення та логування графіка Feature Importance
    plt.figure(figsize=(10, 8))
    feat_importances = pd.Series(dt_model.feature_importances_, index=X_train.columns)
    feat_importances.nlargest(10).plot(kind="barh", color="teal").invert_yaxis()
    plt.title("Top 10 Feature Importances - Decision Tree")
    plt.xlabel("Importance Score")

    # Зберігаємо графік
    plot_filename = "feature_importance.png"
    plt.savefig(plot_filename, bbox_inches="tight")
    mlflow.log_artifact(plot_filename)
    plt.close()

print(f"Експеримент залоговано в MLflow під назвою '{experiment_name}'.")
