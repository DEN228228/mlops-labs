import os
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import optuna
import hydra
import joblib
from omegaconf import DictConfig, OmegaConf, ListConfig
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_validate


def build_model(model_type: str, params: dict, seed: int):
    if model_type == "DecisionTreeRegressor":
        return DecisionTreeRegressor(random_state=seed, **params)

    elif model_type == "RegularizedLinear":
        penalty_type = params.pop("penalty")

        if penalty_type == "l1":
            return Lasso(random_state=seed, **params)
        elif penalty_type == "l2":
            return Ridge(random_state=seed, **params)
        else:
            raise ValueError(f"Невідомий тип регуляризації: {penalty_type}")

    else:
        raise ValueError(f"Невідома модель: {model_type}")


def suggest_params(trial: optuna.Trial, cfg: DictConfig) -> dict:
    """Універсальний парсер простору пошуку (int, float, категоріальні)."""
    params = {}
    space = cfg.hpo.search_space.get(cfg.model.type, {})

    for param_name, config_vals in space.items():
        if cfg.hpo.sampler == "grid":
            params[param_name] = trial.suggest_categorical(
                param_name, list(config_vals)
            )
        else:
            if (
                isinstance(config_vals, (dict, DictConfig))
                and "low" in config_vals
                and "high" in config_vals
            ):
                if isinstance(config_vals.low, float) or isinstance(
                    config_vals.high, float
                ):
                    log_scale = config_vals.get("log", False)
                    params[param_name] = trial.suggest_float(
                        param_name,
                        float(config_vals.low),
                        float(config_vals.high),
                        log=log_scale,
                    )
                else:
                    params[param_name] = trial.suggest_int(
                        param_name, config_vals.low, config_vals.high
                    )
            elif isinstance(config_vals, (list, tuple, ListConfig)):
                params[param_name] = trial.suggest_categorical(
                    param_name, list(config_vals)
                )

    return params


def objective_factory(cfg: DictConfig, X_train: pd.DataFrame, y_train: pd.Series):

    def objective(trial: optuna.Trial) -> float:
        with mlflow.start_run(run_name=f"Trial_{trial.number}", nested=True):
            mlflow.set_tags(
                {
                    "author": "Denys Svintsylo",
                    "model_type": cfg.model.type,
                    "optuna_trial": trial.number,
                }
            )

            params = suggest_params(trial, cfg)
            params_to_log = params.copy()

            model = build_model(cfg.model.type, params, cfg.seed)

            kf = KFold(n_splits=cfg.hpo.cv_folds, shuffle=True, random_state=cfg.seed)
            scoring = {
                "neg_mse": "neg_mean_squared_error",
                "neg_mae": "neg_mean_absolute_error",
                "r2": "r2",
            }

            cv_results = cross_validate(model, X_train, y_train, cv=kf, scoring=scoring)

            cv_rmse = np.mean(np.sqrt(-cv_results["test_neg_mse"]))
            cv_mae = np.mean(-cv_results["test_neg_mae"])
            cv_r2 = np.mean(cv_results["test_r2"])

            model.fit(X_train, y_train)

            mlflow.log_params(params_to_log)
            mlflow.log_metric("cv_rmse", float(cv_rmse))
            mlflow.log_metric("cv_mae", float(cv_mae))
            mlflow.log_metric("cv_r2", float(cv_r2))

            mlflow.sklearn.log_model(model, "model")

            os.makedirs("plots", exist_ok=True)
            plt.figure(figsize=(10, 8))

            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                title = "Feature Importances"
            elif hasattr(model, "coef_"):
                importances = np.abs(model.coef_)
                title = "Feature Coefficients (Absolute Magnitude)"
            else:
                importances = None

            if importances is not None:
                feat_importances = pd.Series(importances, index=X_train.columns)
                feat_importances.nlargest(10).plot(
                    kind="barh", color="teal"
                ).invert_yaxis()
                plt.title(f"Top 10 {title} - Trial {trial.number}")
                plt.xlabel("Score")

                plot_filename = f"plots/feature_importance_trial_{trial.number}.png"
                plt.savefig(plot_filename, bbox_inches="tight")
                mlflow.log_artifact(plot_filename)
            plt.close()

            return float(cv_rmse)

    return objective


def main(cfg: DictConfig) -> None:
    train_path = cfg.data.train_path
    test_path = cfg.data.test_path

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if len(train_df) > 5000:
        train_df = train_df.sample(n=5000, random_state=cfg.seed)
    if len(test_df) > 1000:
        test_df = test_df.sample(n=1000, random_state=cfg.seed)

    X_train = train_df.drop(columns=["Rent"])
    y_train = train_df["Rent"]
    X_test = test_df.drop(columns=["Rent"])
    y_test = test_df["Rent"]

    mlflow.set_experiment(cfg.mlflow.experiment_name)

    print(
        f"Запуск MLflow та Optuna ({cfg.model.type}) на {cfg.hpo.get('n_trials', 'всіх')} спроб..."
    )
    hpo_type = cfg.hpo.sampler.upper()

    parent_run_name = f"Optuna_{cfg.model.type}_{hpo_type}"
    print(f"Створення батьківського Run'у з назвою: {parent_run_name}")

    with mlflow.start_run(run_name=parent_run_name):
        mlflow.log_dict(
            OmegaConf.to_container(cfg, resolve=True), "config_resolved.json"
        )

        sampler = None
        if cfg.hpo.sampler == "grid":
            model_space = cfg.hpo.search_space.get(cfg.model.type, {})
            search_space = {k: list(v) for k, v in model_space.items()}
            sampler = optuna.samplers.GridSampler(search_space)
        elif cfg.hpo.sampler == "random":
            sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=cfg.seed)

        study = optuna.create_study(
            direction=cfg.hpo.get("direction", "minimize"), sampler=sampler
        )
        objective = objective_factory(cfg, X_train, y_train)

        n_trials = cfg.hpo.get("n_trials", None) if cfg.hpo.sampler != "grid" else None
        study.optimize(objective, n_trials=n_trials)

        print(f"Найкраще значення CV RMSE: {study.best_value:.4f}")

        best_params = study.best_params
        best_params_to_log = best_params.copy()

        best_model = build_model(cfg.model.type, best_params, cfg.seed)
        best_model.fit(X_train, y_train)

        y_test_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        mlflow.log_params({"best_" + k: v for k, v in best_params_to_log.items()})
        mlflow.log_metric("final_test_rmse", float(test_rmse))
        mlflow.log_metric("final_test_mae", float(test_mae))
        mlflow.log_metric("final_test_r2", float(test_r2))

        os.makedirs("plots", exist_ok=True)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test, y_test_pred, alpha=0.5, color="dodgerblue")
        plt.plot(
            [y_test.min(), y_test.max()],
            [y_test.min(), y_test.max()],
            "r--",
            lw=2,
            label="Ideal Prediction",
        )
        plt.xlabel("Actual Rent")
        plt.ylabel("Predicted Rent")
        plt.title(f"Actual vs Predicted Rent (R2: {test_r2:.3f})")
        plt.legend()

        final_plot_path = "plots/best_model_actual_vs_predicted.png"
        plt.tight_layout()
        plt.savefig(final_plot_path)
        plt.close()
        mlflow.log_artifact(final_plot_path)

        print("Перевірка Quality Gate перед реєстрацією в MLflow...")
        r2_threshold = float(os.getenv("R2_THRESHOLD", "0.70"))

        if test_r2 >= r2_threshold:
            print(f"Успіх! R2 ({test_r2:.4f}) >= {r2_threshold}. Реєструємо модель...")
            mlflow.sklearn.log_model(
                sk_model=best_model,
                artifact_path="best_model",
                registered_model_name=f"{cfg.mlflow.registered_model_name}_{cfg.model.type}",
            )
        else:
            print(
                f"Модель відхилено. R2 ({test_r2:.4f}) < {r2_threshold}. Реєстрацію скасовано."
            )

        os.makedirs("models", exist_ok=True)
        model_filename = f"models/best_model_{cfg.model.type}.pkl"
        joblib.dump(best_model, model_filename)

        metrics = {
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
        }
        with open("metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        print(f"ФІНАЛЬНА ТОЧНІСТЬ МОДЕЛІ (R^2): {test_r2:.4f}")

    print(f"Експеримент залоговано в MLflow під назвою '{cfg.mlflow.experiment_name}'.")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def hydra_entry(cfg: DictConfig) -> None:
    main(cfg)


if __name__ == "__main__":
    hydra_entry()
