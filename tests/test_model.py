import os
import json


def test_artifacts_exist():

    model_type = os.getenv("MODEL_TYPE", "DecisionTreeRegressor")
    model_path = f"models/best_model_{model_type}.pkl"

    assert os.path.exists(model_path), f"Файл моделі не знайдено: {model_path}"
    assert os.path.exists("metrics.json"), "Файл metrics.json не знайдено!"


def test_quality_gate_r2():

    assert os.path.exists(
        "metrics.json"
    ), "Немає файлу metrics.json для перевірки Quality Gate"
    threshold = float(os.getenv("R2_THRESHOLD", "0.70"))

    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
    assert "r2" in metrics, "У файлі metrics.json відсутній ключ 'r2'"

    r2_score = float(metrics["r2"])
    assert (
        r2_score >= threshold
    ), f"Quality Gate не пройдено: R2 ({r2_score:.4f}) < {threshold:.2f}"
