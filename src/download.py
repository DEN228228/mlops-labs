import os
import shutil
import kagglehub


def main():
    print("Завантаження датасету з Kaggle...")
    path = kagglehub.dataset_download("denis11334/house-data")
    print(f"Дані завантажено у кеш: {path}")

    os.makedirs("data/raw", exist_ok=True)

    files = [f for f in os.listdir(path) if f.endswith(".csv")]
    if not files:
        raise FileNotFoundError(
            "CSV файл не знайдено у завантаженому датасеті з Kaggle."
        )

    source_file = os.path.join(path, files[0])

    target_file = "data/raw/cleaned_house_rent_data_cropped.csv"

    print(f"Копіювання датасету у робочу папку: {target_file}")
    shutil.copy2(source_file, target_file)
    print("Етап завантаження успішно завершено!")


if __name__ == "__main__":
    main()
