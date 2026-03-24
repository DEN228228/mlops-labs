import os
import pandas as pd


def test_data_schema_basic():

    data_path = os.getenv("DATA_PATH", "data/raw/cleaned_house_rent_data_cropped.csv")

    assert os.path.exists(data_path), f"Дані не знайдено за шляхом: {data_path}"

    df = pd.read_csv(data_path)

    required_cols = {
        "Year Built",
        "BHK",
    }

    missing = required_cols - set(df.columns)
    assert not missing, f"Відсутні обов'язкові колонки: {sorted(missing)}"

    assert (
        df["Rent"].notna().all()
    ), "Цільова колонка 'Rent' містить пропущені значення (NaN)"

    assert (df["Rent"] > 0).all(), "Знайдено рядки, де ціна 'Rent' <= 0"
