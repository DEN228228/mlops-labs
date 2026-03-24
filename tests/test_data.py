import os
import pandas as pd


def test_data_schema_basic():

    data_path = os.getenv(
        "DATA_PATH", "data/processed/cleaned_house_rent_data_cropped.csv"
    )

    assert os.path.exists(data_path), f"Дані не знайдено за шляхом: {data_path}"

    df = pd.read_csv(data_path)

    required_cols = {
        "Year Built",
        "BHK",
        "Size",
        "Bathroom",
        "Floor_Number",
        "Total_Floors",
        "Building Type_Condo",
        "Building Type_House",
        "Building Type_Land",
        "Building Type_Office",
        "Building Type_Single Family",
        "Area Type_Carpet Area",
        "Area Type_Plot Area",
        "Area Type_Super Area",
        "City_Ahmedabad",
        "City_Amritsar",
        "City_Aurangabad",
        "City_Bangalore",
        "City_Bhopal",
        "City_Chandigarh",
        "City_Chennai",
        "City_Coimbatore",
        "City_Delhi",
        "City_Faridabad",
        "City_Ghaziabad",
        "City_Gwalior",
        "City_Hyderabad",
        "City_Indore",
        "City_Jaipur",
        "City_Jodhpur",
        "City_Kalyan-Dombivli",
        "City_Kanpur",
        "City_Kolkata",
        "City_Lucknow",
        "City_Ludhiana",
        "City_Mangalore",
        "City_Meerut",
        "City_Mumbai",
        "City_Mysore",
        "City_Nagpur",
        "City_Nashik",
        "City_Navi Mumbai",
        "City_Noida",
        "City_Patna",
        "City_Pune",
        "City_Rajkot",
        "City_Ranchi",
        "City_Surat",
        "City_Thane",
        "City_Vadodara",
        "City_Varanasi",
        "City_Vijayawada",
        "City_Visakhapatnam",
        "Furnishing Status_Semi-Furnished",
        "Furnishing Status_Unfurnished",
        "Tenant Preferred_Bachelors/Family",
        "Tenant Preferred_Family",
        "Point of Contact_Contact Owner",
        "Rent",
    }

    missing = required_cols - set(df.columns)
    assert not missing, f"Відсутні обов'язкові колонки: {sorted(missing)}"

    assert (
        df["Rent"].notna().all()
    ), "Цільова колонка 'Rent' містить пропущені значення (NaN)"

    assert (df["Rent"] > 0).all(), "Знайдено рядки, де ціна 'Rent' <= 0"
