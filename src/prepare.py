import pandas as pd
import os
from sklearn.model_selection import train_test_split

# 1. Завантаження даних
print("Завантаження даних...")
df = pd.read_csv('data/raw/cleaned_house_rent_data.csv')

# 2. One-Hot Encoding для категоріальних змінних
print("Кодування категоріальних ознак...")
categorical_cols = df.select_dtypes(include=['object']).columns
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 3. Розділення на X (ознаки) та y (цільова змінна)
X = df.drop(columns=['Rent'])
y = df['Rent']

# 4. Розділення вибірки (80% трен, 20% тест)
print("Розділення на тренувальну та тестову вибірки...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Об'єднуємо X та y для збереження у файли
train_df = pd.concat([X_train, y_train], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

# 5. Збереження готових датасетів
processed_path = 'data/processed/' 
os.makedirs(processed_path, exist_ok=True) 

train_file = 'train.csv'
test_file = 'test.csv'

train_df.to_csv(os.path.join(processed_path, train_file), index=False)
test_df.to_csv(os.path.join(processed_path, test_file), index=False)

print(f"Тренувальний датасет збережено у: {os.path.join(processed_path, train_file)}")
print(f"Тестовий датасет збережено у: {os.path.join(processed_path, test_file)}")