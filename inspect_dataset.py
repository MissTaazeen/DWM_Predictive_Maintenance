
import pandas as pd

# Load the AI4I dataset
file_path = "data/ai4i2020.csv"
df = pd.read_csv(file_path)

# 1. Dataset dimensions
print("\n--- DATASET SHAPE ---")
print(df.shape)

# 2. Column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# 3. Data types
print("\n--- DATA TYPES ---")
print(df.dtypes)

# 4. First five records
print("\n--- FIRST FIVE ROWS ---")
print(df.head())

# 5. Missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# 6. Binary target distribution
print("\n--- MACHINE FAILURE DISTRIBUTION ---")
print(df["Machine failure"].value_counts())

print("\n--- MACHINE FAILURE PERCENTAGES ---")
print(df["Machine failure"].value_counts(normalize=True) * 100)

# 7. Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())
