
import pandas as pd

# Load student dataset
df = pd.read_csv("dataset/student-mat.csv", sep=";")

# Display first 5 records
print("\n--- FIRST 5 STUDENTS ---")
print(df.head())

# Display dataset size
print("\n--- DATASET SHAPE ---")
print(df.shape)

# Display column names
print("\n--- COLUMN NAMES ---")
print(df.columns.tolist())

# Display dataset information
print("\n--- DATASET INFORMATION ---")
df.info()

# Check missing values
print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Statistical summary
print("\n--- STATISTICAL SUMMARY ---")
print(df.describe())

