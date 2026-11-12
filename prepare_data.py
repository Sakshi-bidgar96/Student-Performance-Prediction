
import pandas as pd
import os

from sklearn.model_selection import train_test_split

# Step 1: Load dataset
df = pd.read_csv("dataset/student-mat.csv", sep=";")

# Step 2: Create target variable
df["risk"] = df["G3"].apply(
    lambda grade: "At Risk" if grade < 10 else "Not At Risk"
)

# Step 3: Remove unwanted columns
X = df.drop(columns=["G1", "G2", "G3", "risk"])

# Step 4: Target variable
y = df["risk"]

# Step 5: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Step 6: Create processed folder
os.makedirs("processed", exist_ok=True)

# Step 7: Save datasets
X_train.to_csv("processed/X_train.csv", index=False)
X_test.to_csv("processed/X_test.csv", index=False)

y_train.to_csv("processed/y_train.csv", index=False)
y_test.to_csv("processed/y_test.csv", index=False)

# Step 8: Display results
print("\n--- DATA PREPROCESSING RESULTS ---")

print("Original dataset:", df.shape)
print("Input features:", X.shape[1])

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)

print("\nTraining target distribution:")
print(y_train.value_counts())

print("\nTesting target distribution:")
print(y_test.value_counts())

print("\nInput feature columns:")
print(X.columns.tolist())

print("\nData preprocessing completed successfully!")
