
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from xgboost import XGBClassifier

# 1. Load training and testing data
X_train = pd.read_csv("processed/X_train.csv")
X_test = pd.read_csv("processed/X_test.csv")

y_train = pd.read_csv("processed/y_train.csv").squeeze()
y_test = pd.read_csv("processed/y_test.csv").squeeze()

# 2. Encode target labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Encoded classes:", dict(
    zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
))

# 3. Identify numerical and categorical columns
numeric_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X_train.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical features:", len(numeric_features))
print("Categorical features:", len(categorical_features))

# 4. Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

# 5. Define three ML algorithms
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    ),

    "SVM": SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced"
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="logloss",
        random_state=42
    )
}

# 6. Create folders
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

results = []

# 7. Train and evaluate each model
for name, model in models.items():

    print("\n" + "=" * 45)
    print("Training:", name)

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", model)
    ])

    # Train model
    pipeline.fit(X_train, y_train_encoded)

    # Make predictions
    predictions = pipeline.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test_encoded, predictions)

    # Class 0 = At Risk
    precision = precision_score(
        y_test_encoded,
        predictions,
        pos_label=0,
        zero_division=0
    )

    recall = recall_score(
        y_test_encoded,
        predictions,
        pos_label=0,
        zero_division=0
    )

    f1 = f1_score(
        y_test_encoded,
        predictions,
        pos_label=0,
        zero_division=0
    )

    print("Accuracy:", round(accuracy * 100, 2), "%")
    print("At Risk Precision:", round(precision, 3))
    print("At Risk Recall:", round(recall, 3))
    print("At Risk F1 Score:", round(f1, 3))

    print("\nClassification Report:")
    print(classification_report(
        y_test_encoded,
        predictions,
        labels=[0, 1],
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_test_encoded,
        predictions,
        labels=[0, 1]
    ))

    # Save trained model
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(pipeline, "models/" + filename)

    results.append({
        "Algorithm": name,
        "Accuracy": round(accuracy * 100, 2),
        "At Risk Precision": round(precision, 3),
        "At Risk Recall": round(recall, 3),
        "At Risk F1 Score": round(f1, 3)
    })

# 8. Save label encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")

# 9. Compare model performance
results_df = pd.DataFrame(results)
results_df.to_csv("results/model_comparison.csv", index=False)

print("\n========== FINAL MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nAll three models trained successfully!")
print("Models saved inside models folder.")
print("Comparison saved inside results folder.")
