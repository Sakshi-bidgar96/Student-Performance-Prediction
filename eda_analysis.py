
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load dataset
df = pd.read_csv("dataset/student-mat.csv", sep=";")

# Create output folder
os.makedirs("screenshots", exist_ok=True)

# Create risk category
# G3 below 10 = At Risk
# G3 10 or above = Not At Risk
df["risk"] = df["G3"].apply(
    lambda grade: "At Risk" if grade < 10 else "Not At Risk"
)

# 1. Check duplicate records
print("Duplicate records:", df.duplicated().sum())

# 2. Risk category count
print("\nStudent Risk Distribution:")
print(df["risk"].value_counts())

# 3. Study time vs final grade
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x="studytime", y="G3")
plt.title("Study Time vs Final Grade")
plt.xlabel("Study Time Category")
plt.ylabel("Final Grade (G3)")
plt.savefig("screenshots/study_time_vs_grade.png")
plt.show()

# 4. Student risk distribution
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="risk")
plt.title("Student Risk Distribution")
plt.xlabel("Risk Category")
plt.ylabel("Number of Students")
plt.savefig("screenshots/student_risk.png")
plt.show()

# 5. Absences vs final grade
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x="absences", y="G3")
plt.title("Absences vs Final Grade")
plt.xlabel("Number of Absences")
plt.ylabel("Final Grade")
plt.savefig("screenshots/absences_vs_grade.png")
plt.show()

print("\nEDA analysis completed!")
