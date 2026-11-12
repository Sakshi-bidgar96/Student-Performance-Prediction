# Student Performance & Risk Prediction

## Project Overview
This project predicts whether a student is academically at risk or not at risk using Machine Learning algorithms. It analyzes student-related information and provides predictions through a simple Streamlit web application.

## Problem Statement
Identifying students who may be at academic risk at an early stage can help teachers provide timely support. This project uses machine learning to analyze student data and predict academic risk.

## Objectives
- Analyze student performance data.
- Perform data cleaning and preprocessing.
- Apply three Machine Learning algorithms.
- Compare model performance.
- Build an interactive web application for student risk prediction.

## Dataset
- **Source:** UCI Student Performance Dataset
- **File:** `student-mat.csv`
- **Records:** 395
- **Original Features:** 33
- **Target:** At Risk / Not At Risk

## Machine Learning Algorithms
1. Random Forest Classifier
2. Support Vector Machine (SVM)
3. XGBoost Classifier

## Model Results

| Algorithm | Accuracy |
|---|---:|
| Random Forest | 73.42% |
| SVM | 63.29% |
| XGBoost | 67.09% |

Random Forest achieved the highest accuracy among the three models on the test dataset.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- XGBoost
- Streamlit

## Project Structure
- `dataset/` – Student dataset
- `models/` – Saved trained models
- `app.py` – Streamlit web application
- `inspect_data.py` – Dataset inspection
- `eda_analysis.py` – Exploratory data analysis
- `prepare_data.py` – Data preprocessing
- `train_models.py` – Model training
- `requirements.txt` – Required Python libraries

## How to Run the Project

1. Clone or download this repository.
2. Open the project folder in VS Code.
3. Install the required libraries:

   `pip install -r requirements.txt`

4. Run the Streamlit application:

   `python -m streamlit run app.py`

5. Open the local URL displayed in the terminal.

## Conclusion
This project demonstrates how machine learning can be used to identify students who may be academically at risk. Three classification algorithms were trained and compared. Random Forest achieved the best accuracy of 73.42% on the test dataset.

**Note:** This is an academic demonstration project. Predictions are not definitive assessments of a student's ability or future performance.
