
import streamlit as st
import pandas as pd
import joblib
import os

# Page configuration
st.set_page_config(
    page_title="Student Risk Prediction",
    page_icon="🎓",
    layout="wide"
)

# Load saved models
@st.cache_resource
def load_models():
    rf = joblib.load("models/random_forest.pkl")
    svm = joblib.load("models/svm.pkl")
    xgb = joblib.load("models/xgboost.pkl")
    encoder = joblib.load("models/label_encoder.pkl")
    return rf, svm, xgb, encoder

rf_model, svm_model, xgb_model, label_encoder = load_models()

# Sidebar
st.sidebar.title("🎓 Student Analytics")
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Student Prediction", "Model Comparison", "About Project"]
)

# Home page
if page == "Home":

    st.title("🎓 Student Performance & Risk Prediction")
    st.write(
        "A Machine Learning based system to identify "
        "students who may be at academic risk."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Students", "395")
    col2.metric("ML Algorithms", "3")
    col3.metric("Best Accuracy", "73.42%")

    st.subheader("Project Overview")

    st.write("""
    This project uses student academic, family and
    lifestyle information to predict whether a student
    belongs to the At Risk or Not At Risk category.

    Algorithms used:
    1. Random Forest
    2. Support Vector Machine (SVM)
    3. XGBoost
    """)

    st.info(
        "This is an academic demonstration project, "
        "not a definitive assessment of a student's future."
    )

# Student prediction page
elif page == "Student Prediction":

    st.title("🧑‍🎓 Student Risk Prediction")
    st.write("Enter student information below.")

    with st.form("student_form"):

        st.subheader("Basic Information")

        c1, c2, c3 = st.columns(3)

        school = c1.selectbox("School", ["GP", "MS"])
        sex = c2.selectbox("Gender", ["F", "M"])
        age = c3.number_input("Age", 15, 22, 17)

        address = c1.selectbox("Address", ["U", "R"])
        famsize = c2.selectbox("Family Size", ["GT3", "LE3"])
        Pstatus = c3.selectbox("Parent Status", ["T", "A"])

        st.subheader("Family Information")

        c1, c2 = st.columns(2)

        Medu = c1.selectbox("Mother Education", [0, 1, 2, 3, 4])
        Fedu = c2.selectbox("Father Education", [0, 1, 2, 3, 4])

        Mjob = c1.selectbox(
            "Mother Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        Fjob = c2.selectbox(
            "Father Job",
            ["teacher", "health", "services", "at_home", "other"]
        )

        reason = c1.selectbox(
            "Reason for School",
            ["course", "home", "reputation", "other"]
        )

        guardian = c2.selectbox(
            "Guardian",
            ["mother", "father", "other"]
        )

        st.subheader("Academic Information")

        c1, c2, c3 = st.columns(3)

        traveltime = c1.selectbox("Travel Time", [1, 2, 3, 4])
        studytime = c2.selectbox("Study Time", [1, 2, 3, 4])
        failures = c3.selectbox("Past Failures", [0, 1, 2, 3])

        absences = st.number_input(
            "Number of Absences",
            min_value=0,
            max_value=100,
            value=2
        )

        st.subheader("Student Support")

        c1, c2 = st.columns(2)

        schoolsup = c1.selectbox("School Support", ["yes", "no"])
        famsup = c2.selectbox("Family Support", ["yes", "no"])

        paid = c1.selectbox("Extra Paid Classes", ["yes", "no"])
        activities = c2.selectbox("Extra Activities", ["yes", "no"])

        nursery = c1.selectbox("Attended Nursery", ["yes", "no"])
        higher = c2.selectbox("Wants Higher Education", ["yes", "no"])

        internet = c1.selectbox("Internet Access", ["yes", "no"])
        romantic = c2.selectbox("Romantic Relationship", ["yes", "no"])

        st.subheader("Lifestyle Information")

        c1, c2, c3 = st.columns(3)

        famrel = c1.selectbox("Family Relationship (1-5)", [1, 2, 3, 4, 5])
        freetime = c2.selectbox("Free Time (1-5)", [1, 2, 3, 4, 5])
        goout = c3.selectbox("Going Out (1-5)", [1, 2, 3, 4, 5])

        c1, c2, c3 = st.columns(3)

        Dalc = c1.selectbox("Workday Alcohol Scale (1-5)", [1, 2, 3, 4, 5])
        Walc = c2.selectbox("Weekend Alcohol Scale (1-5)", [1, 2, 3, 4, 5])
        health = c3.selectbox("Health (1-5)", [1, 2, 3, 4, 5])

        submitted = st.form_submit_button(
            "🔍 Predict Student Risk",
            use_container_width=True
        )

    if submitted:

        student_data = pd.DataFrame([{
            "school": school,
            "sex": sex,
            "age": age,
            "address": address,
            "famsize": famsize,
            "Pstatus": Pstatus,
            "Medu": Medu,
            "Fedu": Fedu,
            "Mjob": Mjob,
            "Fjob": Fjob,
            "reason": reason,
            "guardian": guardian,
            "traveltime": traveltime,
            "studytime": studytime,
            "failures": failures,
            "schoolsup": schoolsup,
            "famsup": famsup,
            "paid": paid,
            "activities": activities,
            "nursery": nursery,
            "higher": higher,
            "internet": internet,
            "romantic": romantic,
            "famrel": famrel,
            "freetime": freetime,
            "goout": goout,
            "Dalc": Dalc,
            "Walc": Walc,
            "health": health,
            "absences": absences
        }])

        prediction = rf_model.predict(student_data)[0]

        result = label_encoder.inverse_transform([prediction])[0]

        st.subheader("Prediction Result")

        if result == "At Risk":
            st.error("⚠️ Student Category: AT RISK")
            st.write(
                "Consider reviewing the student's academic "
                "and support needs."
            )
        else:
            st.success("✅ Student Category: NOT AT RISK")
            st.write(
                "The model classified this student as Not At Risk."
            )

        st.caption(
            "This is a model-based estimate, not a final judgment "
            "about the student."
        )

# Model comparison page
elif page == "Model Comparison":

    st.title("📊 Model Performance Comparison")

    if os.path.exists("results/model_comparison.csv"):

        results = pd.read_csv("results/model_comparison.csv")

        st.dataframe(results, use_container_width=True)

        st.subheader("Accuracy Comparison")

        chart_data = results.set_index("Algorithm")[["Accuracy"]]

        st.bar_chart(chart_data)

        st.subheader("Risk Detection Metrics")

        metric_columns = [
            "At Risk Precision",
            "At Risk Recall",
            "At Risk F1 Score"
        ]

        st.bar_chart(
            results.set_index("Algorithm")[metric_columns]
        )

    else:
        st.warning("Model comparison file not found.")

# About page
elif page == "About Project":

    st.title("About Our Project")

    st.write("""
    Project Name:
    Student Performance & Risk Prediction

    Dataset:
    UCI Student Performance Dataset

    Dataset Size:
    395 Students

    Machine Learning Algorithms:
    1. Random Forest
    2. Support Vector Machine
    3. XGBoost

    Technologies:
    Python, Pandas, NumPy, Scikit-learn,
    XGBoost, Streamlit, Matplotlib
    """)

    st.info(
        "Developed as an academic Machine Learning project."
    )
