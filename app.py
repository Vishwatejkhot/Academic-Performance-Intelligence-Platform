import streamlit as st
import joblib
import numpy as np
import pandas as pd
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import os


load_dotenv()

st.set_page_config(
    page_title="Academic Performance Intelligence Platform",
    page_icon="🎓",
    layout="wide"
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("student_performance_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler

model, scaler = load_artifacts()


@st.cache_resource
def load_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY")
    )

llm = load_llm()


st.markdown("""
# 🎓 Academic Performance Intelligence Platform
### End-to-End Machine Learning & Generative AI Academic Analytics System
""")

st.markdown("---")


st.sidebar.header("📋 Student Profile Input")

age = st.sidebar.number_input("Age", 10, 25, 18)
gender = st.sidebar.selectbox("Gender", ["male", "female", "other"])
school_type = st.sidebar.selectbox("School Type", ["public", "private"])
parent_education = st.sidebar.selectbox("Parent Education", ["high_school", "bachelor", "master", "phd"])
study_hours = st.sidebar.number_input("Study Hours per Day", 0.0, 15.0, 2.0)
attendance = st.sidebar.slider("Attendance %", 0.0, 100.0, 75.0)
internet_access = st.sidebar.selectbox("Internet Access", ["no", "yes"])
travel_time = st.sidebar.selectbox("Travel Time", ["<15 min", "15-30 min", "30-60 min", ">60 min"])
extra_activities = st.sidebar.selectbox("Extra Activities", ["no", "yes"])
study_method = st.sidebar.selectbox("Study Method", ["self", "group", "online"])

predict_button = st.sidebar.button("🚀 Predict Performance")


gender_map = {'male':0,'female':1,'other':2}
school_map = {'public':0,'private':1}
travel_map = {
    "<15 min": 0,
    "15-30 min": 1,
    "30-60 min": 2,
    ">60 min": 3
}

parent_map = {"bachelor":0,"high_school":1,"master":2,"phd":3}
internet_map = {"no":0,"yes":1}
extra_map = {"no":0,"yes":1}
study_method_map = {"group":0,"online":1,"self":2}

input_dict = {
    "age": age,
    "gender": gender_map[gender],
    "school_type": school_map[school_type],
    "parent_education": parent_map[parent_education],
    "study_hours": study_hours,
    "attendance": attendance,
    "internet_access": internet_map[internet_access],
    "travel_time": travel_map[travel_time],
    "extra_activities": extra_map[extra_activities],
    "study_method": study_method_map[study_method]
}

input_df = pd.DataFrame([input_dict])

expected_cols = scaler.feature_names_in_

for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_cols]
scaled_input = scaler.transform(input_df)


if predict_button:

    with st.spinner("Running ML model..."):
        prediction = model.predict(scaled_input)[0]

    st.markdown("## 📊 Prediction Results")

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Grade", prediction)
    col2.metric("Study Hours", f"{study_hours} hrs/day")
    col3.metric("Attendance", f"{attendance}%")

    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(scaled_input)[0]
        confidence = np.max(prob)

        st.markdown("### 🎯 Model Confidence")
        st.progress(float(confidence))
        st.write(f"Confidence Score: {confidence*100:.2f}%")

    st.markdown("---")


    st.markdown("## 🤖 AI Performance Analysis")

    prompt = f"""
    Student Profile:
    Age: {age}
    Study Hours: {study_hours}
    Attendance: {attendance}%
    Parent Education: {parent_education}
    Study Method: {study_method}
    Predicted Grade: {prediction}

    Provide:
    1. Short performance explanation
    2. 5 improvement strategies
    3. Motivation tip
    Keep it concise and professional.
    """

    with st.spinner("Generating AI insights..."):
        response = llm.invoke(prompt)

    st.write(response.content)

    st.markdown("---")
    st.caption("Academic Performance Intelligence Platform | ML + Generative AI System")
