# 🎓 Academic Performance Intelligence Platform 

An end-to-end Machine Learning and Generative AI system designed to
predict and analyze academic performance using structured student data.

------------------------------------------------------------------------

## 🚀 Overview

The Academic Performance Intelligence Platform is a production-ready
analytics system that:

-   Predicts student academic outcomes using a trained Random Forest
    model
-   Applies consistent preprocessing and feature scaling
-   Generates AI-driven performance analysis using Groq LLM
-   Provides an interactive dashboard via Streamlit

This project demonstrates real-world ML engineering practices
including: - Feature alignment - Scaler consistency - Model deployment -
LLM integration - Caching and performance optimization

------------------------------------------------------------------------

## 🏗 System Architecture

User Input (Streamlit UI)\
↓\
Preprocessing (Feature Encoding + Scaling)\
↓\
Random Forest Model\
↓\
Prediction + Confidence Score\
↓\
Groq LLM (Generative AI Explanation)\
↓\
Interactive Dashboard Output

------------------------------------------------------------------------

## 🧠 Machine Learning Details

-   Model: Random Forest Classifier\
-   Preprocessing: Manual encoding + StandardScaler\
-   Feature alignment using `scaler.feature_names_in_`\
-   Real-time inference

------------------------------------------------------------------------

## 🤖 Generative AI Integration

-   LLM Provider: Groq\
-   Model Used: `openai/gpt-oss-120b`\
-   Provides structured explanation and improvement strategies\
-   Enhances interpretability beyond raw prediction

------------------------------------------------------------------------

## 📊 Key Features

✔ Academic grade prediction\
✔ Confidence score visualization\
✔ AI-generated improvement insights\
✔ Production-ready Streamlit UI\
✔ Cached model loading for performance

------------------------------------------------------------------------

## 🛠 Tech Stack

-   Python\
-   Scikit-learn\
-   Pandas & NumPy\
-   Streamlit\
-   LangChain\
-   Groq API\
-   Joblib

------------------------------------------------------------------------

## ⚙️ Installation

``` bash
uv add streamlit scikit-learn pandas numpy joblib langchain langchain-groq python-dotenv
```

------------------------------------------------------------------------

## 🔑 Environment Setup

Create a `.env` file:

    GROQ_API_KEY=your_groq_api_key_here

------------------------------------------------------------------------

## ▶️ Run Application

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

## 📈 Future Improvements

-   SHAP Explainability
-   Docker containerization
-   Cloud deployment (AWS/GCP)
-   CI/CD automation
-   Full sklearn Pipeline integration

------------------------------------------------------------------------

## 👨‍💻 Author

Vishwatej Khot\
Machine Learning & AI Engineer

------------------------------------------------------------------------

## 📜 License

MIT License
