# 🧠 Mental Health Score Prediction

A Machine Learning web application that predicts a student's **Mental Health Score (0–10)** based on academic habits, social media usage, lifestyle factors, and stress level.

The project uses **Python, Scikit-learn, Pandas, and Streamlit** to provide an interactive prediction interface.

> ⚠️ **Educational Disclaimer:** This project is developed for educational and demonstration purposes only. The prediction is not a medical diagnosis or a substitute for professional mental-health assessment.

---

## 🚀 Live Demo

🔗 **Streamlit App:**  

---

## 📌 Project Overview

The objective of this project is to build a supervised machine learning regression model capable of estimating a student's mental health score using different behavioral and lifestyle features.

The dataset contains information about **5,000 students** and includes factors such as:

- Age
- Gender
- Country
- Academic Level
- Most Used Social Media Platform
- Purpose of Social Media Usage
- Average Daily Usage Hours
- Daily Unlocks
- Study Hours
- Physical Activity Hours
- Sleep Hours per Night
- Stress Level

The target variable is:

**Mental_Health_Score**

---

## 🤖 Machine Learning Approach

This project follows a complete machine learning workflow:

1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. Data Preprocessing
6. Train-Test Split
7. Model Training
8. Model Evaluation
9. Model Saving
10. Streamlit Deployment

---

## 🧹 Data Preprocessing

The following preprocessing techniques were used:

### Numerical Features

- StandardScaler
- Log transformation for `Study_Hours`

### Ordinal Feature

`Stress_Level` was encoded according to its natural order:

```text
Low < Medium < High < Very High
