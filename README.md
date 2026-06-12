# 🎓 Student Performance Predictor — End-to-End ML Web App

An end-to-end Machine Learning project that predicts whether a student will pass or fail based on study habits and academic history. Includes a live interactive web app built with Streamlit.

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)](https://streamlit.io)

---

## 🚀 Live Demo
> **[Click here to try the live app →](https://myml-app.streamlit.app)**

---

## 📌 Project Overview

Early identification of at-risk students allows educators to intervene before it's too late. This project builds a complete ML pipeline — from raw data to a deployed prediction web app — using three classification algorithms and comparing their performance.

**Key highlight:** Not just a notebook — a fully deployed interactive web application.

---

## 📊 Dataset

| Detail | Info |
|---|---|
| **Source** | Synthetic dataset modelled on UCI Student Performance |
| **Size** | 1,000 student records |
| **Features** | 6 academic and lifestyle features |
| **Target** | Pass (1) / Fail (0) |

**Features:** Study Hours/Day, Attendance %, Previous Score, Sleep Hours, Extra Classes, Parental Education Level

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Language | Python 3.9+ |
| ML | Scikit-learn (Logistic Regression, Random Forest, Gradient Boosting) |
| Data | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Web App | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## 🤖 Model Results

| Model | Accuracy | ROC-AUC | CV Accuracy |
|---|---|---|---|
| Logistic Regression | 86.00% | 0.830 | 82.38% |
| Random Forest | 84.00% | 0.813 | 81.88% |
| **Gradient Boosting ✓** | **87.00%** | **0.830** | **80.25%** |

**Best Model: Gradient Boosting — 87% Accuracy, AUC 0.830**

**Top Features:** Study Hours > Previous Score > Attendance %

---

## 📁 Project Structure

```
student-performance-predictor/
│
├── app.py              # Streamlit web application
├── train_model.py      # ML training script
├── student_data.csv    # Dataset (auto-generated)
├── model.pkl           # Trained Gradient Boosting model
├── scaler.pkl          # StandardScaler
├── requirements.txt    # Dependencies
├── README.md
└── plots/
    ├── model_comparison.png
    ├── feature_importance.png
    └── eda.png
```

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/student-performance-predictor
cd student-performance-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train the model
python train_model.py

# 4. Launch the web app
streamlit run app.py
```

---

## 🔑 Key Learnings

- End-to-end ML pipeline from data generation to deployment
- Gradient Boosting outperforms simpler models on tabular data
- Study hours and previous scores are the strongest performance predictors
- Streamlit enables rapid deployment of ML models as interactive apps

---

## 👩‍💻 Author

  Harhita Rawat | B.Tech CSE | 3rd Year
*Amazon ML Summer School 2026 Portfolio Project*
