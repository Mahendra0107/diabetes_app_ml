# 🧬 DiabetesAI — Predictive Analytics Platform

A futuristic, production-grade ML dashboard built with **Streamlit** for diabetes classification using the CDC BRFSS dataset.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?style=flat-square)

---

## ✨ Features

- **Multi-Model Training** — Logistic Regression, Decision Tree, Random Forest
- **Interactive Controls** — Adjust test split, random seed, model selection from sidebar
- **Dataset Overview** — Distribution charts, correlation heatmap, raw data preview
- **Performance Analysis** — Accuracy trends, confusion matrices, comparison table
- **Real-Time Prediction** — Enter patient parameters and get live predictions with probabilities
- **Futuristic UI** — Orbitron font, neon cyan/purple palette, grid-glass aesthetic

---

## 🚀 Deploy on Streamlit Cloud

1. Fork / push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo
4. Set **Main file path** to `app.py`
5. Click **Deploy** — done! 🎉

---

## 💻 Run Locally

```bash
git clone https://github.com/YOUR_USERNAME/diabetes-ml-app
cd diabetes-ml-app
pip install -r requirements.txt
streamlit run app.py
```

---

## 📁 Project Structure

```
diabetes-ml-app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md           # This file
└── diabetes.csv        # (Optional) Place your dataset here
```

> **Note:** If you have the original `diabetes.csv` (253,680 records from CDC BRFSS), place it in the project root. The app will auto-detect it. Otherwise it uses a generated sample dataset for demo purposes.

---

## 📊 Dataset

- **Source:** CDC Diabetes Health Indicators (BRFSS 2015)
- **Rows:** 253,680
- **Features:** 21 health indicators
- **Target:** `Diabetes_012` — 0 (No diabetes), 1 (Pre-diabetic), 2 (Diabetic)

---

## 🧠 Models & Results

| Model | Accuracy (test=0.1) |
|---|---|
| Logistic Regression | 84.38% |
| Decision Tree | 77.01% |
| Random Forest | 84.32% |

---

## 🎨 Tech Stack

`Streamlit` · `Scikit-learn` · `Pandas` · `NumPy` · `Matplotlib`

---

Made with 🧬 by [A23126512014]
