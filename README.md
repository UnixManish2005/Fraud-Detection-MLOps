# End-to-End MLOps Pipeline for Fraud Detection 

#📌 Overview

This project is an end-to-end Machine Learning Fraud Detection System that detects fraudulent financial transactions using advanced ML models.

It covers the complete pipeline from:

Data preprocessing

Model training & evaluation

Model selection

Deployment using API & Web UI

🎯 Features

✅ Detect fraudulent transactions in real-time

✅ Handles imbalanced dataset using SMOTE

✅ Compares multiple ML models

✅ Selects best model based on ROC-AUC

✅ Explainable AI using SHAP

✅ Interactive UI built with Streamlit

✅ API support using FastAPI

🧠 Machine Learning Pipeline

Data Preprocessing

Feature Engineering (Time → cyclic features)

Handling imbalance (SMOTE)

Model Training:

Logistic Regression

Random Forest

XGBoost

Model Evaluation (ROC-AUC)

Best Model Selection

📊 Model Performance
Model	ROC-AUC
Random Forest	0.9818 ✅
XGBoost	0.9776
Logistic Regression	0.9718
🛠️ Tech Stack

Languages: Python

Libraries: Pandas, NumPy, Scikit-learn, XGBoost, SHAP

Backend: FastAPI

Frontend: Streamlit

Deployment: Render

Tools: Docker, Git

📁 Project Structure
fraud_detection/
│
├── app/                 # Streamlit UI + FastAPI app
├── src/                 # Core ML pipeline
├── data/                # Dataset
├── models/              # Trained models
├── plots/               # Visualizations
├── notebooks/           # Experiments
│
├── train.py             # Training pipeline
├── requirements.txt     # Dependencies
├── Dockerfile           # Container setup
├── .render.yaml         # Deployment config
└── README.md
⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/UnixManish2005/Fraud-Detection-MLOps.git
cd Fraud-Detection-MLOps/fraud_detection
2️⃣ Create Environment (Recommended)
conda create -n fraud_env python=3.10 -y
conda activate fraud_env
3️⃣ Install Dependencies
pip install -r requirements.txt
🚀 Run the Project
🔹 Train Model
python train.py
🔹 Run FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000
🔹 Run Streamlit UI
streamlit run app/streamlit_app.py
🌐 Deployment

This project is deployed on Render.

Web UI: (Add your Streamlit link here)

API: (Add your FastAPI link here)


💡 Key Learnings

Handling imbalanced datasets effectively

Building end-to-end ML pipelines

Model evaluation & selection

Deploying ML models in production

Creating interactive ML dashboards

🤝 Contributing

Feel free to fork this repo and contribute!

📬 Contact

Manishankar Dey
📧 manishankardey2005@gmail.com
🔗 https://www.linkedin.com/in/manish2005/

⭐ If you like this project, give it a star!