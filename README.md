# karunarunjula7-Model-Explainability-Dashboard-with-SHAP
# Model Explainability Dashboard with SHAP

An interactive Streamlit dashboard that explains predictions of a trained XGBoost machine learning model using SHAP (SHapley Additive exPlanations). This project helps users understand why a model makes a particular prediction by visualizing feature contributions at both global and individual levels.

##  Project Overview

Machine learning models often act as black boxes. This project focuses on 'model interpretability', enabling users to:

* See which features are most important globally
* Inspect how each feature contributes to a single prediction
* Interactively explore predictions through a clean dashboard

The Breast Cancer dataset from Scikit-learn is used as a sample dataset.

##  Technologies Used

* Python
* XGBoost
* SHAP
* Streamlit
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

##  Project Structure

Project/
│
├── app.py                 # Streamlit application
├── xgb_model.pkl          # Saved trained model
├── requirements.txt       # Required libraries
├── README.md              # Project documentation
│
└── screenshots/
     ├── 01_global_feature_importance.png
     ├── 02_input_and_prediction.png
     └── 03_waterfall_plot.png

##  Features of the Dashboard

### 1. Global Feature Importance

Displays SHAP bar chart showing which features most influence the model across all predictions.

📁 Screenshot:

01_global_feature_importance.png

### 2. Individual Prediction (Input & Output)

Shows:

* Selected test instance
* Feature values
* Model prediction (Malignant / Benign)

Screenshot:

02_input_and_prediction.png

### 3. Individual Prediction Explanation

SHAP waterfall plot explaining how each feature pushes the prediction towards Malignant or Benign.

Screenshot:

03_waterfall_plot.png

##  How to Run the Project

### Step 1: Install Dependencies

pip install streamlit pandas numpy shap xgboost joblib scikit-learn matplotlib seaborn

### Step 2: Run Application

python -m streamlit run app.py

The dashboard will open at:

http://localhost:8501

##  Model Details

* Algorithm: XGBoost Classifier
* Dataset: Breast Cancer Dataset (sklearn)
* Evaluation Metric: Accuracy & ROC-AUC
* Model saved as: `xgb_model.pkl`

##  Key Learnings

* Built end-to-end ML explainability pipeline
* Applied SHAP for global and local explanations
* Designed interactive ML dashboard using Streamlit
* Improved understanding of model interpretability

##  Deliverables

* Streamlit dashboard code
* Trained model file
* Screenshot evidence
* Documentation

##  Conclusion

This project demonstrates practical experience in building explainable AI systems and deploying them through an interactive interface. It showcases not only machine learning skills but also interpretability, visualization, and deployment abilities.

Developed by Karuna Kumari Runjula
