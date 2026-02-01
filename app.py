import streamlit as st
import pandas as pd
import shap
import xgboost as xgb
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric="logloss"
)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "xgb_model.pkl")

# Predict
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Streamlit UI
st.set_page_config(page_title="SHAP Dashboard")
st.title("Model Explainability Dashboard with SHAP")

st.write(f"Model Accuracy: {accuracy:.2f}")


# Feature Importance
st.subheader("Global Feature Importance")

fig1, ax1 = plt.subplots()
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
st.pyplot(fig1)

# Instance Explanation
st.subheader("Individual Prediction Explanation")

index = st.slider("Select Test Instance", 0, len(X_test)-1, 0)

st.write("Selected Input:")
st.dataframe(X_test.iloc[[index]])

prediction = model.predict(X_test.iloc[[index]])[0]
result = "Malignant" if prediction == 0 else "Benign"
st.write("Prediction:", result)

# Waterfall Plot
fig2, ax2 = plt.subplots()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[index],
        base_values=explainer.expected_value,
        data=X_test.iloc[index],
        feature_names=X_test.columns
    ),
    show=False
)

st.pyplot(fig2)
