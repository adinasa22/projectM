# shap_app.py

import streamlit as st
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Suppress warnings and set global shap settings
import warnings
warnings.filterwarnings('ignore')
shap.initjs()

st.set_page_config(layout="wide")
st.title("🔍 SHAP Value Explorer")
st.write("Upload a dataset and see how features influence model predictions using SHAP.")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    target_column = st.selectbox("Select Target Column", df.columns)

    if st.button("Run SHAP Analysis"):
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Encode categorical columns
        for col in X.select_dtypes(include=['object', 'category']).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        if y.dtype == 'object' or y.dtype.name == 'category':
            y = LabelEncoder().fit_transform(y)

        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        st.success(f"Model Accuracy: {acc:.2f}")

        # SHAP Explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X_test)

        # SHAP Summary Plot
        st.subheader("Feature Importance (SHAP Summary Plot)")
        fig_summary, ax = plt.subplots()
        shap.plots.beeswarm(shap_values, max_display=10, show=False)
        st.pyplot(fig_summary)

        # SHAP Bar Plot
        st.subheader("Mean Absolute SHAP Values")
        fig_bar, ax = plt.subplots()
        shap.plots.bar(shap_values, max_display=10, show=False)
        st.pyplot(fig_bar)

        # SHAP Force Plot for a specific instance
        st.subheader("Force Plot for One Sample")
        idx = st.slider("Select test sample index", 0, X_test.shape[0]-1, 0)
        st_shap = st.empty()
        st_shap.plotly_chart(shap.plots.force(shap_values[idx], matplotlib=False), use_container_width=True)

