import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="Groundwater Prediction", page_icon="🌊", layout="wide")

# Title
st.title("🌊 Groundwater Prediction using Machine Learning")
st.markdown("Predict groundwater levels based on environmental factors")

# Sidebar for model selection
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select ML Model", ["Random Forest", "Linear Regression"])

# Dataset loading
st.header("📁 Dataset")
col1, col2 = st.columns(2)

with col1:
    use_sample = st.checkbox("Use sample dataset", value=True)
    
with col2:
    uploaded_file = st.file_uploader("Or upload your CSV", type=["csv"])

# Load data
df = None
if use_sample and os.path.exists("groundwater.csv"):
    df = pd.read_csv("groundwater.csv")
    st.success("Sample dataset loaded!")
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded!")

if df is not None:
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Drop missing values
    df = df.dropna()

    # Split features & target
    if "GroundwaterLevel" not in df.columns:
        st.error("Dataset must have a column named 'GroundwaterLevel'")
    else:
        X = df.drop("GroundwaterLevel", axis=1)
        y = df["GroundwaterLevel"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model based on selection
        if model_choice == "Random Forest":
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            model = LinearRegression()
            
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        st.subheader("📊 Model Performance")
        st.write(f"RMSE: **{rmse:.2f}**")
        st.write(f"R² Score: **{r2:.2f}**")

        # Plot Actual vs Predicted
        st.subheader("📈 Actual vs Predicted")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred, alpha=0.7, edgecolors="k")
        ax.set_xlabel("Actual Groundwater Level")
        ax.set_ylabel("Predicted Groundwater Level")
        ax.set_title("Actual vs Predicted")
        st.pyplot(fig)

        # Feature Importance (only for Random Forest)
        if model_choice == "Random Forest":
            st.subheader("🔥 Feature Importance")
            importances = model.feature_importances_
            feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

            fig, ax = plt.subplots()
            sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax)
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        # Prediction for new input
        st.subheader("🔮 Predict Groundwater Level")
        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(f"Enter {col}", value=float(df[col].mean()))

        if st.button("🔮 Predict"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Groundwater Level: **{prediction:.2f} meters**")
            
            # Show confidence based on model type
            if model_choice == "Random Forest":
                st.info(f"Model: {model_choice} | R² Score: {r2:.2f}")
else:
    st.info("Please load a dataset to start the prediction model.")
    st.markdown("""
    ### Expected CSV Format:
    Your CSV should contain these columns:
    - **Rainfall**: Amount of rainfall (mm)
    - **Temperature**: Temperature (°C) 
    - **SoilMoisture**: Soil moisture percentage
    - **GroundwaterLevel**: Target variable (meters)
    """)
