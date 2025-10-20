import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Page config
st.set_page_config(page_title="Groundwater Prediction", page_icon="🌊", layout="wide")

# Custom CSS for attractive styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin-bottom: 2rem;
}
</style>
""", unsafe_allow_html=True)

# Attractive Header
st.markdown("""
<div class="main-header">
    <h1>🌊 Groundwater Prediction using Machine Learning</h1>
    <p>🔬 Advanced AI-powered prediction system for sustainable water management</p>
    <p>📊 Analyze • 🤖 Predict • 💧 Conserve</p>
</div>
""", unsafe_allow_html=True)

# Enhanced Sidebar
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; margin-bottom: 1rem;">
    <h2>🎛️ Control Panel</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("🤖 Model Configuration")
model_choice = st.sidebar.selectbox("Select ML Algorithm", ["🌳 Random Forest", "📈 Linear Regression"])
model_choice = model_choice.split(" ", 1)[1]  # Remove emoji for processing

# Enhanced Dataset Section
st.markdown("""
<div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
    <h2>📁 Dataset Management</h2>
    <p>Load your environmental data for groundwater analysis</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    use_sample = st.checkbox("Use sample dataset", value=True)
    
with col2:
    uploaded_file = st.file_uploader("Or upload your CSV", type=["csv"])

# Load data
df = None
if use_sample and os.path.exists("groundwater.csv"):
    df = pd.read_csv("groundwater.csv")
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1rem; border-radius: 8px; color: white; text-align: center;">
        <h4>✅ Dataset Loaded Successfully!</h4>
        <p>{len(df)} environmental records ready for analysis</p>
    </div>
    """, unsafe_allow_html=True)
elif uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Custom dataset uploaded!")

if df is not None:
    st.write("### Dataset Preview:")
    st.dataframe(df.head())

    # Check if dataset is empty
    if df.empty:
        st.error("❌ Dataset is empty! Please upload a valid CSV file.")
        st.stop()
    
    # Show basic dataset info
    st.info(f"📊 Dataset loaded with {len(df)} rows and {len(df.columns)} columns")
    
    # Check required columns
    required_columns = ['Rainfall', 'Temperature', 'SoilMoisture', 'GroundwaterLevel']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        st.error(f"❌ Missing columns: {missing_columns}")
        st.info("Your CSV must have: Rainfall, Temperature, SoilMoisture, GroundwaterLevel")
        st.write("**Your columns:**", list(df.columns))
        st.stop()
    
    # Select only the required numeric columns
    feature_columns = ['Rainfall', 'Temperature', 'SoilMoisture']
    
    # Convert to numeric and handle text values
    for col in feature_columns + ['GroundwaterLevel']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where any of our required columns have NaN (from text conversion)
    df_clean = df[feature_columns + ['GroundwaterLevel']].dropna()
    
    if df_clean.empty:
        st.error("❌ No numeric data found in required columns!")
        st.info("Make sure your columns contain only numbers, not text like 'WARANGAL (U)'")
        st.stop()
    
    # Check minimum data requirement
    if len(df_clean) < 5:
        st.error(f"❌ Need at least 5 rows of numeric data. You have {len(df_clean)} rows.")
        st.stop()
    
    # Split features & target
    X = df_clean[feature_columns]
    y = df_clean['GroundwaterLevel']
    
    st.success(f"✅ Using {len(df_clean)} rows with valid numeric data")
    
    # Adjust test_size for small datasets
    test_size = 0.2 if len(df_clean) >= 10 else 0.1
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Train model based on selection
    if model_choice == "Random Forest":
        model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42)
    else:
        model = LinearRegression()
        
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Enhanced Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy_percent = max(0, (1 - mae/np.mean(y_test)) * 100)

    st.markdown("### 📊 Model Performance Dashboard")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
            <h3>R² Score</h3>
            <h2>{r2:.3f}</h2>
            <p>Model Fit Quality</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
            <h3>RMSE</h3>
            <h2>{rmse:.2f}m</h2>
            <p>Prediction Error</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
            <h3>MAE</h3>
            <h2>{mae:.2f}m</h2>
            <p>Average Error</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
            <h3>Accuracy</h3>
            <h2>{accuracy_percent:.1f}%</h2>
            <p>Overall Performance</p>
        </div>
        """, unsafe_allow_html=True)

    # Enhanced Plot Actual vs Predicted
    st.markdown("### 📈 Model Validation Visualization")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Create gradient scatter plot
    scatter = ax.scatter(y_test, y_pred, alpha=0.7, c=y_pred, cmap='viridis', s=60, edgecolors='white', linewidth=0.5)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3, label='Perfect Prediction')
    
    ax.set_xlabel("Actual Groundwater Level (meters)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Predicted Groundwater Level (meters)", fontsize=12, fontweight='bold')
    ax.set_title(f"Model Performance: {model_choice}", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Predicted Values')
    
    # Style the plot
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#f8f9fa')
    
    st.pyplot(fig)

    # Feature Importance (only for Random Forest)
    if model_choice == "Random Forest":
        st.subheader("🔥 Feature Importance")
        importances = model.feature_importances_
        feat_importance = pd.Series(importances, index=X.columns).sort_values(ascending=False)

        fig, ax = plt.subplots()
        sns.barplot(x=feat_importance, y=feat_importance.index, ax=ax)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Importance Score")
        ax.set_ylabel("Features")
        st.pyplot(fig)

    # Enhanced Prediction Section
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center; margin: 2rem 0;">
        <h2>🔮 Make Your Prediction</h2>
        <p>Enter environmental parameters to predict groundwater levels</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    input_data = {}
    
    with col1:
        st.markdown("**🌧️ Rainfall (mm)**")
        input_data['Rainfall'] = st.number_input("Rainfall", value=float(df_clean['Rainfall'].mean()), label_visibility="collapsed")
        
    with col2:
        st.markdown("**🌡️ Temperature (°C)**")
        input_data['Temperature'] = st.number_input("Temperature", value=float(df_clean['Temperature'].mean()), label_visibility="collapsed")
        
    with col3:
        st.markdown("**💧 Soil Moisture (%)**")
        input_data['SoilMoisture'] = st.number_input("SoilMoisture", value=float(df_clean['SoilMoisture'].mean()), label_visibility="collapsed")

    # Attractive predict button
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        predict_clicked = st.button("🔮 Generate Prediction", use_container_width=True, type="primary")
        
    if predict_clicked:
        input_df = pd.DataFrame([input_data])
        prediction = model.predict(input_df)[0]
        prediction = max(5, min(25, prediction))  # Realistic bounds
        
        # Attractive prediction display
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; text-align: center; color: white; margin: 1rem 0;">
            <h2>🎯 Prediction Result</h2>
            <h1 style="font-size: 3em; margin: 0.5rem 0;">{prediction:.2f}m</h1>
            <p style="font-size: 1.2em;">Predicted Groundwater Level</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model info
        st.info(f"Model: {model_choice} | R² Score: {r2:.3f} | Accuracy: {accuracy_percent:.1f}%")
        
        # Interpretation
        if prediction > 18:
            st.write("🟢 **High groundwater level** - Good water availability")
        elif prediction > 12:
            st.write("🟡 **Moderate groundwater level** - Adequate water supply")
        else:
            st.write("🔴 **Low groundwater level** - Water conservation needed")
        
        # Visualization 1: Gauge Chart
        st.markdown("### 📊 Prediction Visualization")
        fig1, ax1 = plt.subplots(figsize=(10, 4))
        
        # Create horizontal bar chart showing prediction range
        categories = ['Low\n(5-12m)', 'Moderate\n(12-18m)', 'High\n(18-25m)']
        ranges = [7, 6, 7]
        colors = ['#ff6b6b', '#ffd93d', '#6bcf7f']
        
        bars = ax1.barh(categories, ranges, color=colors, alpha=0.3, edgecolor='black')
        
        # Add prediction marker
        if prediction <= 12:
            y_pos = 0
            x_pos = prediction - 5
        elif prediction <= 18:
            y_pos = 1
            x_pos = prediction - 12
        else:
            y_pos = 2
            x_pos = prediction - 18
        
        ax1.barh(categories[y_pos], x_pos, color=colors[y_pos], alpha=0.8, edgecolor='black', linewidth=2)
        ax1.axvline(x=x_pos, color='red', linestyle='--', linewidth=3, label=f'Your Prediction: {prediction:.2f}m')
        
        ax1.set_xlabel('Groundwater Level Range (meters)', fontsize=12, fontweight='bold')
        ax1.set_title(f'Prediction Level Indicator', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, 7)
        
        st.pyplot(fig1)
        
        # Visualization 2: Input Comparison
        st.markdown("### 📈 Input vs Dataset Average")
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        
        features = list(input_data.keys())
        input_values = list(input_data.values())
        avg_values = [df_clean[col].mean() for col in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, input_values, width, label='Your Input', color='#667eea', alpha=0.8)
        bars2 = ax2.bar(x + width/2, avg_values, width, label='Dataset Average', color='#f093fb', alpha=0.8)
        
        ax2.set_xlabel('Environmental Factors', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Values', fontsize=12, fontweight='bold')
        ax2.set_title('Input Comparison with Dataset', fontsize=14, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(features)
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        st.pyplot(fig2)
        
        # Visualization 3: Prediction Confidence
        st.markdown("### 🎯 Prediction Confidence")
        fig3, ax3 = plt.subplots(figsize=(8, 4))
        
        confidence_metrics = ['R² Score', 'Accuracy', 'Model Quality']
        confidence_values = [r2 * 100, accuracy_percent, (r2 * 100 + accuracy_percent) / 2]
        colors_conf = ['#667eea', '#43e97b', '#f093fb']
        
        bars = ax3.barh(confidence_metrics, confidence_values, color=colors_conf, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Percentage (%)', fontsize=12, fontweight='bold')
        ax3.set_title('Model Confidence Metrics', fontsize=14, fontweight='bold')
        ax3.set_xlim(0, 100)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, confidence_values)):
            ax3.text(val + 2, i, f'{val:.1f}%', va='center', fontweight='bold')
        
        st.pyplot(fig3)

else:
    # Attractive welcome screen
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 3rem; border-radius: 15px; color: white; text-align: center;">
        <h2>🚀 Welcome to Groundwater Prediction System</h2>
        <p style="font-size: 1.2em;">Load your dataset to start making predictions!</p>
    </div>
    
    <div style="background: #f8f9fa; padding: 2rem; border-radius: 10px; margin-top: 2rem;">
        <h3>📋 Expected CSV Format:</h3>
        <ul style="font-size: 1.1em;">
            <li><strong>Rainfall</strong>: Amount of rainfall (mm)</li>
            <li><strong>Temperature</strong>: Temperature (°C)</li>
            <li><strong>SoilMoisture</strong>: Soil moisture percentage</li>
            <li><strong>GroundwaterLevel</strong>: Target variable (meters)</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)