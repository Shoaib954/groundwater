# 🌊 Groundwater Prediction Web Application

A machine learning web application built with Streamlit to predict groundwater levels based on environmental factors.

## Features

- **Multiple ML Models**: Random Forest and Linear Regression
- **Interactive Interface**: Easy-to-use web interface
- **Data Visualization**: Performance metrics and feature importance plots
- **Real-time Predictions**: Input custom values for instant predictions
- **Sample Dataset**: Pre-loaded sample data for testing

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
streamlit run app.py
```

## Dataset Format

Your CSV file should contain these columns:
- **Rainfall**: Amount of rainfall (mm)
- **Temperature**: Temperature (°C)
- **SoilMoisture**: Soil moisture percentage
- **GroundwaterLevel**: Target variable (meters)

## Usage

1. Choose to use the sample dataset or upload your own CSV
2. Select a machine learning model from the sidebar
3. View model performance metrics and visualizations
4. Input values to predict groundwater levels

## Models

- **Random Forest**: Ensemble method with feature importance analysis
- **Linear Regression**: Simple linear relationship modeling