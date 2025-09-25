import pandas as pd
import numpy as np

np.random.seed(42)
n_samples = 1200

# Generate realistic environmental data
rainfall = np.random.normal(120, 40, n_samples)
rainfall = np.clip(rainfall, 20, 300)

temperature = np.random.normal(28, 6, n_samples)
temperature = np.clip(temperature, 15, 45)

soil_moisture = np.random.normal(42, 12, n_samples)
soil_moisture = np.clip(soil_moisture, 15, 70)

# Generate groundwater level with realistic relationships
groundwater_level = (
    0.08 * rainfall +
    -0.15 * temperature +
    0.12 * soil_moisture +
    np.random.normal(0, 2, n_samples) +
    8
)
groundwater_level = np.clip(groundwater_level, 5, 25)

# Create DataFrame
df = pd.DataFrame({
    'Rainfall': np.round(rainfall, 1),
    'Temperature': np.round(temperature, 1),
    'SoilMoisture': np.round(soil_moisture, 1),
    'GroundwaterLevel': np.round(groundwater_level, 1)
})

df.to_csv('groundwater.csv', index=False)
print(f"Generated {len(df)} rows of groundwater data")