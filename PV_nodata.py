import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==========================================
# CONFIGURATION
# ==========================================
# Adjust these values based on your specific solar panels
PANEL_AREA = 1.65  # in square meters (m^2)
RATED_POWER = 300  # in Watts (W)

def load_data(filepath=None):
    """
    Loads data from a CSV file or generates mock data if no file is provided.
    """
    if filepath:
        # EXPECTED CSV COLUMNS: 'Timestamp', 'Irradiance', 'Power_Output', 'T_Panel', 'T_Ambient', 'Wind_Speed', 'Humidity'
        df = pd.read_csv(filepath)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    else:
        print("No file provided. Generating mock data for demonstration...")
        # Generate 30 days of hourly data
        dates = pd.date_range(start='2024-06-01', periods=24*30, freq='H')
        n = len(dates)
        
        # Simulate daily solar curve (Irradiance)
        hour = dates.hour
        irradiance = np.maximum(0, 1000 * np.sin((hour - 6) * np.pi / 12)) # Simple sine wave for day
        irradiance += np.random.normal(0, 50, n) # Add noise
        irradiance = np.maximum(0, irradiance) # No negative light
        
        # Simulate Ambient Temp (correlated with sun but lagged)
        t_amb = 20 + 10 * np.sin((hour - 9) * np.pi / 12) + np.random.normal(0, 2, n)
        
        # Simulate Wind and Humidity (random walkish)
        wind = np.abs(np.random.normal(3, 2, n))
        humidity = np.clip(50 + np.random.normal(0, 10, n), 20, 100)
        
        # Simulate Panel Temp (T_amb + heating from sun - cooling from wind)
        t_panel = t_amb + (irradiance / 800) * 25 - (wind * 1.5)
        
        # Simulate Power Output (P = G * Area * Efficiency)
        # Base efficiency 18%, drops by 0.4% per degree C above 25
        base_eff = 0.18
        temp_coeff = -0.004
        efficiency_factor = base_eff * (1 + temp_coeff * (t_panel - 25))
        power = irradiance * PANEL_AREA * efficiency_factor
        
        df = pd.DataFrame({
            'Timestamp': dates,
            'Irradiance': irradiance,
            'Power_Output': power,
            'T_Panel': t_panel,
            'T_Ambient': t_amb,
            'Wind_Speed': wind,
            'Humidity': humidity
        })
        
    return df

def preprocess_data(df):
    """
    Cleans data and calculates efficiency.
    """
    # 1. Filter out night time or very low irradiance (Efficiency is undefined/unstable at G~0)
    # Threshold e.g., 50 W/m^2
    df_clean = df[df['Irradiance'] > 50].copy()
    
    # 2. Calculate Efficiency (%)
    # Efficiency = Power_Output / (Irradiance * Panel_Area)
    df_clean['Efficiency'] = (df_clean['Power_Output'] / (df_clean['Irradiance'] * PANEL_AREA)) * 100
    
    # 3. Calculate Temperature Difference (Delta T)
    if 'T_Ambient' in df_clean.columns:
        df_clean['Delta_T'] = df_clean['T_Panel'] - df_clean['T_Ambient']
        
    return df_clean

def analyze_correlations(df):
    """
    Plots correlation heatmap to see relationships.
    """
    plt.figure(figsize=(10, 8))
    cols = ['Efficiency', 'Irradiance', 'T_Panel', 'T_Ambient', 'Wind_Speed', 'Humidity']
    # Filter cols that exist in df
    cols = [c for c in cols if c in df.columns]
    
    sns.heatmap(df[cols].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

def plot_relationships(df):
    """
    Scatter plots for key variables vs Efficiency.
    """
    features = ['T_Panel', 'Wind_Speed', 'Humidity', 'Irradiance']
    features = [f for f in features if f in df.columns]
    
    plt.figure(figsize=(15, 10))
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i+1)
        sns.scatterplot(data=df, x=feature, y='Efficiency', alpha=0.5)
        
        # Add a trend line
        z = np.polyfit(df[feature], df['Efficiency'], 1)
        p = np.poly1d(z)
        plt.plot(df[feature], p(df[feature]), "r--")
        
        plt.title(f'Efficiency vs {feature}')
        plt.xlabel(feature)
        plt.ylabel('Efficiency (%)')
        
    plt.tight_layout()
    plt.show()

def train_predictive_model(df):
    """
    Trains a Random Forest model to predict Efficiency based on weather conditions.
    """
    print("\n--- Training Machine Learning Model ---")
    
    # Features to use for prediction
    feature_cols = ['Irradiance', 'T_Panel', 'Wind_Speed', 'Humidity']
    # Only use columns that actually exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    
    X = df[feature_cols]
    y = df['Efficiency']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict
    y_pred = model.predict(X_test)
    
    # Evaluate
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance (Random Forest):")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    
    # Feature Importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    print("\nFeature Importance:")
    print(importances.sort_values(ascending=False))
    
    return model

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # 1. Load Data (Set filepath to your CSV when you have it)
    # df = load_data("path/to/your/solar_data.csv")
    df = load_data(None) # Generates mock data for now
    
    # 2. Preprocess
    df_clean = preprocess_data(df)
    
    print(f"Data loaded. {len(df_clean)} valid daylight data points.")
    print(df_clean.head())
    
    # 3. Analysis
    analyze_correlations(df_clean)
    plot_relationships(df_clean)
    
    # 4. ML Modeling
    model = train_predictive_model(df_clean)
