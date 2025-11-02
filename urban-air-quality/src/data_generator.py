"""
Data Generator for Urban Air Quality Dataset
Generates synthetic air quality data with realistic patterns
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_air_quality_data(num_samples=10000, seed=42):
    """
    Generate synthetic air quality dataset with realistic patterns
    
    Parameters:
    -----------
    num_samples : int
        Number of samples to generate
    seed : int
        Random seed for reproducibility
    
    Returns:
    --------
    pd.DataFrame
        Generated air quality dataset
    """
    np.random.seed(seed)
    
    # Generate datetime range (2 years of hourly data)
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)]
    
    # Extract time features
    hours = np.array([d.hour for d in dates])
    days = np.array([d.day for d in dates])
    months = np.array([d.month for d in dates])
    day_of_week = np.array([d.weekday() for d in dates])
    
    # Generate weather features with seasonal patterns
    # Temperature: varies by month and hour
    temp_base = 15 + 10 * np.sin(2 * np.pi * months / 12)
    temp_daily = 5 * np.sin(2 * np.pi * hours / 24)
    temperature = temp_base + temp_daily + np.random.normal(0, 3, num_samples)
    temperature = np.clip(temperature, -5, 40)
    
    # Humidity: inverse relationship with temperature
    humidity = 70 - 0.5 * temperature + np.random.normal(0, 10, num_samples)
    humidity = np.clip(humidity, 20, 95)
    
    # Wind speed: higher during day, varies by season
    wind_speed = 5 + 3 * np.sin(2 * np.pi * hours / 24) + np.random.exponential(2, num_samples)
    wind_speed = np.clip(wind_speed, 0, 25)
    
    # Generate pollutant concentrations
    # PM2.5: higher in winter, during rush hours, lower with wind
    pm25_base = 30 + 20 * np.cos(2 * np.pi * months / 12)  # Seasonal
    pm25_traffic = 15 * ((hours >= 7) & (hours <= 9) | (hours >= 17) & (hours <= 19))  # Rush hours
    pm25_wind = -2 * wind_speed  # Wind disperses pollutants
    pm25 = pm25_base + pm25_traffic + pm25_wind + np.random.gamma(2, 5, num_samples)
    pm25 = np.clip(pm25, 0, 300)
    
    # PM10: correlated with PM2.5 but higher values
    pm10 = pm25 * 1.5 + np.random.gamma(2, 8, num_samples)
    pm10 = np.clip(pm10, 0, 500)
    
    # NO2: traffic-related, higher during weekdays and rush hours
    no2_base = 30
    no2_traffic = 20 * ((hours >= 7) & (hours <= 9) | (hours >= 17) & (hours <= 19))
    no2_weekday = 10 * (day_of_week < 5)  # Higher on weekdays
    no2 = no2_base + no2_traffic + no2_weekday + np.random.gamma(2, 5, num_samples)
    no2 = np.clip(no2, 0, 200)
    
    # CO: similar pattern to NO2 but different scale
    co = 0.5 + 0.3 * no2 / 50 + np.random.gamma(1, 0.3, num_samples)
    co = np.clip(co, 0, 10)
    
    # O3: photochemical, higher during sunny days (summer, midday)
    o3_seasonal = 40 + 30 * np.sin(2 * np.pi * months / 12)
    o3_daily = 20 * np.sin(2 * np.pi * (hours - 6) / 12)  # Peak at noon
    o3_daily = np.maximum(o3_daily, 0)
    o3 = o3_seasonal + o3_daily + np.random.gamma(2, 5, num_samples)
    o3 = np.clip(o3, 0, 200)
    
    # Calculate AQI based on pollutants (simplified)
    # Using PM2.5 as primary indicator with contributions from others
    aqi_pm25 = pm25 * 2
    aqi_pm10 = pm10 * 1
    aqi_no2 = no2 * 1.5
    aqi_o3 = o3 * 1.2
    aqi_co = co * 10
    
    aqi = np.maximum.reduce([aqi_pm25, aqi_pm10, aqi_no2, aqi_o3, aqi_co])
    aqi = np.clip(aqi, 0, 500)
    
    # Categorize AQI
    def categorize_aqi(value):
        if value <= 50:
            return 'Good'
        elif value <= 100:
            return 'Moderate'
        elif value <= 150:
            return 'Unhealthy for Sensitive Groups'
        elif value <= 200:
            return 'Unhealthy'
        elif value <= 300:
            return 'Very Unhealthy'
        else:
            return 'Hazardous'
    
    aqi_category = [categorize_aqi(val) for val in aqi]
    
    # Create DataFrame
    df = pd.DataFrame({
        'datetime': dates,
        'year': [d.year for d in dates],
        'month': months,
        'day': days,
        'hour': hours,
        'day_of_week': day_of_week,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'PM2.5': pm25,
        'PM10': pm10,
        'NO2': no2,
        'CO': co,
        'O3': o3,
        'AQI': aqi,
        'AQI_Category': aqi_category
    })
    
    return df

if __name__ == "__main__":
    # Generate dataset
    print("Generating air quality dataset...")
    df = generate_air_quality_data(num_samples=10000)
    
    # Save to CSV
    output_path = '../data/air_quality_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nFirst few rows:")
    print(df.head())
    print(f"\nDataset info:")
    print(df.info())
    print(f"\nAQI Category distribution:")
    print(df['AQI_Category'].value_counts())
