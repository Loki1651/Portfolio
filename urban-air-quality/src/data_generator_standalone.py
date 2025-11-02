"""
Standalone Data Generator for Urban Air Quality Dataset
Generates synthetic air quality data without external dependencies
"""

import random
import math
import csv
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
    list of dict
        Generated air quality dataset
    """
    random.seed(seed)
    
    # Generate datetime range (hourly data)
    start_date = datetime(2022, 1, 1)
    
    data = []
    
    for i in range(num_samples):
        current_date = start_date + timedelta(hours=i)
        
        hour = current_date.hour
        day = current_date.day
        month = current_date.month
        year = current_date.year
        day_of_week = current_date.weekday()
        
        # Generate weather features with seasonal patterns
        # Temperature: varies by month and hour
        temp_base = 15 + 10 * math.sin(2 * math.pi * month / 12)
        temp_daily = 5 * math.sin(2 * math.pi * hour / 24)
        temperature = temp_base + temp_daily + random.gauss(0, 3)
        temperature = max(-5, min(40, temperature))
        
        # Humidity: inverse relationship with temperature
        humidity = 70 - 0.5 * temperature + random.gauss(0, 10)
        humidity = max(20, min(95, humidity))
        
        # Wind speed: higher during day
        wind_speed = 5 + 3 * math.sin(2 * math.pi * hour / 24) + random.expovariate(0.5)
        wind_speed = max(0, min(25, wind_speed))
        
        # Generate pollutant concentrations
        # PM2.5: higher in winter, during rush hours, lower with wind
        pm25_base = 30 + 20 * math.cos(2 * math.pi * month / 12)
        pm25_traffic = 15 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        pm25_wind = -2 * wind_speed
        pm25 = pm25_base + pm25_traffic + pm25_wind + random.gammavariate(2, 5)
        pm25 = max(0, min(300, pm25))
        
        # PM10: correlated with PM2.5 but higher values
        pm10 = pm25 * 1.5 + random.gammavariate(2, 8)
        pm10 = max(0, min(500, pm10))
        
        # NO2: traffic-related, higher during weekdays and rush hours
        no2_base = 30
        no2_traffic = 20 if (7 <= hour <= 9 or 17 <= hour <= 19) else 0
        no2_weekday = 10 if day_of_week < 5 else 0
        no2 = no2_base + no2_traffic + no2_weekday + random.gammavariate(2, 5)
        no2 = max(0, min(200, no2))
        
        # CO: similar pattern to NO2
        co = 0.5 + 0.3 * no2 / 50 + random.gammavariate(1, 0.3)
        co = max(0, min(10, co))
        
        # O3: photochemical, higher during sunny days
        o3_seasonal = 40 + 30 * math.sin(2 * math.pi * month / 12)
        o3_daily = 20 * math.sin(2 * math.pi * (hour - 6) / 12)
        o3_daily = max(0, o3_daily)
        o3 = o3_seasonal + o3_daily + random.gammavariate(2, 5)
        o3 = max(0, min(200, o3))
        
        # Calculate AQI
        aqi = max(pm25 * 2, pm10 * 1, no2 * 1.5, o3 * 1.2, co * 10)
        aqi = max(0, min(500, aqi))
        
        # Categorize AQI
        if aqi <= 50:
            aqi_category = 'Good'
        elif aqi <= 100:
            aqi_category = 'Moderate'
        elif aqi <= 150:
            aqi_category = 'Unhealthy for Sensitive Groups'
        elif aqi <= 200:
            aqi_category = 'Unhealthy'
        elif aqi <= 300:
            aqi_category = 'Very Unhealthy'
        else:
            aqi_category = 'Hazardous'
        
        data.append({
            'datetime': current_date.strftime('%Y-%m-%d %H:%M:%S'),
            'year': year,
            'month': month,
            'day': day,
            'hour': hour,
            'day_of_week': day_of_week,
            'temperature': round(temperature, 2),
            'humidity': round(humidity, 2),
            'wind_speed': round(wind_speed, 2),
            'PM2.5': round(pm25, 2),
            'PM10': round(pm10, 2),
            'NO2': round(no2, 2),
            'CO': round(co, 2),
            'O3': round(o3, 2),
            'AQI': round(aqi, 2),
            'AQI_Category': aqi_category
        })
    
    return data

if __name__ == "__main__":
    print("Generating air quality dataset...")
    data = generate_air_quality_data(num_samples=10000)
    
    # Save to CSV
    output_path = '../data/air_quality_data.csv'
    
    fieldnames = ['datetime', 'year', 'month', 'day', 'hour', 'day_of_week',
                  'temperature', 'humidity', 'wind_speed', 'PM2.5', 'PM10',
                  'NO2', 'CO', 'O3', 'AQI', 'AQI_Category']
    
    with open(output_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Dataset saved to {output_path}")
    print(f"Dataset size: {len(data)} rows, {len(fieldnames)} columns")
    print(f"\nFirst 5 rows:")
    for i in range(min(5, len(data))):
        print(data[i])
    
    # Count AQI categories
    category_counts = {}
    for row in data:
        cat = row['AQI_Category']
        category_counts[cat] = category_counts.get(cat, 0) + 1
    
    print(f"\nAQI Category distribution:")
    for cat, count in sorted(category_counts.items()):
        print(f"  {cat}: {count}")
