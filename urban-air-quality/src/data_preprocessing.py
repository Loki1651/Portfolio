"""
Data Preprocessing Module for Air Quality Analysis
Handles data loading, cleaning, feature engineering, and train-test split
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os

class AirQualityPreprocessor:
    """
    Preprocessor for air quality data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        
    def load_data(self, filepath):
        """Load air quality data from CSV"""
        df = pd.read_csv(filepath)
        df['datetime'] = pd.to_datetime(df['datetime'])
        return df
    
    def create_features(self, df):
        """
        Create additional features from existing data
        
        Features created:
        - Rolling averages for pollutants
        - Time-based features (is_weekend, is_rush_hour)
        - Interaction features
        """
        df = df.copy()
        
        # Time-based features
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_rush_hour'] = ((df['hour'] >= 7) & (df['hour'] <= 9) | 
                              (df['hour'] >= 17) & (df['hour'] <= 19)).astype(int)
        
        # Cyclical encoding for hour (to capture circular nature of time)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Cyclical encoding for month
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Rolling averages (3-hour window) for pollutants
        pollutants = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3']
        for pollutant in pollutants:
            df[f'{pollutant}_rolling_3h'] = df[pollutant].rolling(window=3, min_periods=1).mean()
        
        # Interaction features
        df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
        df['pm_ratio'] = df['PM2.5'] / (df['PM10'] + 1)  # +1 to avoid division by zero
        
        return df
    
    def prepare_features(self, df, target_column='AQI_Category'):
        """
        Prepare features for modeling
        
        Returns:
        - X: Feature matrix
        - y: Target variable
        - feature_names: List of feature names
        """
        # Select feature columns
        feature_cols = [
            'temperature', 'humidity', 'wind_speed',
            'PM2.5', 'PM10', 'NO2', 'CO', 'O3',
            'hour', 'day_of_week', 'month',
            'is_weekend', 'is_rush_hour',
            'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
            'PM2.5_rolling_3h', 'PM10_rolling_3h', 'NO2_rolling_3h', 
            'CO_rolling_3h', 'O3_rolling_3h',
            'temp_humidity_interaction', 'pm_ratio'
        ]
        
        X = df[feature_cols].copy()
        y = df[target_column].copy()
        
        self.feature_columns = feature_cols
        
        return X, y, feature_cols
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, stratify=y)
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled
    
    def encode_labels(self, y_train, y_test):
        """Encode categorical labels to numeric"""
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)
        
        return y_train_encoded, y_test_encoded
    
    def save_preprocessor(self, output_dir='../models'):
        """Save scaler and label encoder"""
        os.makedirs(output_dir, exist_ok=True)
        joblib.dump(self.scaler, os.path.join(output_dir, 'scaler.pkl'))
        joblib.dump(self.label_encoder, os.path.join(output_dir, 'label_encoder.pkl'))
        print(f"Preprocessor saved to {output_dir}")
    
    def load_preprocessor(self, model_dir='../models'):
        """Load saved scaler and label encoder"""
        self.scaler = joblib.load(os.path.join(model_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(model_dir, 'label_encoder.pkl'))
        print(f"Preprocessor loaded from {model_dir}")

def main():
    """Main preprocessing pipeline"""
    print("=" * 60)
    print("AIR QUALITY DATA PREPROCESSING")
    print("=" * 60)
    
    # Initialize preprocessor
    preprocessor = AirQualityPreprocessor()
    
    # Load data
    print("\n1. Loading data...")
    df = preprocessor.load_data('../data/air_quality_data.csv')
    print(f"   Data shape: {df.shape}")
    print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
    
    # Create features
    print("\n2. Creating features...")
    df = preprocessor.create_features(df)
    print(f"   Total features created: {df.shape[1]}")
    
    # Prepare features
    print("\n3. Preparing features for modeling...")
    X, y, feature_names = preprocessor.prepare_features(df)
    print(f"   Feature matrix shape: {X.shape}")
    print(f"   Number of features: {len(feature_names)}")
    
    # Split data
    print("\n4. Splitting data...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set: {X_test.shape[0]} samples")
    
    # Scale features
    print("\n5. Scaling features...")
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    print(f"   Features scaled using StandardScaler")
    
    # Encode labels
    print("\n6. Encoding labels...")
    y_train_encoded, y_test_encoded = preprocessor.encode_labels(y_train, y_test)
    print(f"   Labels encoded: {list(preprocessor.label_encoder.classes_)}")
    
    # Save preprocessor
    print("\n7. Saving preprocessor...")
    preprocessor.save_preprocessor()
    
    # Save processed data
    print("\n8. Saving processed data...")
    np.save('../data/X_train.npy', X_train_scaled)
    np.save('../data/X_test.npy', X_test_scaled)
    np.save('../data/y_train.npy', y_train_encoded)
    np.save('../data/y_test.npy', y_test_encoded)
    
    # Save feature names
    with open('../data/feature_names.txt', 'w') as f:
        f.write('\n'.join(feature_names))
    
    print("   Processed data saved to ../data/")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    
    # Display summary statistics
    print("\nTarget Distribution:")
    print(pd.Series(y_train).value_counts().sort_index())

if __name__ == "__main__":
    main()
