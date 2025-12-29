import numpy as np
import pandas as pd
import joblib
import os

class Preprocessor:
    def __init__(self, models_dir):
        self.scaler_X = joblib.load(os.path.join(models_dir, 'scaler_X.pkl'))
        self.scaler_y = joblib.load(os.path.join(models_dir, 'scaler_y.pkl'))
        self.le_train_type = joblib.load(os.path.join(models_dir, 'le_train_type.pkl'))
        
    def transform_input(self, train_data, section_data):
        """
        Derives features for the model using real-time data inputs.
        Expected Feature Vector: [hour, day_of_week, train_type_idx, station_id]
        """
        # Temporal Features
        now = pd.Timestamp.now()
        hour = now.hour
        day_of_week = now.dayofweek
        
        # Train Type Encoding
        try:
            train_type_idx = self.le_train_type.transform([train_data['type']])[0]
        except ValueError:
            # Handle unknown types via fallback or mapping
            train_type_idx = 0 
            
        # Station/Section ID Handling 
        # (Assuming simple extraction of numeric part for the model input, 
        # consistent with how training data was likely messy but model needs numbers)
        # Note: In a real scenario, we'd have a specific station encoder.
        # For this demo, we'll extract digits from the alphanumeric ID if possible, or hash it.
        try:
            station_id = int(''.join(filter(str.isdigit, section_data['id'])))
        except ValueError:
            station_id = 0
            
        # Construct Feature Vector
        features = np.array([[hour, day_of_week, train_type_idx, station_id]])
        
        # Scale
        features_scaled = self.scaler_X.transform(features)
        
        return torch.FloatTensor(features_scaled)

    def inverse_transform_output(self, prediction_scaled):
        return self.scaler_y.inverse_transform(prediction_scaled.detach().numpy().reshape(-1, 1))[0][0]

import torch # Need torch for tensor return type
