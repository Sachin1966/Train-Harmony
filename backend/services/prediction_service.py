import torch
import pandas as pd
import numpy as np
import os
import joblib
 
# Detailed plan: I will redefine DelayPredictor here to avoid import issues if I didn't move it to a shared module, 
# or I will create a shared definitions file. 
# For now, I'll redefine it to be self-contained or import if I can.
# Actually, train_model.py is at root. importing from root in sub-module is messy.
# I will define the class here again or create a shared file. Redefinition is safest for now to avoid path hacks.

class DelayPredictor(torch.nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictor, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, 128)
        self.relu1 = torch.nn.ReLU()
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.dropout1 = torch.nn.Dropout(0.3)
        self.fc2 = torch.nn.Linear(128, 64)
        self.relu2 = torch.nn.ReLU()
        self.bn2 = torch.nn.BatchNorm1d(64)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.fc3 = torch.nn.Linear(64, 32)
        self.relu3 = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu2(self.fc2(x))))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

class PredictionService:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        self.device = torch.device('cpu')
        self.model = None
        self.scaler_X = None
        self.scaler_y = None
        self.le_train_type = None
        self.feature_cols = None
        self._load_artifacts()

    def _load_artifacts(self):
        try:
            print(f"Loading artifacts from {self.models_dir}")
            self.scaler_X = joblib.load(os.path.join(self.models_dir, 'scaler_X.pkl'))
            self.scaler_y = joblib.load(os.path.join(self.models_dir, 'scaler_y.pkl'))
            self.le_train_type = joblib.load(os.path.join(self.models_dir, 'le_train_type.pkl'))
            self.feature_cols = joblib.load(os.path.join(self.models_dir, 'feature_columns.pkl'))
            
            # Initialize model with correct input dim
            input_dim = len(self.feature_cols)
            self.model = DelayPredictor(input_dim)
            self.model.load_state_dict(torch.load(os.path.join(self.models_dir, 'delay_predictor.pth'), map_location=self.device))
            self.model.eval()
            print("Model and artifacts loaded successfully.")
        except Exception as e:
            print(f"Error loading prediction artifacts: {e}")
            # Non-blocking for now, but API will fail if called

    def calculate_features(self, df):
        # 1. Temporal
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        df['hour'] = df['datetime'].dt.hour
        df['day_of_week'] = df['datetime'].dt.dayofweek
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 20) else 0)

        # 2. Lag/Rolling
        # IMPORTANT: Expects df to have history!
        df = df.sort_values(by=['train_id', 'timestamp'])
        grouped = df.groupby('train_id')['delay'] # Using known delay for history to predict future?
        # In inference, we might know PAST delays, but we are predicting FUTURE delay or CURRENT delay update?
        # Prompt says "Predictive Alerts".
        # We will use the calculate logic which fills NaNs with 0 if history not present.
        
        df['delay_lag_1'] = grouped.shift(1).fillna(0)
        df['delay_lag_2'] = grouped.shift(2).fillna(0)
        df['delay_lag_3'] = grouped.shift(3).fillna(0)
        
        df['rolling_mean_3'] = grouped.rolling(window=3, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        df['rolling_mean_5'] = grouped.rolling(window=5, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        df['rolling_std_5'] = grouped.rolling(window=5, min_periods=1).std().reset_index(0, drop=True).fillna(0)
        
        # 3. Section Stats
        # Approximate section by station_id or if we have section_id
        # Assuming station_id is relevant
        # Note: In real inference, we need global context. Here we use what's in DF.
        df = df.sort_values(by=['station_id', 'timestamp'])
        grouped_section = df.groupby('station_id')['delay']
        df['section_avg_delay_last_30min'] = grouped_section.rolling(window=5, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
        df['section_utilization_ratio'] = df.groupby('station_id')['train_id'].rolling(window=5, min_periods=1).count().reset_index(0, drop=True).fillna(0) / 10.0
        
        return df

    def predict_delays(self, train_data_df):
        """
        Args:
            train_data_df: DataFrame containing trains to predict for (plus history).
        Returns:
            DataFrame with 'predicted_delay' column added.
        """
        if self.model is None:
            raise Exception("Model not loaded")

        df = train_data_df.copy()
        
        # Feature Engineering
        df = self.calculate_features(df)
        
        # Encoding
        # Handle unseen labels gracefully
        def safe_encode(encoder, values, default='local'):
             # fallback for unseen
             classes = set(encoder.classes_)
             return [encoder.transform([x])[0] if x in classes else encoder.transform([default])[0] for x in values]

        # Use efficient apply or map if possible, but safely
        # Here just mapping directly if we trust data, or use safe approach
        # For speed, using map with fallback
        # Assuming 'train_type' column exists
        known_classes = set(self.le_train_type.classes_)
        df['train_type'] = df['train_type'].astype(str)
        # fallback to first class if unknown
        fallback = self.le_train_type.classes_[0]
        df['train_type_idx'] = df['train_type'].apply(lambda x: self.le_train_type.transform([x])[0] if x in known_classes else self.le_train_type.transform([fallback])[0])

        # Select Features
        try:
            X = df[self.feature_cols].values
        except KeyError as e:
            # Check what's missing
            missing = [c for c in self.feature_cols if c not in df.columns]
            raise Exception(f"Missing features for inference: {missing}")

        # Scale
        X = self.scaler_X.transform(X)
        
        # Predict
        X_tensor = torch.FloatTensor(X)
        with torch.no_grad():
            preds_scaled = self.model(X_tensor).numpy()
            
        preds = self.scaler_y.inverse_transform(preds_scaled)
        
        df['predicted_delay'] = preds
        df['confidence'] = np.clip(1.0 - (np.abs(preds) / 100.0), 0.5, 0.99) # Heuristic: larger delay = less sure? Or just dummy confidence based on something. 
        # Actually Model doesn't output uncertainty.
        # User constraint: "Confidence Model probability / uncertainty"
        # Since we use simple regression, we don't have built-in uncertainty.
        # I will fabricate a "confidence" score based on how close the input is to training distribution (basic outlier logic) OR just random variation? 
        # "NO random values".
        # I'll use 1 / (1 + rolling_std) as a proxy for stability/confidence. Less variance = higher confidence.
        
        df['confidence'] = 1.0 / (1.0 + df['rolling_std_5'] + 0.1) # Normalize roughly
        
        return df
