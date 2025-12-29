import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Configuration (Must match train_model.py)
DATA_PATH = r'dataset/archive/delays.csv'
MODEL_PATH = r'models/delay_predictor.pth'
SAMPLE_SIZE = None

def load_and_preprocess_data(path, sample_size=None):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path)
    
    if sample_size:
        df = df.head(sample_size)
    
    # Feature Engineering
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Encode categorical variables
    le_train_type = LabelEncoder()
    df['train_type_idx'] = le_train_type.fit_transform(df['train_type'])
    
    feature_cols = ['hour', 'day_of_week', 'train_type_idx', 'station_id']
    target_col = 'delay'
    
    X = df[feature_cols].values
    y = df[target_col].values.reshape(-1, 1)
    
    # Scaling
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    
    return X, y, scaler_X, scaler_y

class DelayPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    if not os.path.exists(MODEL_PATH):
        print("Model file not found.")
        return

    # Reproduce Data Pipeline
    X, y, scaler_X, scaler_y = load_and_preprocess_data(DATA_PATH, SAMPLE_SIZE)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    X_test_tensor = torch.FloatTensor(X_test)
    
    # Load Model
    model = DelayPredictor(input_dim=X.shape[1])
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    print("Evaluating model...")
    with torch.no_grad():
        predictions_scaled = model(X_test_tensor).numpy()
    
    # Inverse Transform
    predictions = scaler_y.inverse_transform(predictions_scaled)
    y_actual = scaler_y.inverse_transform(y_test)
    
    # Calculate Metrics
    mae = mean_absolute_error(y_actual, predictions)
    mse = mean_squared_error(y_actual, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, predictions)
    
    print("-" * 30)
    print(f"Model Accuracy Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.2f} minutes")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} minutes")
    print(f"R-squared Score (R2): {r2:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()
