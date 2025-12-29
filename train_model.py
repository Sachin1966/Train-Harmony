import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from torch.utils.data import DataLoader, TensorDataset
import os
import joblib

# Configuration
DATA_PATH = r'dataset/archive/delays.csv'
MODEL_PATH = r'models/delay_predictor.pth'
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001
SAMPLE_SIZE = 100000  # Use a subset for faster training demonstration

def calculate_temporal_features(df):
    """Adds temporal features: hour sin/cos, day of week, peak hour."""
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    
    # Hour sin/cos
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Peak hour flag (Assuming 7-10 AM and 5-8 PM are peaks)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if (7 <= x <= 10) or (17 <= x <= 20) else 0)
    return df

def calculate_lag_rolling_features(df):
    """Adds lag and rolling features per train."""
    # Ensure data is sorted by train and time
    df = df.sort_values(by=['train_id', 'timestamp'])
    
    # Lag Features via shift based on train_id groups
    # We assume 'delay' column exists. If not, we might need to compute it from scheduled vs actual.
    # The prompt says "Historical train delay records" are available. 
    # The snippet showed 'delay' column.
    
    # Group by train_id to avoid leaking data between trains
    grouped = df.groupby('train_id')['delay']
    
    df['delay_lag_1'] = grouped.shift(1).fillna(0)
    df['delay_lag_2'] = grouped.shift(2).fillna(0)
    df['delay_lag_3'] = grouped.shift(3).fillna(0)
    
    # Rolling Features
    df['rolling_mean_3'] = grouped.rolling(window=3, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
    df['rolling_mean_5'] = grouped.rolling(window=5, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
    df['rolling_std_5'] = grouped.rolling(window=5, min_periods=1).std().reset_index(0, drop=True).fillna(0)
    
    return df

def calculate_section_features(df):
    """Adds section context features."""
    # This requires sorting by time to calculate "current" state
    # For a static dataset, we can compute rolling stats per section
    
    df = df.sort_values(by=['station_id', 'timestamp']) # approximating section by station_id
    
    grouped_section = df.groupby('station_id')['delay']
    
    # Average delay in last 3 records as a proxy for "last 30 min" if records are frequent
    # Or just rolling mean of delays at this station
    df['section_avg_delay_last_30min'] = grouped_section.rolling(window=5, min_periods=1).mean().reset_index(0, drop=True).fillna(0)
    
    # Utilization ratio requires capacity. We don't have capacity in CSV potentially.
    # We will simulate capacity or normalize train count.
    # Let's count active trains in a time window per station as proxy for utilization.
    # Simplified: rolling count of trains in last 5 ticks at this station
    df['section_utilization_ratio'] = df.groupby('station_id')['train_id'].rolling(window=5, min_periods=1).count().reset_index(0, drop=True).fillna(0) / 10.0 # Assign random capacity of 10 for normalization logic
    
    return df

def load_and_preprocess_data(path, sample_size=None):
    print(f"Loading data from {path}...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"File not found: {path}")
        # Create dummy data for development/demonstration if file missing (though prompt forbids, this is just to prevent crash if file genuinely missing in env logic)
        # But we saw files 90MB, so it should be there.
        raise
        
    if sample_size:
        df = df.head(sample_size)
    
    print(f"Data shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    
    # Normalize column names if needed (assuming based on previous snippet)
    # The snippet had: timestamp, train_no, train_type, station_id, delay_type, delay
    # We need 'train_id'. Let's rename 'train_no' to 'train_id' if present
    if 'train_no' in df.columns:
        df.rename(columns={'train_no': 'train_id'}, inplace=True)
    
    # 1. Feature Engineering
    df = calculate_temporal_features(df)
    df = calculate_lag_rolling_features(df)
    df = calculate_section_features(df)
    
    # 2. Encoding
    le_train_type = LabelEncoder()
    # Check if train_type exists, else fill
    if 'train_type' not in df.columns:
        df['train_type'] = 'local' # fallback
        
    df['train_type_idx'] = le_train_type.fit_transform(df['train_type'].astype(str))
    
    # 3. Selection
    feature_columns = [
        'hour_sin', 'hour_cos', 'day_of_week', 'is_peak_hour',
        'delay_lag_1', 'delay_lag_2', 'delay_lag_3',
        'rolling_mean_3', 'rolling_mean_5', 'rolling_std_5',
        'section_utilization_ratio', 'section_avg_delay_last_30min',
        'train_type_idx'
    ]
    
    target_col = 'delay'
    
    # Drop rows with NaNs generated by shifts if any remain (fillna was used but safe check)
    df_clean = df.dropna(subset=feature_columns + [target_col])
    
    X = df_clean[feature_columns].values
    y = df_clean[target_col].values.reshape(-1, 1)
    
    # 4. Scaling
    scaler_X = StandardScaler()
    X = scaler_X.fit_transform(X)
    
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y)
    
    return X, y, scaler_X, scaler_y, le_train_type, feature_columns

class DelayPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DelayPredictor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        self.fc3 = nn.Linear(64, 32)
        self.relu3 = nn.ReLU()
        
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = self.dropout1(self.bn1(self.relu1(self.fc1(x))))
        x = self.dropout2(self.bn2(self.relu2(self.fc2(x))))
        x = self.relu3(self.fc3(x))
        x = self.fc4(x)
        return x

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Error: Dataset not found at {DATA_PATH}")
        return

    # Load Data
    try:
         X, y, scaler_X, scaler_y, le_train_type, feature_cols = load_and_preprocess_data(DATA_PATH, SAMPLE_SIZE)
    except Exception as e:
        print(f"Failed during data loading: {e}")
        return
        
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Model
    model = DelayPredictor(input_dim=X.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss/len(train_loader):.4f}")
        
    # Eval
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f"Test Loss (MSE): {test_loss.item():.4f}")
        
    # Save
    if not os.path.exists('models'):
        os.makedirs('models')
        
    torch.save(model.state_dict(), MODEL_PATH)
    
    # Save Artifacts
    joblib.dump(scaler_X, 'models/scaler_X.pkl')
    joblib.dump(scaler_y, 'models/scaler_y.pkl')
    joblib.dump(le_train_type, 'models/le_train_type.pkl')
    joblib.dump(feature_cols, 'models/feature_columns.pkl') # Save feature list
    
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
