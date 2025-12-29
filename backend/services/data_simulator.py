import pandas as pd
import numpy as np
import os
import datetime

class DataSimulator:
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self._load_data()
        
    def _load_data(self):
        if not os.path.exists(self.data_path):
            print(f"Dataset not found at {self.data_path}")
            # Create a minimal dummy DF to prevent crash if file missing, 
            # though prompt implies file is there.
            self.df = pd.DataFrame(columns=[
                'train_id', 'timestamp', 'station_id', 'delay', 
                'train_type', 'station_name'
            ])
            return

        print(f"Loading simulation data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)
        
        # Standardize columns
        if 'train_no' in self.df.columns:
            self.df.rename(columns={'train_no': 'train_id'}, inplace=True)
            
        # Ensure timestamp is int/float
        self.df['timestamp'] = pd.to_numeric(self.df['timestamp'], errors='coerce')
        self.df = self.df.dropna(subset=['timestamp'])
        self.df = self.df.sort_values('timestamp')
        
        # Cache min/max time for rollover simulation
        self.min_time = self.df['timestamp'].min()
        self.max_time = self.df['timestamp'].max()
        self.duration = self.max_time - self.min_time
        print(f"Data loaded. Time range: {self.min_time} to {self.max_time}")

    def get_current_state(self, requested_time=None):
        """
        Returns a DataFrame containing active trains and their recent history (for lag/rolling features).
        Simulates 'current time' by mapping real clock to dataset time.
        """
        if self.df is None or self.df.empty:
            return pd.DataFrame()

        # 1. Determine "Simulation Time"
        # If requested_time is None, use system time mapped to dataset range
        # to ensure we always show data "moving"
        if requested_time is None:
            now = datetime.datetime.now().timestamp()
            # Map 'now' to a point in the dataset cycle
            # Use module arithmetic to cycle through dataset every X duration
            # Offset by min_time
            offset = (now % self.duration)
            sim_time = self.min_time + offset
        else:
            sim_time = requested_time

        # 2. Define Window
        # We want to show trains active "now".
        # Let's say "active" means they had a record in the last 60 minutes.
        window_size = 3600 * 2 # 2 hours to be safe to capture history for rolling
        start_time = sim_time - window_size
        
        mask = (self.df['timestamp'] >= start_time) & (self.df['timestamp'] <= sim_time)
        window_df = self.df.loc[mask].copy()
        
        if window_df.empty:
            # If empty (gap in data), try jumping forward or pick random window?
            # Prompt: "No random values".
            # If gap, user sees empty. That's real data behavior.
            return pd.DataFrame()
            
        return window_df, sim_time

    def get_throughput_stats(self, end_time, duration_seconds=3600):
        # Count unique trains that *completed* or appeared in the window?
        # Throughput = Completed trains. 
        # Hard to know "completed" without destination logic.
        # Proxy: Number of unique trains observed in the window.
        start_time = end_time - duration_seconds
        mask = (self.df['timestamp'] >= start_time) & (self.df['timestamp'] <= end_time)
        subset = self.df.loc[mask]
        return subset['train_id'].nunique()
