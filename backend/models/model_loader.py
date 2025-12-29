import torch
import torch.nn as nn
import os

class DelayPredictor(nn.Module):
    def __init__(self, input_dim=4):
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

def load_model(models_dir):
    model_path = os.path.join(models_dir, 'delay_predictor.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
        
    model = DelayPredictor()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model
