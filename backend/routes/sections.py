from flask import Blueprint, jsonify
from backend.services.prediction_service import PredictionService
import os

bp = Blueprint('sections', __name__, url_prefix='/api/sections')
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models')
prediction_service = PredictionService(MODELS_DIR)

SECTIONS_METADATA = [
    {'id': 'DLI-GZB-01', 'capacity': 12, 'occupancy': 8},
    {'id': 'CSMT-DDR-01', 'capacity': 15, 'occupancy': 14},
    {'id': 'GZB-MRT-01', 'capacity': 10, 'occupancy': 4},
]

@bp.route('', methods=['GET'])
def get_sections():
    print("Processing sections request...")
    results = []
    
    for section in SECTIONS_METADATA:
        # ML-driven Risk Assessment
        # We simulate a "probe train" to gauge section delay risk
        probe_train = {'id': 'PROBE', 'type': 'express'} 
        section_input = {'id': section['id']}
        
        prediction = prediction_service.predict_delay(probe_train, section_input)
        
        utilization = (section['occupancy'] / section['capacity']) * 100
        
        # Determine Congestion Level based on BOTH physical occupancy AND predicted delay
        if utilization > 90 or prediction['risk_level'] == 'CRITICAL':
            congestion = 'critical'
        elif utilization > 70 or prediction['risk_level'] == 'HIGH':
            congestion = 'high'
        elif utilization > 40:
            congestion = 'medium'
        else:
            congestion = 'low'
            
        results.append({
            'id': section['id'],
            'capacity': section['capacity'],
            'utilization_pct': round(utilization, 1),
            'congestion_level': congestion,
            'predicted_delay_impact': round(prediction['predicted_delay_minutes'], 1)
        })
        
    return jsonify(results)
