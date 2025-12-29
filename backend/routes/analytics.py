from flask import Blueprint, jsonify, request
from backend.shared_state import set_simulation_active

bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')

@bp.route('/simulation/apply', methods=['POST'])
def apply_simulation():
    # In a real system, this would trigger complex state updates or write to DB
    set_simulation_active(True)
    return jsonify({
        'status': 'success',
        'message': 'Optimizations applied',
        'impact': {
            'throughput_increase': 22.1,
            'delay_reduction': 54.0
        }
    })

@bp.route('', methods=['GET'])
def get_analytics():
    # Placeholder for aggregate stats - normally would query a DB of established predictions
    return jsonify({
        'average_network_delay': 12.5,
        'critical_sections_count': 1,
        'prediction_accuracy_24h': 0.85
    })
