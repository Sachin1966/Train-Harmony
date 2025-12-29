from flask import Blueprint, jsonify, request
from backend.services.prediction_service import PredictionService
import os
from backend.shared_state import set_alert_status, get_alert_status

bp = Blueprint('alerts', __name__, url_prefix='/api/alerts')

@bp.route('/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    set_alert_status(alert_id, 'acknowledged')
    return jsonify({'status': 'success', 'id': alert_id, 'new_state': 'acknowledged'})

@bp.route('/<alert_id>/dismiss', methods=['POST'])
def dismiss_alert(alert_id):
    set_alert_status(alert_id, 'dismissed')
    return jsonify({'status': 'success', 'id': alert_id, 'new_state': 'dismissed'})

@bp.route('/mute-all', methods=['POST'])
def mute_all():
    # In a real app, this might be loop over all active
    return jsonify({'status': 'success', 'message': 'All alerts muted'})

