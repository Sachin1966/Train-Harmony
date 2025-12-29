from flask import Blueprint, request, jsonify
from backend.database import get_user_settings, update_user_settings, update_user_profile, change_user_password, get_user_by_id
from werkzeug.security import check_password_hash

bp = Blueprint('settings', __name__, url_prefix='/api/settings')

@bp.route('', methods=['GET'])
def get_settings():
    # In a real app we'd get user_id from token/session.
    # For this prototype we expect user_id in query param or header, 
    # but let's assume the frontend sends user_id for simplicity or we use a fixed admin for now if missing.
    # Ideally, we should use the token.
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
        
    settings = get_user_settings(user_id)
    return jsonify(settings)

@bp.route('', methods=['PUT'])
def update_settings():
    data = request.get_json()
    user_id = data.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
        
    success = update_user_settings(user_id, data)
    if success:
        return jsonify({'message': 'Settings updated successfully'})
    else:
        return jsonify({'error': 'Failed to update settings'}), 500

@bp.route('/profile', methods=['PUT'])
def update_profile():
    data = request.get_json()
    user_id = data.get('user_id')
    name = data.get('name')
    email = data.get('email')
    
    if not all([user_id, name, email]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    success = update_user_profile(user_id, name, email)
    if success:
        return jsonify({'message': 'Profile updated successfully'})
    else:
        return jsonify({'error': 'Failed to update profile. Email may be in use.'}), 400

@bp.route('/password', methods=['PUT'])
def update_password():
    data = request.get_json()
    user_id = data.get('user_id')
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not all([user_id, current_password, new_password]):
        return jsonify({'error': 'Missing required fields'}), 400
        
    # Verify current password
    user = get_user_by_id(user_id)
    if not user or not check_password_hash(user['password'], current_password):
        return jsonify({'error': 'Invalid current password'}), 401
        
    success = change_user_password(user_id, new_password)
    if success:
        return jsonify({'message': 'Password changed successfully'})
    else:
        return jsonify({'error': 'Failed to update password'}), 500
