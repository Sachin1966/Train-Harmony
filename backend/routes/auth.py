from flask import Blueprint, request, jsonify
from backend.database import get_user_by_email, create_user
from werkzeug.security import check_password_hash
import uuid

bp = Blueprint('auth', __name__, url_prefix='/api/auth')

@bp.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    
    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400
        
    user_row = get_user_by_email(email)
    
    if not user_row:
        return jsonify({'error': 'Invalid credentials'}), 401
        
    # user_row is a sqlite3.Row, access by name
    user = dict(user_row)
    
    if check_password_hash(user['password'], password):
        # Successful login
        # Update status to Online
        from backend.database import update_user_status
        update_user_status(user['id'], 'Online')
        
        # Remove password from response
        del user['password']
        user['status'] = 'Online' # Return updated status
        
        # In a real app we'd issue a JWT here. 
        # For this prototype, we'll return the user object and a dummy token.
        return jsonify({
            'message': 'Login successful',
            'user': user,
            'token': str(uuid.uuid4())
        })
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@bp.route('/logout', methods=['POST'])
def logout():
    data = request.get_json()
    user_id = data.get('user_id')
    
    if user_id:
        from backend.database import update_user_status
        update_user_status(user_id, 'Offline')
        return jsonify({'message': 'Logged out successfully'}), 200
    return jsonify({'message': 'No user ID provided'}), 400

@bp.route('/me', methods=['GET'])
def get_current_user():
    user_id = request.args.get('user_id')
    if not user_id:
        return jsonify({'error': 'User ID required'}), 400
        
    from backend.database import get_user_by_id
    user_row = get_user_by_id(user_id)
    if user_row:
        user = dict(user_row)
        if 'password' in user:
            del user['password']
        return jsonify(user)
    return jsonify({'error': 'User not found'}), 404
