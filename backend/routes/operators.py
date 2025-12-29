from flask import Blueprint, jsonify, request
from backend.database import get_all_operators, create_user, get_user_by_id

bp = Blueprint('operators', __name__, url_prefix='/api/operators')

@bp.route('', methods=['GET'])
def get_operators():
    operators = get_all_operators()
    return jsonify(operators)

@bp.route('', methods=['POST'])
def add_operator():
    data = request.get_json()
    
    # Basic validation
    required = ['name', 'email', 'password', 'role']
    for field in required:
        if field not in data:
            return jsonify({'error': f'Missing required field: {field}'}), 400
            
    # Generate ID if not provided (simple custom logic)
    # real app might let DB handle auto-increment or UUID
    # Here we simulate the ID format OP-XXX
    import random
    new_id = f"OP-{random.randint(100, 999)}"
    
    user_data = {
        'id': new_id,
        'name': data['name'],
        'email': data['email'],
        'password': data['password'],
        'role': data['role'],
        'status': 'Offline', # Default status
        'shift': data.get('shift', 'Morning'),
        'location': data.get('location', 'Headquarters')
    }
    
    success = create_user(user_data)
    
    if success:
        # Return the created user (without password)
        created_user = get_user_by_id(new_id)
        user_dict = dict(created_user)
        del user_dict['password']
        return jsonify(user_dict), 201
    else:
        return jsonify({'error': 'Failed to create operator. Email might already exist.'}), 409
