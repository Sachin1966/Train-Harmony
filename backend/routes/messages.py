from flask import Blueprint, jsonify, request
from backend.services.email_service import send_email
from backend.database import create_message, get_user_messages, mark_message_read, get_user_by_email

bp = Blueprint('messages', __name__, url_prefix='/api/messages')

@bp.route('/send', methods=['POST'])
def send_message():
    data = request.get_json()
    
    sender_email = data.get('sender_email') # Ideally get from Auth token, but for now trusting frontend or using a default
    recipient = data.get('recipient_email')
    subject = data.get('subject')
    message = data.get('message')
    
    if not all([recipient, subject, message]):
        return jsonify({'error': 'Missing required fields'}), 400
    
    # Resolve sender ID (simple lookup)
    sender = get_user_by_email(sender_email) if sender_email else None
    sender_id = sender['id'] if sender else 'AD-001' # Default to Admin if not specified
        
    # 1. Persist to Database
    db_success = create_message(sender_id, recipient, subject, message)
    
    # 2. Send via Email
    email_success = send_email(recipient, subject, message)
    
    if db_success:
        return jsonify({
            'message': 'Message sent successfully',
            'email_sent': email_success
        }), 200
    else:
        return jsonify({'error': 'Failed to save message'}), 500

@bp.route('/', methods=['GET'])
def get_messages():
    email = request.args.get('email')
    if not email:
        return jsonify({'error': 'Email required'}), 400
        
    messages = get_user_messages(email)
    return jsonify(messages)

@bp.route('/<int:message_id>/read', methods=['POST'])
def read_message(message_id):
    mark_message_read(message_id)
    return jsonify({'success': True})
