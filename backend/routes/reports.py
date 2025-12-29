from flask import Blueprint, jsonify
import datetime

bp = Blueprint('reports', __name__, url_prefix='/api/reports')

@bp.route('', methods=['GET'])
def get_reports():
    # Mocked reports list
    # In a real system, this would scan a storage bucket or DB
    today = datetime.date.today()
    reports = [
        {
            'id': 'RPT-2025-001',
            'title': 'Daily Operations Summary',
            'type': 'pdf',
            'date': today.isoformat(),
            'size': '2.4 MB',
            'status': 'Ready'
        },
        {
            'id': 'RPT-2025-002',
            'title': 'Weekly Incident Log',
            'type': 'csv',
            'date': (today - datetime.timedelta(days=1)).isoformat(),
            'size': '0.8 MB',
            'status': 'Ready'
        },
        {
            'id': 'RPT-2025-003',
            'title': 'System Performance Audit',
            'type': 'pdf',
            'date': (today - datetime.timedelta(days=2)).isoformat(),
            'size': '5.1 MB',
            'status': 'Archived'
        },
         {
            'id': 'RPT-2025-004',
            'title': 'Staff Scheduling Optimization',
            'type': 'xlsx',
            'date': (today - datetime.timedelta(days=3)).isoformat(),
            'size': '1.2 MB',
            'status': 'Ready'
        }
    ]
    return jsonify(reports)
