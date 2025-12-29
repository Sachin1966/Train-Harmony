from flask import Flask
from flask_cors import CORS
import os
import sys

# Add project root to path for imports to work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.prediction_service import PredictionService

app = Flask(__name__)
CORS(app)

# Initialize Services
# Initialize Services
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset', 'archive', 'delays.csv')

from backend.services.prediction_service import PredictionService
from backend.services.data_simulator import DataSimulator

# Create instances (attached to app context later or global)
prediction_service = PredictionService(MODELS_DIR)
data_simulator = DataSimulator(DATA_PATH)

app.prediction_service = prediction_service
app.data_simulator = data_simulator

# Import Routes
from backend.routes import trains, sections, alerts, analytics, dashboard, reports, operators, messages

app.register_blueprint(trains.bp)
app.register_blueprint(sections.bp)
app.register_blueprint(alerts.bp)
app.register_blueprint(analytics.bp)
app.register_blueprint(dashboard.bp)
app.register_blueprint(reports.bp)
app.register_blueprint(operators.bp)
app.register_blueprint(messages.bp)

# Import Auth and DB
from backend.routes import auth
from backend.database import init_db

app.register_blueprint(auth.bp)
from backend.routes import settings
app.register_blueprint(settings.bp)

# Initialize Database
with app.app_context():
    init_db()

@app.route('/health')
def health_check():
    return {'status': 'healthy', 'components': ['model_loaded']}

@app.route('/')
def index():
    return """
    <div style="font-family: sans-serif; text-align: center; padding: 2rem;">
        <h1>RailFlow AI Backend is Running</h1>
        <p>This is the API server.</p>
        <p>Please visit the frontend dashboard at: <a href="http://localhost:8080">http://localhost:8080</a></p>
    </div>
    """

if __name__ == '__main__':
    app.run(port=5000, debug=True)
