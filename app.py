import os
import sys

# Add project root to path explicitly (though running this file usually sets it)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the configured app from backend/app.py
from backend.app import app

# Note: backend/app.py already registers all blueprints including dashboard.
# It also initializes services.

if __name__ == '__main__':
    print("Starting RailFlow AI Server...")
    print("Serving API at: http://localhost:5000/api/")
    app.run(port=5000, debug=True)
