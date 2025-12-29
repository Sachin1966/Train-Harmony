# Train-Harmony (RailFlow AI)

**Train-Harmony** (also known as RailFlow AI) is an advanced AI-powered platform designed to optimize train schedules and predict delays. By leveraging historical data and real-time telemetry, the system uses deep learning to forecast delays and suggest schedule optimizations, aiming to improve the efficiency and reliability of railway networks.

## 🚀 Key Features

*   **AI-Powered Delay Prediction**: Uses a PyTorch-based neural network (`DelayPredictor`) to forecast train delays based on temporal patterns, rolling statistics, and section utilization.
*   **Real-Time Dashboard**: A modern, interactive web interface built with React and Tailwind CSS to visualize schedules and delay metrics.
*   **Optimization Engine**: Analyzes section capacity and train loads to suggest schedule adjustments.
*   **Scalable Architecture**: Decoupled Flask backend and Vite+React frontend for flexibility and performance.
*   **Git LFS Support**: Efficient handling of large datasets within the repository.

## 🛠️ Tech Stack

### Frontend
*   **Framework**: [React](https://react.dev/) v18
*   **Build Tool**: [Vite](https://vitejs.dev/)
*   **Language**: [TypeScript](https://www.typescriptlang.org/)
*   **Styling**: [Tailwind CSS](https://tailwindcss.com/)
*   **Components**: [shadcn/ui](https://ui.shadcn.com/) (Radix UI)
*   **Data Fetching**: TanStack Query

### Backend
*   **Framework**: [Flask](https://flask.palletsprojects.com/)
*   **API**: RESTful API with `flask-cors`
*   **Serialization**: Marshmallow
*   **Database**: SQLite (`railflow.db`)

### Machine Learning
*   **Framework**: [PyTorch](https://pytorch.org/) (Neural Networks)
*   **Data Processing**: Pandas, NumPy, Scikit-learn
*   **Model Storage**: Joblib (scalers/encoders), Torch (`.pth`)

## 📂 Project Structure

```bash
├── app.py                  # Entry point for the Backend Server
├── backend/                # Flask Application Logic
│   ├── routes/             # API Endpoints
│   ├── services/           # Business Logic
│   ├── models/             # Database Models
│   └── database.py         # Database Connection
├── train-harmony/          # Frontend React Application
│   ├── src/                # UI Components and Logic
│   └── package.json        # Frontend Dependencies
├── train_model.py          # Script to training the Delay Prediction Model
├── dataset/                # Data storage (delays.csv, schedules.json)
├── models/                 # Saved ML models (.pth) and scalers (.pkl)
└── requirements.txt        # Backend Python Dependencies
```

## ⚙️ Installation & Setup

### Prerequisites
*   **Python 3.8+**
*   **Node.js 18+** & **npm**
*   **Git LFS** (Required for large dataset files)

### 1. Clone the Repository
Ensure Git LFS is installed and initialized before cloning to pull large files correctly.
```bash
git lfs install
git clone https://github.com/Sachin1966/Train-Harmony.git
cd Train-Harmony
```

### 2. Backend Setup
Create a virtual environment and install Python dependencies.
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/Mac
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r backend/requirements.txt
```

### 3. Frontend Setup
Navigate to the frontend directory and install Node modules.
```bash
cd train-harmony
npm install
```

## 🏃‍♂️ Usage

### Training the AI Model
Before running the dashboard, ensure the AI model is trained and artifacts are generated.
```bash
# From the project root (with venv activated)
python train_model.py
```
This will read data from `dataset/`, train the neural network, and save the model to `models/delay_predictor.pth` along with necessary scalers.

### Running the Application

**Option 1: Run Backend and Frontend Separately (Recommended for Dev)**

1.  **Start Backend:**
    ```bash
    # Terminal 1 (Root Dir)
    python app.py
    ```
    The API will run at `http://localhost:5000`.

2.  **Start Frontend:**
    ```bash
    # Terminal 2 (train-harmony Dir)
    cd train-harmony
    npm run dev
    ```
    The UI will be accessible at the URL provided by Vite (usually `http://localhost:8080` or similar).

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

