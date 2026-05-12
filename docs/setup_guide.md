# Setup Guide

This guide will walk you through the process of setting up and running the Stock Manipulation Detection System on your local machine.

## Prerequisites

Before you begin, ensure you have the following installed:
- **Python 3.12+**
- **MongoDB** (Running locally on default port 27017)
- **Git**

## 1. Clone the Repository

If you haven't already, clone the project and navigate into the directory:

```bash
git clone https://github.com/Kamalakannan-Tech/Stock-Manipulation-Detector.git
cd Stock-Manipulation-Detector
```

## 2. Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

**Windows:**
```powershell
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the required Python packages:
```bash
pip install -r requirements.txt
```

## 3. Configuration

1. Copy the example environment file to create your local `.env` file:
   **Windows:**
   ```powershell
   copy .env.example .env
   ```
   **macOS/Linux:**
   ```bash
   cp .env.example .env
   ```
2. Open the `.env` file in your text editor and update the variables if necessary. By default, it is configured for local MongoDB development. If you have API keys for social media collection (e.g., Reddit, Twitter), add them here.

## 4. Initializing the System

You have two options for running the system: **Demo Mode** or **Full ML Mode**.

### Option A: Quick Start (Demo Mode)
If you want to run the application immediately without training the Machine Learning model, the system will fall back to rule-based heuristics.

1. Ensure MongoDB is running (open a terminal and run `mongod`).
2. Start the API server:
   ```bash
   python api/app.py
   ```
3. Open `frontend/index.html` in your web browser to view the dashboard.

### Option B: Full System with ML Model
To fully utilize the Temporal Fusion Transformer for predictions, you must collect data and train the model first.

1. **Ensure MongoDB is running:**
   ```bash
   mongod
   ```

2. **Collect Historical Data** (e.g., 3 months of data):
   ```bash
   python scripts/02_collect_historical_data.py --months 3
   ```

3. **Preprocess Data** (generates labels and features):
   ```bash
   python scripts/03_preprocess_data.py
   ```

4. **Train the Model**:
   ```bash
   python scripts/04_train_model.py --epochs 10
   ```

5. **Start the API Server**:
   ```bash
   python api/app.py
   ```

6. Open `frontend/index.html` in your web browser.

## Troubleshooting

- **MongoDB Connection Error:** Ensure the MongoDB daemon (`mongod`) is running in the background.
- **Model Not Found Error:** You are trying to use ML predictions without training the model. Either follow "Option B" to train it, or the system will automatically fall back to rule-based detection.
- **API/Social Data Errors:** If you haven't provided API credentials in your `.env` file, the real-time social data collectors may fail or return empty data. The system handles this gracefully in Demo Mode.
