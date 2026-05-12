# Architecture Overview

The Stock Manipulation Detection System is a full-stack, AI-driven application designed to detect market manipulation by fusing market data with social media sentiment. The system is built with a modular architecture, separating data collection, model inference, and presentation layers.

## High-Level Architecture

The system is composed of the following core components:

1. **Data Collection Layer**
2. **Preprocessing & Feature Engineering**
3. **Machine Learning Models**
4. **Inference & Alerting Engine**
5. **REST API & WebSocket Server**
6. **Frontend Dashboard**
7. **Database Storage**

---

### 1. Data Collection Layer (`src/data_collection/`)
This module is responsible for ingesting raw data from multiple sources:
- **Market Data**: Fetches historical and real-time stock price, volume, and technical indicators (e.g., using Yahoo Finance).
- **Social Media Data**: Scrapes and aggregates sentiment and discussion volume from platforms like Twitter, Reddit, and StockTwits.

### 2. Preprocessing & Feature Engineering (`src/preprocessing/`)
Raw data is cleaned, aligned, and transformed into features suitable for the machine learning model.
- **Labeler (`labeler.py`)**: Applies heuristics to historical data to generate "manipulation" labels (typically aiming for a 5-15% anomaly rate) for supervised learning.
- **Data Fusing**: Combines market metrics (volatility, volume spikes) with social metrics (sentiment scores, mention velocity).

### 3. Machine Learning Models (`src/models/`)
The predictive core of the system.
- **Temporal Fusion Transformer (TFT)**: A PyTorch-based model designed for multi-horizon time series forecasting. It excels at combining static metadata, known future inputs, and observed historical time series (market + social data) to predict manipulation risk.

### 4. Inference & Alerting Engine (`src/inference/`)
Processes real-time data streams against the trained model to generate actionable insights.
- Evaluates risk levels (Low, Medium, High, Critical).
- Triggers alerts when manipulation probabilities exceed defined thresholds.

### 5. REST API & WebSocket Server (`api/`)
A Flask-based backend that serves the frontend and manages communication.
- **REST Endpoints**: Provides historical analysis, stock metadata, and manual prediction triggers.
- **WebSockets**: Streams real-time alerts and risk updates directly to connected clients.

### 6. Frontend Dashboard (`frontend/`)
A responsive web interface (HTML/JS/React) that provides:
- Real-time stock cards with risk indicators.
- Live alert feeds.
- Interactive charts and technical analysis views.

### 7. Database Storage (`data/` & MongoDB)
- **MongoDB**: Primary data store for historical market data, social media posts, and generated alerts.
- **Redis (Optional)**: Used for caching and message brokering.
- **Local File System**: Stores raw/processed datasets (`data/`) and trained model weights (`models/`).

## Data Flow

1. Market and Social data are ingested into MongoDB.
2. The Preprocessing engine cleans the data and generates features/labels.
3. The Model trains on the processed data and saves its weights.
4. The Inference engine continuously reads live data, consults the Model, and generates Risk Scores.
5. The API serves these Risk Scores to the Frontend Dashboard via REST and WebSockets.
