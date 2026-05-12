# Stock Manipulation Detection System

AI-powered system for detecting stock price manipulation using market-social data fusion.

## Features

- Real-time social media monitoring (Twitter, Reddit, StockTwits)
- Market data analysis with technical indicators
- Temporal Fusion Transformer for manipulation detection
- Sector decoupling to filter market-wide movements
- Web dashboard for monitoring and alerts
- RESTful API and WebSocket support

## Quick Start
```bash
# 1. Setup environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env
# Edit .env with your credentials

# 3. Initialize database
python scripts/01_setup_database.py

# 4. Download models
python scripts/utils/download_models.py

# 5. Collect data
python scripts/02_collect_historical_data.py

# 6. Preprocess data
python scripts/03_preprocess_data.py

# 7. Train model
python scripts/04_train_model.py

# 8. Start application
docker-compose up --build
```

## Project Structure
```
stock-manipulation-detector/
├── config/              # Configuration files
├── data/                # Data storage
├── models/              # Trained models
├── src/                 # Source code
├── api/                 # Flask API
├── frontend/            # React dashboard
├── scripts/             # Execution scripts
└── tests/               # Unit tests
```

## Documentation

- Architecture: `docs/architecture.md`
- API Documentation: `docs/api_documentation.md`
- Setup Guide: `docs/setup_guide.md`

## License

MIT License
