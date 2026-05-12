"""
Flask API application — works standalone without MongoDB/Redis.
All data is served from CSV/JSON files in the data/ directory.
"""
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO
import logging, os, sys
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-in-prod')
CORS(app, resources={r'/api/*': {'origins': '*'}})
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')
app.config['SOCKETIO'] = socketio

# Optional: heavy services (safe to fail — CSV fallback is always active)
for service_key, module_path, cls_name, init_kwargs in [
    ('PREDICTOR',     'src.inference.realtime_predictor', 'RealtimePredictor',
     {'model_path': os.getenv('MODEL_PATH', 'models/saved_models/best_model.pth'),
      'device':     os.getenv('DEVICE', 'cpu')}),
]:
    try:
        mod = __import__(module_path, fromlist=[cls_name])
        instance = getattr(mod, cls_name)(**init_kwargs)
        app.config[service_key] = instance
        logger.info(f'{cls_name} initialised')
    except Exception as e:
        logger.warning(f'{cls_name} not available ({e}) — CSV fallback active')

# Register blueprints
from api.routes import stock_routes, alert_routes, analysis_routes
app.register_blueprint(stock_routes.bp,   url_prefix='/api')
app.register_blueprint(alert_routes.bp,   url_prefix='/api')
app.register_blueprint(analysis_routes.bp, url_prefix='/api')

try:
    from api.websocket import websocket_handler
    websocket_handler.init_socketio(socketio)
except Exception as e:
    logger.warning(f'WebSocket handlers not loaded: {e}')


# ── Static dashboard ────────────────────────────────────────────────────────
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'frontend')

@app.route('/')
def dashboard():
    """Serve the dashboard SPA."""
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve any other static asset from the frontend/ directory."""
    return send_from_directory(FRONTEND_DIR, filename)


@app.route('/health', methods=['GET'])
def health():
    import datetime
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.now().isoformat(),
        'tickers': os.getenv('MONITORED_TICKERS', 'GME,AMC,TSLA,AAPL,MSFT,NVDA').split(','),
        'model_available': os.path.exists('models/saved_models/best_model.pth'),
        'data_available':  os.path.exists('data/processed/combined_features.csv'),
    })


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f'Server error: {e}')
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    logger.info(f'Starting on port {port}')
    socketio.run(app, host='0.0.0.0', port=port, debug=debug,
                 allow_unsafe_werkzeug=True)
