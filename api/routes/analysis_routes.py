"""
Analysis routes — serves evaluation metrics and plots for the dashboard.
"""
from flask import Blueprint, jsonify, send_file, current_app
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
bp = Blueprint('analysis', __name__)

# Absolute paths — anchored at the project root (two levels up from api/routes/)
_ROOT     = Path(__file__).resolve().parent.parent.parent
EVAL_DIR  = _ROOT / 'data' / 'evaluation'
METRICS_FILE = EVAL_DIR / 'metrics.json'


def _load_metrics() -> dict:
    if not METRICS_FILE.exists():
        return {}
    try:
        with open(METRICS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


@bp.route('/analysis/metrics', methods=['GET'])
def get_metrics():
    """GET /api/analysis/metrics — model evaluation metrics."""
    metrics = _load_metrics()
    if not metrics:
        return jsonify({
            'available': False,
            'message': 'No metrics yet. Run scripts/05_evaluate_model.py after training.',
        })
    # Strip non-serializable fields (plot paths stay as strings)
    return jsonify({'available': True, 'metrics': metrics,
                    'timestamp': datetime.now().isoformat()})


@bp.route('/analysis/plots', methods=['GET'])
def list_plots():
    """GET /api/analysis/plots — list available evaluation plot files."""
    if not EVAL_DIR.exists():
        return jsonify({'plots': []})
    plots = []
    for f in EVAL_DIR.glob('*.png'):
        plots.append({
            'name':  f.stem,
            'file':  f.name,
            'url':   f'/api/analysis/plot/{f.name}',
            'size':  f.stat().st_size,
        })
    return jsonify({'plots': plots})


@bp.route('/analysis/plot/<string:filename>', methods=['GET'])
def serve_plot(filename):
    """GET /api/analysis/plot/<file> — serve a PNG evaluation plot."""
    # Sanitise filename
    safe = Path(filename).name
    path = EVAL_DIR / safe
    if not path.exists() or path.suffix != '.png':
        return jsonify({'error': 'Plot not found'}), 404
    return send_file(str(path), mimetype='image/png')


@bp.route('/analysis/history', methods=['GET'])
def training_history():
    """GET /api/analysis/history — training loss / metric curves."""
    hist_path = _ROOT / 'models' / 'saved_models' / 'training_history.json'
    if not hist_path.exists():
        return jsonify({'available': False, 'history': {}})
    try:
        with open(hist_path) as f:
            history = json.load(f)
        return jsonify({'available': True, 'history': history})
    except Exception as e:
        return jsonify({'available': False, 'error': str(e)})
