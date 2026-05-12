"""
Alert routes — reads/writes from data/alerts.json (no MongoDB required).
"""
from flask import Blueprint, jsonify, request
import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)
bp = Blueprint('alerts', __name__)

# Absolute path — project root is two levels up from api/routes/
_ROOT        = Path(__file__).resolve().parent.parent.parent
ALERTS_FILE  = _ROOT / 'data' / 'alerts.json'


def _load_alerts() -> list:
    if not ALERTS_FILE.exists():
        return []
    try:
        with open(ALERTS_FILE) as f:
            return json.load(f)
    except Exception:
        return []


def _save_alerts(alerts: list):
    ALERTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(ALERTS_FILE, 'w') as f:
        json.dump(alerts[-500:], f, indent=2, default=str)   # keep last 500


@bp.route('/alerts', methods=['GET'])
def get_alerts():
    """GET /api/alerts — recent alerts, newest first."""
    limit  = request.args.get('limit', 50, type=int)
    ticker = request.args.get('ticker', '').upper().strip()
    alerts = _load_alerts()
    if ticker:
        alerts = [a for a in alerts if a.get('ticker') == ticker]
    alerts = list(reversed(alerts))[:limit]
    return jsonify({'alerts': alerts, 'count': len(alerts),
                    'timestamp': datetime.now().isoformat()})


@bp.route('/alerts', methods=['POST'])
def create_alert():
    """POST /api/alerts — create a new alert."""
    data = request.get_json(silent=True) or {}
    required = ('ticker', 'severity', 'message')
    if not all(k in data for k in required):
        return jsonify({'error': f'Missing required fields: {required}'}), 400

    alert = {
        'id':        f"alert_{datetime.now().strftime('%Y%m%d%H%M%S%f')}",
        'ticker':    data['ticker'].upper(),
        'severity':  data['severity'],
        'message':   data['message'],
        'probability': data.get('probability', 0),
        'timestamp': datetime.now().isoformat(),
    }
    alerts = _load_alerts()
    alerts.append(alert)
    _save_alerts(alerts)

    # Emit via WebSocket if available
    try:
        from flask import current_app
        socketio = current_app.config.get('SOCKETIO')
        if socketio:
            socketio.emit('new_alert', alert)
    except Exception:
        pass

    return jsonify({'alert': alert}), 201


@bp.route('/alerts/<string:alert_id>', methods=['DELETE'])
def delete_alert(alert_id):
    """DELETE /api/alerts/<id>."""
    alerts = _load_alerts()
    new_alerts = [a for a in alerts if a.get('id') != alert_id]
    if len(new_alerts) == len(alerts):
        return jsonify({'error': 'Alert not found'}), 404
    _save_alerts(new_alerts)
    return jsonify({'deleted': alert_id})
