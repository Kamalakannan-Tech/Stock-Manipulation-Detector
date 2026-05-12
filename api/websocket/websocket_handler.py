"""
WebSocket handler for real-time updates.
"""
from flask import current_app
from flask_socketio import emit, join_room, leave_room
import logging

logger = logging.getLogger(__name__)

def init_socketio(socketio):
    """Initialize WebSocket event handlers."""
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        logger.info("Client connected")
        emit('connection_response', {'status': 'connected'})
    
    @socketio.on('disconnect')
    def handle_disconnect():
        """Handle client disconnection."""
        logger.info("Client disconnected")
    
    @socketio.on('subscribe')
    def handle_subscribe(data):
        """Subscribe to updates for specific tickers."""
        ticker = data.get('ticker', '').upper()
        
        if ticker:
            join_room(f'ticker_{ticker}')
            logger.info(f"Client subscribed to {ticker}")
            emit('subscription_response', {
                'ticker': ticker,
                'status': 'subscribed'
            })
            
            # Send latest data
            predictor = current_app.config.get('PREDICTOR')
            if predictor:
                prediction = predictor.get_latest_prediction(ticker)
                if prediction:
                    emit('prediction_update', {
                        'ticker': ticker,
                        'prediction': {
                            'risk_level': prediction.get('risk_level'),
                            'risk_score': prediction.get('risk_score'),
                            'manipulation_probability': prediction.get('manipulation_probability')
                        }
                    })
    
    @socketio.on('unsubscribe')
    def handle_unsubscribe(data):
        """Unsubscribe from ticker updates."""
        ticker = data.get('ticker', '').upper()
        
        if ticker:
            leave_room(f'ticker_{ticker}')
            logger.info(f"Client unsubscribed from {ticker}")
            emit('subscription_response', {
                'ticker': ticker,
                'status': 'unsubscribed'
            })
    
    @socketio.on('request_update')
    def handle_update_request(data):
        """Handle request for immediate update."""
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            emit('error', {'message': 'Ticker required'})
            return
        
        predictor = current_app.config.get('PREDICTOR')
        fusion = current_app.config.get('FUSION_SERVICE')
        
        if not predictor or not fusion:
            emit('error', {'message': 'Service unavailable'})
            return
        
        # Get latest data
        summary = fusion.get_latest_data_summary(ticker)
        prediction = predictor.get_latest_prediction(ticker)
        
        emit('update_response', {
            'ticker': ticker,
            'market_data': {
                'available': summary.get('market_data_available', False),
                'latest_price': summary.get('latest_price')
            },
            'social_data': {
                'available': summary.get('social_data_available', False),
                'sentiment': summary.get('social_sentiment')
            },
            'prediction': {
                'risk_level': prediction.get('risk_level') if prediction else 'unknown',
                'risk_score': prediction.get('risk_score', 0) if prediction else 0
            }
        })


