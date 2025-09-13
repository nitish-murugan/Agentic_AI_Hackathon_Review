"""
Enhanced ChatGPT-Style Flask Application with Advanced NLP
Real-time warehouse intelligence with sophisticated query understanding
"""

from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS
import json
import uuid
from datetime import datetime
import time
from simplified_nlp_engine import get_nlp_engine
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Global conversation storage
conversations = {}

@app.route('/')
def index():
    """Serve the enhanced ChatGPT-style interface"""
    return render_template('advanced_chatgpt.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Enhanced chat endpoint with advanced NLP processing"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        session_id = data.get('session_id', str(uuid.uuid4()))
        
        if not user_message:
            return jsonify({
                'success': False,
                'response': 'Please provide a message.',
                'session_id': session_id
            })
        
        # Initialize conversation if new session
        if session_id not in conversations:
            conversations[session_id] = []
        
        # Process with advanced NLP engine
        nlp_engine = get_nlp_engine()
        result = nlp_engine.process_advanced_query(user_message, session_id)
        
        # Store conversation
        conversations[session_id].append({
            'user': user_message,
            'assistant': result['response'],
            'timestamp': datetime.now().isoformat(),
            'intent': result.get('intent', 'unknown'),
            'entities': result.get('entities', {}),
            'confidence': result.get('confidence', 0.0)
        })
        
        logger.info(f"Processed query: {user_message[:50]}... | Intent: {result.get('intent')} | Session: {session_id[:8]}")
        
        return jsonify({
            'success': True,
            'response': result['response'],
            'intent': result.get('intent'),
            'entities': result.get('entities'),
            'confidence': result.get('confidence'),
            'session_id': session_id
        })
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        return jsonify({
            'success': False,
            'response': f'I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question.',
            'session_id': session_id
        })

@app.route('/api/chat/stream', methods=['POST'])
def chat_stream():
    """Streaming chat endpoint for real-time responses"""
    def generate():
        try:
            data = request.get_json()
            user_message = data.get('message', '').strip()
            session_id = data.get('session_id', str(uuid.uuid4()))
            
            if not user_message:
                yield f"data: {json.dumps({'error': 'Please provide a message.'})}\n\n"
                return
            
            # Initialize conversation if new session
            if session_id not in conversations:
                conversations[session_id] = []
            
            # Process with advanced NLP engine
            nlp_engine = get_nlp_engine()
            result = nlp_engine.process_advanced_query(user_message, session_id)
            
            # Stream the response word by word
            response_text = result['response']
            words = response_text.split(' ')
            
            for i, word in enumerate(words):
                chunk_data = {
                    'chunk': word + ' ',
                    'is_complete': i == len(words) - 1,
                    'intent': result.get('intent'),
                    'session_id': session_id
                }
                
                yield f"data: {json.dumps(chunk_data)}\n\n"
                time.sleep(0.05)  # Adjust streaming speed
            
            # Store conversation
            conversations[session_id].append({
                'user': user_message,
                'assistant': result['response'],
                'timestamp': datetime.now().isoformat(),
                'intent': result.get('intent', 'unknown'),
                'entities': result.get('entities', {}),
                'confidence': result.get('confidence', 0.0)
            })
            
        except Exception as e:
            logger.error(f"Error in streaming chat: {str(e)}")
            yield f"data: {json.dumps({'error': f'Error: {str(e)}'})}\n\n"
    
    return Response(generate(), mimetype='text/plain')

@app.route('/api/suggestions', methods=['GET'])
def get_suggestions():
    """Get intelligent query suggestions based on warehouse context"""
    suggestions = [
        {
            'text': 'What are our top selling products?',
            'category': 'Sales Analysis',
            'icon': '📊'
        },
        {
            'text': 'I have 1000 units, which places should I distribute them?',
            'category': 'Distribution',
            'icon': '🚚'
        },
        {
            'text': 'Show me inventory levels across all cities',
            'category': 'Inventory',
            'icon': '📦'
        },
        {
            'text': 'Predict demand for next month',
            'category': 'Forecasting',
            'icon': '🔮'
        },
        {
            'text': 'Which city has the best sales performance?',
            'category': 'Analytics',
            'icon': '🏆'
        },
        {
            'text': 'Show me low stock alerts',
            'category': 'Alerts',
            'icon': '⚠️'
        }
    ]
    
    return jsonify({'suggestions': suggestions})

@app.route('/api/conversation/<session_id>', methods=['GET'])
def get_conversation(session_id):
    """Get conversation history for a session"""
    conversation = conversations.get(session_id, [])
    return jsonify({
        'conversation': conversation,
        'session_id': session_id,
        'message_count': len(conversation)
    })

@app.route('/api/sessions', methods=['GET'])
def get_sessions():
    """Get all active conversation sessions"""
    session_list = []
    for session_id, conversation in conversations.items():
        if conversation:
            last_message = conversation[-1]
            session_list.append({
                'session_id': session_id,
                'last_message': last_message['user'][:50] + '...' if len(last_message['user']) > 50 else last_message['user'],
                'timestamp': last_message['timestamp'],
                'message_count': len(conversation)
            })
    
    # Sort by most recent
    session_list.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'sessions': session_list})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get system analytics and insights"""
    try:
        nlp_engine = get_nlp_engine()
        
        # Calculate basic analytics
        total_sessions = len(conversations)
        total_messages = sum(len(conv) for conv in conversations.values())
        
        # Intent distribution
        intent_counts = {}
        for conversation in conversations.values():
            for message in conversation:
                intent = message.get('intent', 'unknown')
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        # Recent activity
        recent_activity = []
        for session_id, conversation in conversations.items():
            for message in conversation[-5:]:  # Last 5 messages
                recent_activity.append({
                    'session_id': session_id[:8],
                    'query': message['user'][:30] + '...' if len(message['user']) > 30 else message['user'],
                    'intent': message.get('intent', 'unknown'),
                    'timestamp': message['timestamp']
                })
        
        recent_activity.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return jsonify({
            'total_sessions': total_sessions,
            'total_messages': total_messages,
            'intent_distribution': intent_counts,
            'recent_activity': recent_activity[:10],
            'system_status': 'operational',
            'nlp_model': 'Advanced NLP with spaCy' if nlp_engine.nlp else 'Basic NLP (fallback)'
        })
        
    except Exception as e:
        logger.error(f"Error getting analytics: {str(e)}")
        return jsonify({'error': str(e)})

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        nlp_engine = get_nlp_engine()
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'nlp_engine': 'loaded',
            'database': 'connected' if hasattr(nlp_engine, 'sales_df') else 'demo_mode',
            'ml_model': 'loaded' if nlp_engine.ml_model else 'statistical_fallback',
            'active_sessions': len(conversations)
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Enhanced Warehouse AI Assistant - Advanced NLP")
    print("="*60)
    print("🧠 Features:")
    print("   • Advanced NLP with spaCy entity extraction")
    print("   • Intelligent distribution optimization")
    print("   • Comprehensive sales and inventory analytics")
    print("   • Real-time demand forecasting")
    print("   • Smart query understanding")
    print("   • Streaming responses")
    print("="*60)
    print("💬 Chat Interface: http://localhost:5002")
    print("📊 Health Check: http://localhost:5002/api/health")
    print("📈 Analytics: http://localhost:5002/api/analytics")
    print("="*60)
    
    app.run(
        host='0.0.0.0',
        port=5002,
        debug=True,
        threaded=True
    )