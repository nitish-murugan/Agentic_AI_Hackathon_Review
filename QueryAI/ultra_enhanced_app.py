"""
Ultra-Enhanced ChatGPT-style Flask Application
High-performance warehouse management with advanced NLP and request queuing
"""

from flask import Flask, render_template, request, jsonify, Response, session
import uuid
import time
import json
import threading
from datetime import datetime
from ultra_advanced_nlp_engine import get_nlp_engine
import queue
import logging

app = Flask(__name__)
app.secret_key = 'ultra-advanced-warehouse-nlp-key-2024'

# Configure logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Request queue management
request_queue = queue.Queue()
processing_status = {}
request_lock = threading.Lock()

# Initialize NLP engine
nlp_engine = None

def initialize_nlp_engine():
    """Initialize NLP engine with error handling"""
    global nlp_engine
    try:
        nlp_engine = get_nlp_engine()
        logger.info("✅ Ultra-Advanced NLP Engine initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize NLP engine: {e}")
        nlp_engine = None

def process_requests():
    """Background thread to process queued requests"""
    while True:
        try:
            if not request_queue.empty():
                request_data = request_queue.get(timeout=1)
                request_id = request_data['id']
                query = request_data['query']
                
                logger.info(f"Processing request {request_id}: {query[:50]}...")
                
                with request_lock:
                    processing_status[request_id] = {'status': 'processing', 'response': ''}
                
                # Process with NLP engine
                if nlp_engine:
                    try:
                        response_parts = []
                        for chunk in nlp_engine.process_query(query):
                            if chunk:  # Only process non-empty chunks
                                response_parts.append(chunk)
                                with request_lock:
                                    processing_status[request_id]['response'] = ''.join(response_parts)
                        
                        with request_lock:
                            processing_status[request_id]['status'] = 'completed'
                            
                        logger.info(f"Completed request {request_id}")
                        
                    except Exception as e:
                        logger.error(f"Error processing request {request_id}: {e}")
                        with request_lock:
                            processing_status[request_id] = {
                                'status': 'completed',
                                'response': f'❌ Error processing query: {str(e)}'
                            }
                else:
                    with request_lock:
                        processing_status[request_id] = {
                            'status': 'completed',
                            'response': '❌ NLP engine not available. Please restart the application.'
                        }
                
                request_queue.task_done()
            else:
                time.sleep(0.1)  # Short sleep when queue is empty
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Error in request processing thread: {e}")
            time.sleep(1)

# Start background processing thread
processing_thread = threading.Thread(target=process_requests, daemon=True)
processing_thread.start()

@app.route('/')
def index():
    """Render the ultra-enhanced ChatGPT interface"""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return render_template('ultra_advanced_chatgpt.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Handle chat requests with enhanced queuing and performance"""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        # Don't check nlp_engine.is_processing here - use our own queue system
        logger.info(f"Received chat request: {user_message[:50]}...")
        
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        
        # Add to queue
        request_queue.put({
            'id': request_id,
            'query': user_message,
            'timestamp': datetime.now()
        })
        
        with request_lock:
            processing_status[request_id] = {'status': 'queued', 'response': ''}
        
        logger.info(f"Queued request {request_id}")
        
        return jsonify({
            'request_id': request_id,
            'queued': True,
            'message': 'Request queued for processing'
        })
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/stream/<request_id>')
def stream_response(request_id):
    """Stream response for a specific request with optimized performance"""
    def generate():
        logger.info(f"Starting stream for request {request_id}")
        last_response = ''
        max_wait_time = 30  # 30 seconds timeout
        start_time = time.time()
        
        # Set proper headers for Server-Sent Events
        yield "retry: 1000\n\n"
        
        while time.time() - start_time < max_wait_time:
            try:
                with request_lock:
                    if request_id in processing_status:
                        status_data = processing_status[request_id]
                        current_response = status_data['response']
                        status = status_data['status']
                        
                        # Send new content
                        if current_response != last_response:
                            new_content = current_response[len(last_response):]
                            if new_content:
                                yield f"data: {json.dumps({'type': 'content', 'data': new_content})}\n\n"
                                last_response = current_response
                        
                        # Check if completed
                        if status == 'completed':
                            yield f"data: {json.dumps({'type': 'done'})}\n\n"
                            logger.info(f"Stream completed for request {request_id}")
                            # Clean up
                            del processing_status[request_id]
                            return
                    else:
                        yield f"data: {json.dumps({'type': 'error', 'message': 'Request not found'})}\n\n"
                        logger.warning(f"Request {request_id} not found in processing status")
                        return
                
                time.sleep(0.05)  # Faster polling for better responsiveness
                
            except Exception as e:
                logger.error(f"Error in stream generation: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
                return
        
        # Timeout
        logger.warning(f"Stream timeout for request {request_id}")
        yield f"data: {json.dumps({'type': 'timeout'})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream', headers={
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
    })

@app.route('/api/health')
def health_check():
    """Enhanced health check with detailed status"""
    engine_status = 'healthy' if nlp_engine else 'unavailable'
    queue_size = request_queue.qsize()
    active_requests = len(processing_status)
    
    return jsonify({
        'status': 'healthy',
        'nlp_engine': engine_status,
        'queue_size': queue_size,
        'active_requests': active_requests,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analytics')
def analytics():
    """Provide system analytics and performance metrics"""
    if not nlp_engine:
        return jsonify({'error': 'NLP engine not available'}), 503
    
    try:
        # Get data statistics
        stats = {
            'total_products': len(nlp_engine.inventory_df),
            'total_cities': len(nlp_engine.city_lookup),
            'total_sales_records': len(nlp_engine.sales_df),
            'low_stock_items': len(nlp_engine.low_stock_items),
            'cache_size': len(nlp_engine.response_cache),
            'system_status': 'optimal'
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Initializing Ultra-Enhanced Warehouse AI Assistant...")
    initialize_nlp_engine()
    print("🌟 Starting Flask application on http://localhost:5002")
    print("💡 Features: Advanced NLP, Request Queuing, Optimized Performance")
    app.run(debug=True, host='0.0.0.0', port=5002, threaded=True)