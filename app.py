"""
Flask Web Application for Quick Commerce Demand Prediction
"""
from flask import Flask, render_template, request, jsonify, redirect, url_for
from flask_cors import CORS
import json
import plotly.graph_objs as go
import plotly.utils
from datetime import datetime, timedelta
from prediction_api import DemandPredictionAPI
from database import QuickCommerceDB
import pandas as pd

app = Flask(__name__)
CORS(app)

# Initialize the prediction API
prediction_api = DemandPredictionAPI()
db = QuickCommerceDB()

@app.route('/')
def index():
    """Home page with dashboard overview"""
    try:
        # Get summary statistics
        city_performance = db.get_city_performance(days=7)
        top_products = db.get_top_selling_products(days=7, limit=5)
        low_stock = db.get_low_stock_products(threshold=10)
        active_alerts = db.get_active_alerts()
        
        # Calculate KPIs
        total_sales_7d = city_performance['total_units'].sum() if not city_performance.empty else 0
        total_gmv_7d = city_performance['total_gmv'].sum() if not city_performance.empty else 0
        total_cities = len(city_performance) if not city_performance.empty else 0
        total_alerts = len(active_alerts) if not active_alerts.empty else 0
        
        # Create charts data
        city_chart = create_city_performance_chart(city_performance.head(10))
        product_chart = create_product_performance_chart(top_products)
        
        return render_template('dashboard.html',
                             total_sales_7d=total_sales_7d,
                             total_gmv_7d=round(total_gmv_7d, 2),
                             total_cities=total_cities,
                             total_alerts=total_alerts,
                             city_chart=city_chart,
                             product_chart=product_chart,
                             top_products=top_products.to_dict('records') if not top_products.empty else [],
                             recent_alerts=active_alerts.head(5).to_dict('records') if not active_alerts.empty else [])
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/predict')
def predict_page():
    """Prediction interface page"""
    try:
        # Get available cities and products from API to ensure consistency
        cities = prediction_api.get_available_cities()
        products = prediction_api.get_available_products()
        
        return render_template('predict.html', 
                             cities=cities,
                             products=products)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for demand prediction"""
    try:
        data = request.get_json()
        product_id = int(data.get('product_id'))
        city_name = data.get('city_name')
        days_ahead = int(data.get('days_ahead', 1))
        stock_quantity = int(data.get('stock_quantity', 0)) if data.get('stock_quantity') else None
        
        result = prediction_api.predict_product_demand(
            product_id=product_id,
            city_name=city_name,
            days_ahead=days_ahead,
            stock_quantity=stock_quantity
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/city_predict', methods=['POST'])
def api_city_predict():
    """API endpoint for city-wide demand prediction"""
    try:
        data = request.get_json()
        city_name = data.get('city_name')
        days_ahead = int(data.get('days_ahead', 1))
        top_products = int(data.get('top_products', 10))
        
        result = prediction_api.predict_city_demand(
            city_name=city_name,
            days_ahead=days_ahead,
            top_products=top_products
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/allocation')
def allocation_page():
    """Inventory allocation page"""
    try:
        # Get products from API to ensure consistency
        products = prediction_api.get_available_products()
        
        return render_template('allocation.html', 
                             products=products)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/allocation', methods=['POST'])
def api_allocation():
    """API endpoint for inventory allocation recommendations"""
    try:
        data = request.get_json()
        product_id = int(data.get('product_id'))
        total_units = int(data.get('total_units'))
        
        result = prediction_api.get_inventory_allocation_recommendations(
            total_units=total_units,
            product_id=product_id
        )
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/analytics')
def analytics():
    """Analytics dashboard page with optimized performance"""
    try:
        # Get underperforming cities (limited analysis period for faster loading)
        underperforming = prediction_api.get_underperforming_cities(days=7)  # Reduced from 30 to 7 days
        
        # Get restock alerts (limited to top 20 for faster loading)
        restock_alerts = prediction_api.get_restock_alerts(days_threshold=7, limit=20)
        
        return render_template('analytics.html',
                             underperforming=underperforming,
                             restock_alerts=restock_alerts)
    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/api/underperforming')
def api_underperforming():
    """API endpoint for underperforming cities analysis"""
    try:
        days = int(request.args.get('days', 7))  # Default to 7 days for faster performance
        result = prediction_api.get_underperforming_cities(days=days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/restock_alerts')
def api_restock_alerts():
    """API endpoint for restock alerts"""
    try:
        days_threshold = int(request.args.get('days_threshold', 7))
        limit = int(request.args.get('limit', 20))  # Add limit parameter for performance
        result = prediction_api.get_restock_alerts(days_threshold=days_threshold, limit=limit)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)})

def create_city_performance_chart(city_data):
    """Create city performance chart"""
    if city_data.empty:
        return json.dumps({})
    
    fig = go.Figure(data=[
        go.Bar(
            x=city_data['city_name'],
            y=city_data['total_units'],
            name='Units Sold',
            marker_color='steelblue'
        )
    ])
    
    fig.update_layout(
        title='Top Cities by Sales (Last 7 Days)',
        xaxis_title='City',
        yaxis_title='Units Sold',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_product_performance_chart(product_data):
    """Create product performance chart"""
    if product_data.empty:
        return json.dumps({})
    
    fig = go.Figure(data=[
        go.Bar(
            x=product_data['product_name'].str[:30] + '...',  # Truncate long names
            y=product_data['total_sales'],
            name='Sales',
            marker_color='lightgreen'
        )
    ])
    
    fig.update_layout(
        title='Top Products by Sales (Last 7 Days)',
        xaxis_title='Product',
        yaxis_title='Units Sold',
        height=400,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

@app.route('/api/cities')
def api_get_cities():
    """API endpoint to get available cities"""
    try:
        cities = prediction_api.get_available_cities()
        return jsonify({"cities": cities})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/api/products')
def api_get_products():
    """API endpoint to get available products"""
    try:
        products = prediction_api.get_available_products()
        return jsonify({"products": products})
    except Exception as e:
        return jsonify({"error": str(e)})

@app.errorhandler(404)
def page_not_found(e):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_server_error(e):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    print("Starting Quick Commerce Demand Prediction Web Application...")
    print("Dashboard: http://localhost:5000")
    print("Predictions: http://localhost:5000/predict")
    print("Allocation: http://localhost:5000/allocation")
    print("Analytics: http://localhost:5000/analytics")
    app.run(debug=True, host='0.0.0.0', port=5000)