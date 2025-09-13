"""
Quick Commerce Demand Prediction API Module
Integrates ML model with database for the Agentic AI system
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from demand_predictor import DemandPredictor
from database import QuickCommerceDB
import json

def convert_to_json_serializable(obj):
    """Convert numpy/pandas types to JSON serializable Python types"""
    if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

class DemandPredictionAPI:
    """
    API module for demand prediction in Quick Commerce system
    """
    
    def __init__(self, model_path='demand_predictor.pkl', db_path='quickcommerce.db'):
        self.predictor = DemandPredictor()
        self.db = QuickCommerceDB(db_path)
        
        # Load trained model
        try:
            self.predictor.load_model(model_path)
            print("Demand prediction model loaded successfully")
        except FileNotFoundError:
            print("Model not found. Please train the model first.")
            self.predictor = None
    
    def get_available_cities(self):
        """
        Get list of cities available in the CSV data
        
        Returns:
            list: Sorted list of available city names
        """
        try:
            sales_data = self.db.get_sales_data()
            if sales_data.empty:
                return []
            return sorted(sales_data['city_name'].unique().tolist())
        except Exception as e:
            print(f"Error getting available cities: {e}")
            return []
    
    def get_available_products(self):
        """
        Get list of products available in the CSV data
        
        Returns:
            list: List of dictionaries with product_id and product_name
        """
        try:
            sales_data = self.db.get_sales_data()
            if sales_data.empty:
                return []
            products = sales_data[['product_id', 'product_name']].drop_duplicates()
            return products.to_dict('records')
        except Exception as e:
            print(f"Error getting available products: {e}")
            return []
    
    def validate_city(self, city_name):
        """
        Validate if a city name exists in the CSV data
        
        Args:
            city_name: City name to validate
            
        Returns:
            bool: True if city exists, False otherwise
        """
        available_cities = self.get_available_cities()
        # Normalize to lowercase for comparison
        return city_name.lower() in [city.lower() for city in available_cities]
    
    def predict_product_demand(self, product_id, city_name, days_ahead=1, stock_quantity=None):
        """
        Predict demand for a specific product in a city
        
        Args:
            product_id: Product ID to predict demand for
            city_name: City name
            days_ahead: Number of days ahead to predict (default: 1)
            stock_quantity: Current stock quantity (if None, fetch from DB)
        
        Returns:
            dict: Prediction results
        """
        
        if not self.predictor or not self.predictor.is_trained:
            return {"error": "Model not available or not trained"}
        
        # Validate city name
        if not self.validate_city(city_name):
            available_cities = self.get_available_cities()
            return {"error": f"City '{city_name}' not found in available cities. Available cities: {available_cities[:10]}{'...' if len(available_cities) > 10 else ''}"}
        
        # Normalize city name to lowercase for database queries
        city_name = city_name.lower()
        
        try:
            # Get current stock if not provided
            if stock_quantity is None:
                inventory_data = self.db.get_inventory_data(city_name=city_name, product_id=product_id)
                if not inventory_data.empty:
                    stock_quantity = inventory_data['stock_quantity'].sum()
                    # Create store breakdown for transparency
                    store_breakdown = []
                    for _, row in inventory_data.iterrows():
                        store_breakdown.append({
                            "store_name": str(row['store_name']),
                            "store_type": str(row['store_type']),
                            "stock_quantity": int(row['stock_quantity'])
                        })
                else:
                    stock_quantity = 0
                    store_breakdown = []
            else:
                store_breakdown = [{"note": "Stock quantity manually provided"}]
            
            # Get historical data for better prediction context
            historical_sales = self.db.get_sales_data(
                city_name=city_name, 
                product_id=product_id,
                start_date=(datetime.now() - timedelta(days=30)).date()
            )
            
            # Calculate recent average for context
            recent_avg = historical_sales['units_sold'].mean() if not historical_sales.empty else 0
            
            # Predict for target date
            target_date = datetime.now() + timedelta(days=days_ahead)
            predicted_demand = self.predictor.predict_demand(
                city_name=city_name,
                product_id=product_id,
                date=target_date,
                stock_quantity=stock_quantity,
                db_handler=self.db
            )
            
            # Store prediction in database
            self.db.add_prediction(
                product_id=product_id,
                city_name=city_name,
                prediction_date=target_date.date(),
                predicted_demand=predicted_demand
            )
            
            # Calculate stock sufficiency
            stock_days = stock_quantity / predicted_demand if predicted_demand > 0 else float('inf')
            
            result = {
                "product_id": int(product_id),  # Convert to native Python int
                "city_name": str(city_name),    # Ensure string type
                "prediction_date": target_date.strftime('%Y-%m-%d'),
                "predicted_demand": int(predicted_demand),  # Convert to native Python int
                "current_stock": int(stock_quantity),       # Convert to native Python int
                "stock_days_remaining": float(round(stock_days, 1)),  # Convert to native Python float
                "recent_avg_sales": float(round(recent_avg, 1)),      # Convert to native Python float
                "store_breakdown": store_breakdown,  # Show individual store stocks
                "status": "success"
            }
            
            # Generate alerts if needed
            if stock_days < 3:
                alert_msg = f"Critical: Only {stock_days:.1f} days of stock remaining for product {product_id} in {city_name}"
                self.db.add_alert('CRITICAL_STOCK', product_id, city_name, alert_msg, 'HIGH')
                result["alert"] = alert_msg
            elif stock_days < 7:
                alert_msg = f"Warning: Only {stock_days:.1f} days of stock remaining for product {product_id} in {city_name}"
                self.db.add_alert('LOW_STOCK', product_id, city_name, alert_msg, 'MEDIUM')
                result["alert"] = alert_msg
            
            return result
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def predict_city_demand(self, city_name, days_ahead=1, top_products=10):
        """
        Predict demand for top products in a city
        
        Args:
            city_name: City name
            days_ahead: Number of days ahead to predict
            top_products: Number of top products to analyze
        
        Returns:
            dict: City-wide demand predictions
        """
        
        # Validate city name
        if not self.validate_city(city_name):
            available_cities = self.get_available_cities()
            return {"error": f"City '{city_name}' not found in available cities. Available cities: {available_cities[:10]}{'...' if len(available_cities) > 10 else ''}"}
        
        # Normalize city name to lowercase for database queries
        city_name = city_name.lower()
        
        try:
            # Get top selling products in the city
            top_products_data = self.db.get_top_selling_products(
                city_name=city_name, 
                days=30, 
                limit=top_products
            )
            
            if top_products_data.empty:
                return {"error": f"No sales data found for city: {city_name}"}
            
            predictions = []
            total_predicted_demand = 0
            
            for _, row in top_products_data.iterrows():
                prediction = self.predict_product_demand(
                    product_id=row['product_id'],
                    city_name=city_name,
                    days_ahead=days_ahead
                )
                
                if "error" not in prediction:
                    predictions.append(prediction)
                    total_predicted_demand += prediction['predicted_demand']
            
            # City performance analysis
            city_perf = self.db.get_city_performance(days=7)
            city_data = city_perf[city_perf['city_name'] == city_name]
            
            result = {
                "city_name": str(city_name),  # Ensure string type
                "prediction_date": (datetime.now() + timedelta(days=days_ahead)).strftime('%Y-%m-%d'),
                "total_predicted_demand": int(total_predicted_demand),  # Convert to native Python int
                "product_predictions": predictions,
                "status": "success"
            }
            
            if not city_data.empty:
                result["recent_performance"] = {
                    "total_units_7days": int(city_data.iloc[0]['total_units']),
                    "total_gmv_7days": float(city_data.iloc[0]['total_gmv']),
                    "unique_products": int(city_data.iloc[0]['unique_products'])
                }
            
            return result
            
        except Exception as e:
            return {"error": f"City prediction failed: {str(e)}"}
    
    def get_inventory_allocation_recommendations(self, total_units, product_id):
        """
        Get recommendations for allocating inventory across cities
        
        Args:
            total_units: Total units to allocate
            product_id: Product ID
        
        Returns:
            dict: Allocation recommendations
        """
        
        try:
            # Get recent sales data for the product across all cities
            recent_sales = self.db.get_sales_data(
                product_id=product_id,
                start_date=(datetime.now() - timedelta(days=7)).date()
            )
            
            if recent_sales.empty:
                return {"error": f"No recent sales data found for product {product_id}"}
            
            # Calculate city-wise sales percentages
            city_sales = recent_sales.groupby('city_name')['units_sold'].sum().sort_values(ascending=False)
            total_sales = city_sales.sum()
            
            if total_sales == 0:
                return {"error": "No sales found for this product"}
            
            # Calculate allocation based on sales percentages
            allocations = []
            remaining_units = total_units
            
            for city, sales in city_sales.items():
                percentage = sales / total_sales
                allocated_units = int(total_units * percentage)
                
                if remaining_units >= allocated_units:
                    remaining_units -= allocated_units
                else:
                    allocated_units = remaining_units
                    remaining_units = 0
                
                # Get current stock
                current_stock = self.db.get_inventory_data(city_name=city, product_id=product_id)
                current_stock_qty = current_stock['stock_quantity'].sum() if not current_stock.empty else 0
                
                # Predict demand for validation
                prediction = self.predict_product_demand(product_id, city, days_ahead=1)
                predicted_demand = prediction.get('predicted_demand', 0) if 'error' not in prediction else 0
                
                allocations.append({
                    "city_name": str(city),  # Ensure string type
                    "allocated_units": int(allocated_units),  # Convert to native Python int
                    "sales_percentage": float(round(percentage * 100, 1)),  # Convert to native Python float
                    "recent_7day_sales": int(sales),  # Convert to native Python int
                    "current_stock": int(current_stock_qty),  # Convert to native Python int
                    "predicted_demand_tomorrow": int(predicted_demand),  # Convert to native Python int
                    "priority": "HIGH" if percentage > 0.2 else "MEDIUM" if percentage > 0.1 else "LOW"
                })
            
            # Distribute remaining units to top cities
            if remaining_units > 0:
                top_cities = sorted(allocations, key=lambda x: x['sales_percentage'], reverse=True)
                for i, allocation in enumerate(top_cities):
                    if remaining_units <= 0:
                        break
                    allocation['allocated_units'] += 1
                    remaining_units -= 1
            
            result = {
                "product_id": int(product_id),  # Convert to native Python int
                "total_units_to_allocate": int(total_units),  # Convert to native Python int
                "allocation_strategy": "based_on_7day_sales_percentage",
                "allocations": allocations,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Allocation recommendation failed: {str(e)}"}
    
    def get_underperforming_cities(self, days=7, min_threshold=100):
        """
        Identify underperforming cities that need attention
        
        Args:
            days: Number of days to analyze
            min_threshold: Minimum sales threshold
        
        Returns:
            dict: Underperforming cities analysis
        """
        
        try:
            city_performance = self.db.get_city_performance(days=days)
            
            if city_performance.empty:
                return {"error": "No performance data available"}
            
            # Calculate performance metrics
            avg_units = city_performance['total_units'].mean()
            avg_gmv = city_performance['total_gmv'].mean()
            
            underperforming = []
            
            for _, city in city_performance.iterrows():
                if city['total_units'] < min_threshold or city['total_units'] < avg_units * 0.5:
                    underperforming.append({
                        "city_name": city['city_name'],
                        "total_units": int(city['total_units']),
                        "total_gmv": float(city['total_gmv']),
                        "unique_products": int(city['unique_products']),
                        "performance_vs_avg": round((city['total_units'] / avg_units) * 100, 1),
                        "recommendation": self._get_city_recommendation(city)
                    })
            
            result = {
                "analysis_period_days": days,
                "total_cities_analyzed": len(city_performance),
                "underperforming_cities_count": len(underperforming),
                "average_units_sold": round(avg_units, 1),
                "underperforming_cities": underperforming,
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Underperforming cities analysis failed: {str(e)}"}
    
    def _get_city_recommendation(self, city_data):
        """Generate recommendation for underperforming city"""
        
        if city_data['unique_products'] < 5:
            return "Increase product variety to improve sales"
        elif city_data['total_units'] < 50:
            return "Focus on marketing and promotional activities"
        else:
            return "Analyze local demand patterns and optimize inventory"
    
    def get_restock_alerts(self, days_threshold=7, limit=50):
        """
        Get products that need restocking based on predicted demand
        
        Args:
            days_threshold: Alert when stock will last less than this many days
            limit: Maximum number of alerts to generate (for performance)
        
        Returns:
            dict: Restock alerts
        """
        
        try:
            # Get all unique product-city combinations with low stock first
            inventory_data = self.db.get_inventory_data()
            
            if inventory_data.empty:
                return {"error": "No inventory data available"}
            
            restock_alerts = []
            processed_count = 0
            
            # Group by product and city, prioritize low stock items
            grouped = inventory_data.groupby(['product_id', 'city_name'])
            
            # Sort groups by total stock to prioritize low stock items
            sorted_groups = sorted(grouped, key=lambda x: x[1]['stock_quantity'].sum())
            
            for (product_id, city_name), group in sorted_groups:
                if processed_count >= limit:  # Limit processing for performance
                    break
                    
                current_stock = group['stock_quantity'].sum()
                
                if pd.isna(product_id) or current_stock > 100:  # Skip high stock items
                    continue
                
                processed_count += 1
                
                # Predict demand
                prediction = self.predict_product_demand(
                    product_id=int(product_id),
                    city_name=city_name,
                    days_ahead=1,
                    stock_quantity=current_stock
                )
                
                if "error" not in prediction:
                    stock_days = prediction.get('stock_days_remaining', float('inf'))
                    
                    if stock_days < days_threshold:
                        severity = "CRITICAL" if stock_days < 3 else "HIGH" if stock_days < 5 else "MEDIUM"
                        
                        restock_alerts.append({
                            "product_id": int(product_id),
                            "city_name": city_name,
                            "current_stock": int(current_stock),
                            "predicted_daily_demand": int(prediction['predicted_demand']),
                            "days_remaining": float(stock_days),
                            "severity": severity,
                            "recommended_restock": int(max(
                                prediction['predicted_demand'] * 14, 
                                prediction['predicted_demand'] * 7 - current_stock
                            ))
                        })
            
            # Sort by urgency
            restock_alerts.sort(key=lambda x: x['days_remaining'])
            
            result = {
                "restock_alerts": restock_alerts[:20],  # Top 20 most urgent
                "total_alerts": len(restock_alerts),
                "critical_count": len([a for a in restock_alerts if a['severity'] == 'CRITICAL']),
                "status": "success"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"Restock alerts failed: {str(e)}"}

# Example usage and testing
def main():
    """Test the Demand Prediction API"""
    
    print("Quick Commerce Demand Prediction API Test")
    print("=" * 45)
    
    # Initialize API
    api = DemandPredictionAPI()
    
    # Test 1: Predict demand for a specific product
    print("\n1. Testing product demand prediction...")
    result = api.predict_product_demand(product_id=445285, city_name='delhi')
    print(json.dumps(result, indent=2))
    
    # Test 2: Predict city-wide demand
    print("\n2. Testing city-wide demand prediction...")
    result = api.predict_city_demand(city_name='mumbai', top_products=5)
    print(json.dumps(result, indent=2))
    
    # Test 3: Get allocation recommendations
    print("\n3. Testing inventory allocation recommendations...")
    result = api.get_inventory_allocation_recommendations(total_units=1000, product_id=445285)
    print(json.dumps(result, indent=2))
    
    # Test 4: Get underperforming cities
    print("\n4. Testing underperforming cities analysis...")
    result = api.get_underperforming_cities(days=30)
    print(json.dumps(result, indent=2))
    
    # Test 5: Get restock alerts
    print("\n5. Testing restock alerts...")
    result = api.get_restock_alerts(days_threshold=10)
    print(json.dumps(result, indent=2))
    
    print("\nAPI testing completed!")

if __name__ == "__main__":
    main()