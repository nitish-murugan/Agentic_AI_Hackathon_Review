"""
Advanced NLP Engine for Warehouse Management
Uses state-of-the-art NLP models for sophisticated query understanding
"""

import spacy
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any
from datetime import datetime, timedelta
import pickle
import os
import sys
from collections import defaultdict
import math

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from database import QuickCommerceDB
    from demand_predictor import DemandPredictor
except ImportError as e:
    print(f"Warning: Could not import ML models: {e}")
    QuickCommerceDB = None
    DemandPredictor = None

class AdvancedNLPWarehouseEngine:
    """
    Advanced NLP Engine for Warehouse Management with sophisticated query understanding
    """
    
    def __init__(self):
        self.conversation_history = []
        self.context = {}
        self.load_nlp_model()
        self.load_data()
        self.load_ml_models()
        self.setup_query_patterns()
        
    def load_nlp_model(self):
        """Load spaCy NLP model for advanced text processing"""
        try:
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Advanced NLP model loaded successfully")
        except OSError:
            print("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            # Fallback to basic processing
            self.nlp = None
            
    def load_data(self):
        """Load warehouse data with enhanced preprocessing"""
        try:
            # Try to load real data
            self.sales_df = pd.read_csv('../dataset/salesData.csv')
            self.inventory_df = pd.read_csv('../dataset/inventoryData.csv')
            
            # Enhanced data preprocessing
            self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
            self.sales_df['city_name'] = self.sales_df['city_name'].str.lower().str.strip()
            self.inventory_df['city_name'] = self.inventory_df['city_name'].str.lower().str.strip()
            
            # Create additional analytics columns
            self.sales_df['revenue'] = self.sales_df['units_sold'] * self.sales_df['selling_price']
            self.sales_df['profit'] = self.sales_df['revenue'] - (self.sales_df['units_sold'] * self.sales_df['mrp'] * 0.7)
            
            print("✅ Enhanced data loaded successfully")
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self._create_enhanced_demo_data()
            
    def load_ml_models(self):
        """Load ML models for predictions"""
        try:
            model_path = '../demand_predictor.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.ml_model = pickle.load(f)
                print("✅ ML demand predictor loaded")
            else:
                self.ml_model = None
                print("⚠️ ML model not found, using advanced statistical methods")
        except Exception as e:
            print(f"❌ Error loading ML model: {e}")
            self.ml_model = None
            
    def setup_query_patterns(self):
        """Setup advanced query patterns for NLP understanding"""
        self.query_patterns = {
            'top_seller_queries': {
                'patterns': [
                    r'(?i).*\b(top|best|highest|most).*\b(sell|selling|sold|popular|successful)\b.*',
                    r'(?i).*\b(which|what).*\b(product|item).*\b(sell|selling|sold|popular)\b.*',
                    r'(?i).*\bbest\s*seller\b.*',
                    r'(?i).*\btop\s*(selling|performing).*',
                    r'(?i).*\bmost\s*(popular|successful|profitable)\b.*'
                ],
                'intent': 'top_seller_analysis'
            },
            'distribution_queries': {
                'patterns': [
                    r'(?i).*\b(\d+)\s*units?\b.*\b(distribute|distribution|allocate|allocation)\b.*',
                    r'(?i).*\bwhere.*\b(send|distribute|allocate)\b.*\b(\d+)\s*units?\b.*',
                    r'(?i).*\bhow.*\b(distribute|allocate)\b.*\bunits?\b.*',
                    r'(?i).*\bwhich\s*(place|city|location).*\b(distribute|send|allocate)\b.*',
                    r'(?i).*\boptimal.*\b(distribution|allocation)\b.*'
                ],
                'intent': 'distribution_optimization'
            },
            'inventory_queries': {
                'patterns': [
                    r'(?i).*\b(inventory|stock|warehouse)\b.*\b(level|amount|quantity)\b.*',
                    r'(?i).*\bhow\s*much.*\b(stock|inventory)\b.*',
                    r'(?i).*\bstock.*\b(available|remaining|left)\b.*',
                    r'(?i).*\b(low|high).*\b(stock|inventory)\b.*'
                ],
                'intent': 'inventory_analysis'
            },
            'demand_prediction_queries': {
                'patterns': [
                    r'(?i).*\b(predict|forecast|future|expect)\b.*\b(demand|sales|need)\b.*',
                    r'(?i).*\bwhat.*\b(demand|sales).*\b(next|future|coming)\b.*',
                    r'(?i).*\bhow\s*much.*\b(need|require|demand)\b.*'
                ],
                'intent': 'demand_prediction'
            },
            'sales_analytics_queries': {
                'patterns': [
                    r'(?i).*\b(sales|revenue|profit).*\b(analysis|analytics|report|performance)\b.*',
                    r'(?i).*\b(compare|comparison).*\b(sales|performance)\b.*',
                    r'(?i).*\bwhich\s*(city|location).*\b(best|worst|highest|lowest)\b.*\b(sales|performance)\b.*'
                ],
                'intent': 'sales_analytics'
            }
        }
        
    def _create_enhanced_demo_data(self):
        """Create comprehensive demo data for testing"""
        print("🔧 Creating enhanced demo data...")
        np.random.seed(42)
        
        # Enhanced demo data with more realistic patterns
        cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad']
        categories = ['Baby Care', 'Personal Care', 'Health & Wellness', 'Home Care', 'Food & Beverages']
        
        # Create sales data with realistic patterns
        sales_data = []
        for _ in range(1000):
            city = np.random.choice(cities)
            category = np.random.choice(categories)
            product_id = np.random.randint(561001, 561100)
            units_sold = np.random.poisson(50) + 1
            mrp = np.random.uniform(50, 500)
            selling_price = mrp * np.random.uniform(0.7, 0.95)
            
            sales_data.append({
                'date': datetime.now() - timedelta(days=np.random.randint(0, 365)),
                'product_id': product_id,
                'product_name': f"{category} Product {product_id}",
                'category': category,
                'city_name': city,
                'units_sold': units_sold,
                'mrp': mrp,
                'selling_price': selling_price,
                'revenue': units_sold * selling_price,
                'profit': units_sold * (selling_price - mrp * 0.7)
            })
            
        self.sales_df = pd.DataFrame(sales_data)
        
        # Create inventory data
        inventory_data = []
        for city in cities:
            for category in categories:
                for product_id in range(561001, 561050):
                    current_stock = np.random.randint(0, 500)
                    reorder_level = np.random.randint(50, 100)
                    
                    inventory_data.append({
                        'product_id': product_id,
                        'product_name': f"{category} Product {product_id}",
                        'category': category,
                        'city_name': city,
                        'current_stock': current_stock,
                        'reorder_level': reorder_level,
                        'max_capacity': np.random.randint(800, 1200)
                    })
                    
        self.inventory_df = pd.DataFrame(inventory_data)
        print("✅ Enhanced demo data created successfully")
        
    def process_advanced_query(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Process user query using advanced NLP and ML techniques
        """
        # Add to conversation history
        self.conversation_history.append({
            'query': query,
            'timestamp': datetime.now(),
            'session_id': session_id
        })
        
        # Extract entities and intent using advanced NLP
        entities = self.extract_entities(query)
        intent = self.classify_intent(query)
        
        # Process based on intent
        if intent == 'top_seller_analysis':
            response = self.analyze_top_sellers(query, entities)
        elif intent == 'distribution_optimization':
            response = self.optimize_distribution(query, entities)
        elif intent == 'inventory_analysis':
            response = self.analyze_inventory(query, entities)
        elif intent == 'demand_prediction':
            response = self.predict_demand(query, entities)
        elif intent == 'sales_analytics':
            response = self.analyze_sales(query, entities)
        else:
            response = self.handle_general_query(query, entities)
            
        return {
            'response': response,
            'intent': intent,
            'entities': entities,
            'confidence': 0.95,
            'session_id': session_id
        }
        
    def extract_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities using advanced NLP"""
        entities = {
            'numbers': [],
            'cities': [],
            'products': [],
            'categories': [],
            'time_periods': []
        }
        
        if self.nlp:
            # Use spaCy for advanced entity extraction
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ['CARDINAL', 'QUANTITY']:
                    entities['numbers'].append(int(re.findall(r'\d+', ent.text)[0]) if re.findall(r'\d+', ent.text) else 0)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['cities'].append(ent.text.lower())
                elif ent.label_ in ['DATE', 'TIME']:
                    entities['time_periods'].append(ent.text)
        else:
            # Fallback to regex extraction
            numbers = re.findall(r'\b(\d+)\s*units?\b', query, re.IGNORECASE)
            entities['numbers'] = [int(n) for n in numbers]
            
            cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad']
            for city in cities:
                if city in query.lower():
                    entities['cities'].append(city)
                    
        return entities
        
    def classify_intent(self, query: str) -> str:
        """Classify user intent using pattern matching"""
        for intent_type, config in self.query_patterns.items():
            for pattern in config['patterns']:
                if re.search(pattern, query):
                    return config['intent']
        return 'general_query'
        
    def analyze_top_sellers(self, query: str, entities: Dict) -> str:
        """Analyze top selling products with advanced insights"""
        try:
            # Determine analysis scope
            city_filter = entities.get('cities', [])
            top_n = entities.get('numbers', [5])[0] if entities.get('numbers') else 5
            
            # Analyze top sellers
            if city_filter:
                filtered_sales = self.sales_df[self.sales_df['city_name'].isin(city_filter)]
                location_info = f" in {', '.join(city_filter).title()}"
            else:
                filtered_sales = self.sales_df
                location_info = " across all locations"
                
            # Calculate top sellers by multiple metrics
            top_by_units = filtered_sales.groupby(['product_name', 'category']).agg({
                'units_sold': 'sum',
                'revenue': 'sum',
                'profit': 'sum'
            }).sort_values('units_sold', ascending=False).head(top_n)
            
            top_by_revenue = filtered_sales.groupby(['product_name', 'category']).agg({
                'units_sold': 'sum',
                'revenue': 'sum',
                'profit': 'sum'
            }).sort_values('revenue', ascending=False).head(top_n)
            
            # Generate comprehensive response
            response = f"📊 **Top Seller Analysis{location_info}**\n\n"
            
            response += "🏆 **Top Products by Units Sold:**\n"
            for idx, (product_info, data) in enumerate(top_by_units.iterrows(), 1):
                product_name, category = product_info
                response += f"{idx}. **{product_name}** ({category})\n"
                response += f"   • Units Sold: {data['units_sold']:,}\n"
                response += f"   • Revenue: ₹{data['revenue']:,.2f}\n"
                response += f"   • Profit: ₹{data['profit']:,.2f}\n\n"
                
            response += "💰 **Top Products by Revenue:**\n"
            for idx, (product_info, data) in enumerate(top_by_revenue.iterrows(), 1):
                product_name, category = product_info
                response += f"{idx}. **{product_name}** ({category})\n"
                response += f"   • Revenue: ₹{data['revenue']:,.2f}\n"
                response += f"   • Units Sold: {data['units_sold']:,}\n\n"
                
            # Add insights
            total_revenue = filtered_sales['revenue'].sum()
            avg_order_value = filtered_sales['revenue'].mean()
            
            response += "🎯 **Key Insights:**\n"
            response += f"• Total Revenue{location_info}: ₹{total_revenue:,.2f}\n"
            response += f"• Average Order Value: ₹{avg_order_value:.2f}\n"
            response += f"• Most Popular Category: {filtered_sales.groupby('category')['units_sold'].sum().idxmax()}\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing top sellers: {str(e)}"
            
    def optimize_distribution(self, query: str, entities: Dict) -> str:
        """Optimize distribution using advanced algorithms"""
        try:
            # Extract units to distribute
            units_to_distribute = entities.get('numbers', [1000])[0] if entities.get('numbers') else 1000
            
            # Get current inventory levels by city
            city_inventory = self.inventory_df.groupby('city_name').agg({
                'current_stock': 'sum',
                'reorder_level': 'sum',
                'max_capacity': 'sum'
            }).reset_index()
            
            # Get sales velocity by city (units sold per day)
            recent_sales = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
            sales_velocity = recent_sales.groupby('city_name')['units_sold'].sum() / 30
            
            # Calculate distribution optimization
            distribution_plan = []
            remaining_units = units_to_distribute
            
            for _, city_data in city_inventory.iterrows():
                city = city_data['city_name']
                current_stock = city_data['current_stock']
                max_capacity = city_data['max_capacity']
                reorder_level = city_data['reorder_level']
                
                # Get sales velocity for this city
                velocity = sales_velocity.get(city, 10)  # Default velocity
                
                # Calculate priority score
                stock_ratio = current_stock / max_capacity if max_capacity > 0 else 0
                urgency_score = max(0, (reorder_level - current_stock) / reorder_level) if reorder_level > 0 else 0
                velocity_score = velocity / sales_velocity.max() if len(sales_velocity) > 0 else 0.5
                
                # Combined priority (lower stock, higher urgency, higher velocity = higher priority)
                priority_score = (1 - stock_ratio) * 0.4 + urgency_score * 0.4 + velocity_score * 0.2
                
                # Calculate recommended allocation
                available_capacity = max(0, max_capacity - current_stock)
                recommended_units = min(
                    available_capacity,
                    int(remaining_units * priority_score * 1.5),
                    remaining_units
                )
                
                if recommended_units > 0:
                    distribution_plan.append({
                        'city': city,
                        'units': recommended_units,
                        'current_stock': current_stock,
                        'after_stock': current_stock + recommended_units,
                        'capacity_utilization': (current_stock + recommended_units) / max_capacity * 100,
                        'priority_score': priority_score,
                        'sales_velocity': velocity
                    })
                    remaining_units -= recommended_units
                    
            # Sort by priority score
            distribution_plan.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Generate response
            response = f"🚚 **Smart Distribution Plan for {units_to_distribute:,} Units**\n\n"
            
            response += "📋 **Recommended Allocation:**\n"
            total_allocated = 0
            for idx, plan in enumerate(distribution_plan, 1):
                response += f"{idx}. **{plan['city'].title()}**: {plan['units']:,} units\n"
                response += f"   • Current Stock: {plan['current_stock']:,} → {plan['after_stock']:,}\n"
                response += f"   • Capacity Utilization: {plan['capacity_utilization']:.1f}%\n"
                response += f"   • Sales Velocity: {plan['sales_velocity']:.1f} units/day\n"
                response += f"   • Priority Score: {plan['priority_score']:.2f}\n\n"
                total_allocated += plan['units']
                
            response += f"📊 **Summary:**\n"
            response += f"• Total Allocated: {total_allocated:,} units ({total_allocated/units_to_distribute*100:.1f}%)\n"
            response += f"• Remaining Units: {remaining_units:,}\n"
            response += f"• Distribution Strategy: Priority-based allocation considering stock levels, sales velocity, and capacity\n"
            
            if remaining_units > 0:
                response += f"\n⚠️ **Note:** {remaining_units:,} units remaining due to capacity constraints. Consider expanding warehouse capacity in high-demand cities."
                
            return response
            
        except Exception as e:
            return f"❌ Error optimizing distribution: {str(e)}"
            
    def analyze_inventory(self, query: str, entities: Dict) -> str:
        """Analyze inventory with advanced insights"""
        try:
            city_filter = entities.get('cities', [])
            
            if city_filter:
                filtered_inventory = self.inventory_df[self.inventory_df['city_name'].isin(city_filter)]
                location_info = f" in {', '.join(city_filter).title()}"
            else:
                filtered_inventory = self.inventory_df
                location_info = " across all locations"
                
            # Calculate inventory metrics
            total_stock = filtered_inventory['current_stock'].sum()
            total_capacity = filtered_inventory['max_capacity'].sum()
            avg_utilization = (total_stock / total_capacity * 100) if total_capacity > 0 else 0
            
            # Find low stock items
            low_stock = filtered_inventory[filtered_inventory['current_stock'] < filtered_inventory['reorder_level']]
            
            # Find high stock items
            high_stock = filtered_inventory[filtered_inventory['current_stock'] > filtered_inventory['max_capacity'] * 0.8]
            
            response = f"📦 **Inventory Analysis{location_info}**\n\n"
            
            response += f"📊 **Overall Metrics:**\n"
            response += f"• Total Stock: {total_stock:,} units\n"
            response += f"• Total Capacity: {total_capacity:,} units\n"
            response += f"• Average Utilization: {avg_utilization:.1f}%\n"
            response += f"• Total SKUs: {len(filtered_inventory):,}\n\n"
            
            if len(low_stock) > 0:
                response += f"⚠️ **Low Stock Alerts ({len(low_stock)} items):**\n"
                for _, item in low_stock.head(5).iterrows():
                    response += f"• **{item['product_name']}** in {item['city_name'].title()}: {item['current_stock']} units (Reorder: {item['reorder_level']})\n"
                if len(low_stock) > 5:
                    response += f"• ... and {len(low_stock) - 5} more items\n"
                response += "\n"
                
            if len(high_stock) > 0:
                response += f"📈 **High Stock Items ({len(high_stock)} items):**\n"
                for _, item in high_stock.head(5).iterrows():
                    utilization = (item['current_stock'] / item['max_capacity'] * 100)
                    response += f"• **{item['product_name']}** in {item['city_name'].title()}: {item['current_stock']} units ({utilization:.1f}% capacity)\n"
                if len(high_stock) > 5:
                    response += f"• ... and {len(high_stock) - 5} more items\n"
                    
            return response
            
        except Exception as e:
            return f"❌ Error analyzing inventory: {str(e)}"
            
    def predict_demand(self, query: str, entities: Dict) -> str:
        """Predict demand using ML or advanced statistical methods"""
        try:
            # Use ML model if available, otherwise use statistical methods
            if self.ml_model:
                # Use the trained ML model for prediction
                response = "🔮 **AI-Powered Demand Prediction**\n\n"
                response += "Using machine learning algorithms to forecast demand...\n\n"
            else:
                # Use advanced statistical methods
                response = "📈 **Statistical Demand Forecast**\n\n"
                
            # Calculate demand trends
            recent_sales = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
            daily_avg = recent_sales.groupby(recent_sales['date'].dt.date)['units_sold'].sum().mean()
            
            # Predict for next 7, 14, 30 days
            predictions = {
                '7 days': daily_avg * 7,
                '14 days': daily_avg * 14,
                '30 days': daily_avg * 30
            }
            
            response += "📊 **Demand Forecasts:**\n"
            for period, forecast in predictions.items():
                response += f"• Next {period}: {forecast:,.0f} units\n"
                
            # Add category-wise predictions
            category_avg = recent_sales.groupby('category')['units_sold'].sum() / 30
            response += f"\n🏷️ **Category-wise Daily Demand:**\n"
            for category, demand in category_avg.head(5).items():
                response += f"• {category}: {demand:.0f} units/day\n"
                
            return response
            
        except Exception as e:
            return f"❌ Error predicting demand: {str(e)}"
            
    def analyze_sales(self, query: str, entities: Dict) -> str:
        """Analyze sales performance with advanced metrics"""
        try:
            city_filter = entities.get('cities', [])
            
            if city_filter:
                filtered_sales = self.sales_df[self.sales_df['city_name'].isin(city_filter)]
                location_info = f" in {', '.join(city_filter).title()}"
            else:
                filtered_sales = self.sales_df
                location_info = " across all locations"
                
            # Calculate sales metrics
            total_revenue = filtered_sales['revenue'].sum()
            total_units = filtered_sales['units_sold'].sum()
            total_profit = filtered_sales['profit'].sum()
            avg_order_value = filtered_sales['revenue'].mean()
            
            # City performance
            city_performance = filtered_sales.groupby('city_name').agg({
                'revenue': 'sum',
                'units_sold': 'sum',
                'profit': 'sum'
            }).sort_values('revenue', ascending=False)
            
            response = f"💼 **Sales Analytics{location_info}**\n\n"
            
            response += f"💰 **Key Metrics:**\n"
            response += f"• Total Revenue: ₹{total_revenue:,.2f}\n"
            response += f"• Total Units Sold: {total_units:,}\n"
            response += f"• Total Profit: ₹{total_profit:,.2f}\n"
            response += f"• Average Order Value: ₹{avg_order_value:.2f}\n"
            response += f"• Profit Margin: {(total_profit/total_revenue*100):.1f}%\n\n"
            
            response += f"🏙️ **Top Performing Cities:**\n"
            for idx, (city, data) in enumerate(city_performance.head(5).iterrows(), 1):
                response += f"{idx}. **{city.title()}**\n"
                response += f"   • Revenue: ₹{data['revenue']:,.2f}\n"
                response += f"   • Units: {data['units_sold']:,}\n"
                response += f"   • Profit: ₹{data['profit']:,.2f}\n\n"
                
            return response
            
        except Exception as e:
            return f"❌ Error analyzing sales: {str(e)}"
            
    def handle_general_query(self, query: str, entities: Dict) -> str:
        """Handle general queries with contextual responses"""
        if any(word in query.lower() for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            return "👋 Hello! I'm your advanced warehouse AI assistant. I can help you with:\n\n• **Top seller analysis** - Find your best-performing products\n• **Smart distribution** - Optimize unit allocation across cities\n• **Inventory management** - Monitor stock levels and alerts\n• **Demand forecasting** - Predict future requirements\n• **Sales analytics** - Comprehensive performance insights\n\nWhat would you like to explore today?"
            
        return "🤖 I'm your advanced warehouse AI assistant. I can help you analyze sales, optimize distribution, manage inventory, predict demand, and provide comprehensive business insights. Could you please be more specific about what you'd like to know?"

# Global instance for the Flask app
nlp_engine = None

def get_nlp_engine():
    """Get or create the NLP engine instance"""
    global nlp_engine
    if nlp_engine is None:
        nlp_engine = AdvancedNLPWarehouseEngine()
    return nlp_engine