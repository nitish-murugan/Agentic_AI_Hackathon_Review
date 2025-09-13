"""
Simplified Advanced NLP Engine for Warehouse Management
Enhanced version without heavy ML dependencies for faster startup
"""

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

class SimplifiedNLPWarehouseEngine:
    """
    Simplified Advanced NLP Engine for Warehouse Management
    """
    
    def __init__(self):
        self.conversation_history = []
        self.context = {}
        self.load_nlp_capabilities()
        self.load_data()
        self.setup_query_patterns()
        
    def load_nlp_capabilities(self):
        """Load basic NLP capabilities"""
        try:
            import spacy
            # Try to load English model
            self.nlp = spacy.load("en_core_web_sm")
            print("✅ Advanced spaCy NLP model loaded successfully")
        except Exception as e:
            print(f"⚠️ spaCy model not available: {e}")
            print("🔧 Using basic NLP processing")
            self.nlp = None
            
    def load_data(self):
        """Load warehouse data with enhanced preprocessing"""
        try:
            # Try to load real data - check multiple paths
            dataset_paths = [
                '../dataset/salesData.csv',
                '../../dataset/salesData.csv', 
                'E:/Projects/Agentic_AI_Business_model/dataset/salesData.csv'
            ]
            
            sales_path = None
            inventory_path = None
            
            for path in dataset_paths:
                sales_test = path
                inventory_test = path.replace('salesData.csv', 'inventoryData.csv')
                
                if os.path.exists(sales_test) and os.path.exists(inventory_test):
                    sales_path = sales_test
                    inventory_path = inventory_test
                    break
            
            if sales_path and inventory_path:
                self.sales_df = pd.read_csv(sales_path)
                self.inventory_df = pd.read_csv(inventory_path)
                
                # Enhanced data preprocessing for sales data
                self.sales_df['date'] = pd.to_datetime(self.sales_df['date'])
                self.sales_df['city_name'] = self.sales_df['city_name'].str.lower().str.strip()
                
                # Enhanced data preprocessing for inventory data
                self.inventory_df['city_name'] = self.inventory_df['city_name'].str.lower().str.strip()
                
                # Clean product names and ensure consistent formatting
                self.sales_df['product_name'] = self.sales_df['product_name'].str.strip()
                self.inventory_df['product_name'] = self.inventory_df['product_name'].str.strip()
                
                print("✅ Real warehouse data loaded successfully")
                print(f"📊 Sales records: {len(self.sales_df):,}")
                print(f"📦 Inventory records: {len(self.inventory_df):,}")
            else:
                raise FileNotFoundError("Dataset files not found in expected locations")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self._create_enhanced_demo_data()
            
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
                    
                    inventory_data.append({
                        'product_id': product_id,
                        'product_name': f"{category} Product {product_id}",
                        'category': category,
                        'city_name': city,
                        'stock_quantity': current_stock  # Use correct column name
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
            'product_ids': [],
            'categories': [],
            'time_periods': []
        }
        
        # Extract product IDs (6-digit numbers that could be product IDs)
        product_id_matches = re.findall(r'\b(56\d{4}|47\d{4}|44\d{4}|49\d{4}|54\d{4})\b', query)
        entities['product_ids'] = [int(pid) for pid in product_id_matches]
        
        if self.nlp:
            # Use spaCy for advanced entity extraction
            doc = self.nlp(query)
            for ent in doc.ents:
                if ent.label_ in ['CARDINAL', 'QUANTITY']:
                    numbers = re.findall(r'\d+', ent.text)
                    if numbers:
                        num = int(numbers[0])
                        # Separate product IDs from regular numbers
                        if num >= 440000 and num <= 570000:
                            entities['product_ids'].append(num)
                        else:
                            entities['numbers'].append(num)
                elif ent.label_ in ['GPE', 'LOC']:
                    entities['cities'].append(ent.text.lower())
                elif ent.label_ in ['DATE', 'TIME']:
                    entities['time_periods'].append(ent.text)
        else:
            # Fallback to regex extraction
            numbers = re.findall(r'\b(\d+)\s*(?:units?|pieces?|items?)?\b', query, re.IGNORECASE)
            for n in numbers:
                num = int(n)
                # Separate product IDs from regular numbers
                if num >= 440000 and num <= 570000:
                    entities['product_ids'].append(num)
                else:
                    entities['numbers'].append(num)
            
            cities = ['mumbai', 'delhi', 'bangalore', 'chennai', 'kolkata', 'hyderabad', 'pune', 'ahmedabad', 'gurgaon', 'noida', 'bengaluru']
            for city in cities:
                if city in query.lower():
                    entities['cities'].append(city)
                    
        # Remove duplicates
        entities['product_ids'] = list(set(entities['product_ids']))
        entities['numbers'] = list(set(entities['numbers']))
        entities['cities'] = list(set(entities['cities']))
                    
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
            
            # Get current inventory levels by city using correct column name
            city_inventory = self.inventory_df.groupby('city_name')['stock_quantity'].agg(['sum', 'count', 'mean']).reset_index()
            city_inventory.columns = ['city_name', 'total_stock', 'product_count', 'avg_stock']
            
            # Get sales velocity by city (units sold per day)
            recent_sales = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
            sales_velocity = recent_sales.groupby('city_name')['units_sold'].sum() / 30
            
            # Calculate distribution optimization
            distribution_plan = []
            remaining_units = units_to_distribute
            
            for _, city_data in city_inventory.iterrows():
                city = city_data['city_name']
                current_stock = city_data['total_stock']
                product_count = city_data['product_count']
                
                # Get sales velocity for this city
                velocity = sales_velocity.get(city, 10)  # Default velocity
                
                # Calculate priority score based on available data
                # Higher velocity and lower stock per product = higher priority
                stock_per_product = current_stock / product_count if product_count > 0 else 0
                velocity_score = velocity / sales_velocity.max() if len(sales_velocity) > 0 and sales_velocity.max() > 0 else 0.5
                stock_score = 1 / (1 + stock_per_product / 50)  # Inverse relationship with stock density
                
                # Combined priority score
                priority_score = velocity_score * 0.7 + stock_score * 0.3
                
                # Calculate recommended allocation
                # Assume capacity of 1000 units per city as a reasonable estimate
                estimated_capacity = 1000
                available_capacity = max(0, estimated_capacity - current_stock)
                recommended_units = min(
                    available_capacity,
                    int(remaining_units * priority_score * 2),
                    remaining_units
                )
                
                if recommended_units > 0:
                    distribution_plan.append({
                        'city': city,
                        'units': recommended_units,
                        'current_stock': current_stock,
                        'after_stock': current_stock + recommended_units,
                        'priority_score': priority_score,
                        'sales_velocity': velocity,
                        'product_count': product_count
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
                response += f"   • Products in City: {plan['product_count']}\n"
                response += f"   • Sales Velocity: {plan['sales_velocity']:.1f} units/day\n"
                response += f"   • Priority Score: {plan['priority_score']:.2f}\n\n"
                total_allocated += plan['units']
                
            response += f"📊 **Summary:**\n"
            response += f"• Total Allocated: {total_allocated:,} units ({total_allocated/units_to_distribute*100:.1f}%)\n"
            response += f"• Remaining Units: {remaining_units:,}\n"
            response += f"• Distribution Strategy: Priority-based allocation considering sales velocity and current stock levels\n"
            
            if remaining_units > 0:
                response += f"\n⚠️ **Note:** {remaining_units:,} units remaining. Consider expanding distribution to additional cities or increasing capacity in high-priority cities."
                
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
                
            # Calculate inventory metrics using correct column names
            total_stock = filtered_inventory['stock_quantity'].sum()
            
            # Calculate basic metrics without reorder_level and max_capacity (not in real data)
            avg_stock_per_product = filtered_inventory['stock_quantity'].mean()
            zero_stock_items = filtered_inventory[filtered_inventory['stock_quantity'] == 0]
            low_stock_items = filtered_inventory[filtered_inventory['stock_quantity'] <= 10]  # Threshold of 10
            high_stock_items = filtered_inventory[filtered_inventory['stock_quantity'] >= 100]  # Threshold of 100
            
            response = f"📦 **Inventory Analysis{location_info}**\n\n"
            
            response += f"📊 **Overall Metrics:**\n"
            response += f"• Total Stock: {total_stock:,} units\n"
            response += f"• Average Stock per Product: {avg_stock_per_product:.1f} units\n"
            response += f"• Total SKUs: {len(filtered_inventory):,}\n"
            response += f"• Out of Stock Items: {len(zero_stock_items):,}\n\n"
            
            if len(low_stock_items) > 0:
                response += f"⚠️ **Low Stock Alerts ({len(low_stock_items)} items with ≤10 units):**\n"
                for _, item in low_stock_items.head(10).iterrows():
                    response += f"• **{item['product_name'].strip()}** in {item['city_name'].title()}: {item['stock_quantity']} units\n"
                if len(low_stock_items) > 10:
                    response += f"• ... and {len(low_stock_items) - 10} more items\n"
                response += "\n"
                
            if len(zero_stock_items) > 0:
                response += f"� **Out of Stock Items ({len(zero_stock_items)} items):**\n"
                for _, item in zero_stock_items.head(5).iterrows():
                    response += f"• **{item['product_name'].strip()}** in {item['city_name'].title()}\n"
                if len(zero_stock_items) > 5:
                    response += f"• ... and {len(zero_stock_items) - 5} more items\n"
                response += "\n"
                
            if len(high_stock_items) > 0:
                response += f"📈 **High Stock Items ({len(high_stock_items)} items with ≥100 units):**\n"
                for _, item in high_stock_items.head(5).iterrows():
                    response += f"• **{item['product_name'].strip()}** in {item['city_name'].title()}: {item['stock_quantity']} units\n"
                if len(high_stock_items) > 5:
                    response += f"• ... and {len(high_stock_items) - 5} more items\n"
                response += "\n"
                    
            # Add city-wise breakdown
            city_stock = filtered_inventory.groupby('city_name')['stock_quantity'].agg(['sum', 'count', 'mean']).sort_values('sum', ascending=False)
            
            response += f"🏙️ **Stock by City:**\n"
            for idx, (city, data) in enumerate(city_stock.head(8).iterrows(), 1):
                response += f"{idx}. **{city.title()}**: {data['sum']:,} units ({data['count']} products, avg: {data['mean']:.1f})\n"
                
            return response
            
        except Exception as e:
            return f"❌ Error analyzing inventory: {str(e)}"
            
    def predict_demand(self, query: str, entities: Dict) -> str:
        """Predict demand using advanced statistical methods"""
        try:
            # Check if specific product IDs were mentioned
            product_ids = entities.get('product_ids', [])
            
            if product_ids:
                # Specific product demand prediction
                response = f"🔮 **Product-Specific Demand Forecast**\n\n"
                
                for product_id in product_ids:
                    # Find product in sales data
                    product_sales = self.sales_df[self.sales_df['product_id'] == product_id]
                    
                    if len(product_sales) > 0:
                        # Get product details
                        product_name = product_sales['product_name'].iloc[0].strip()
                        category = product_sales['category'].iloc[0]
                        
                        response += f"📦 **Product ID: {product_id}**\n"
                        response += f"**Name:** {product_name}\n"
                        response += f"**Category:** {category}\n\n"
                        
                        # Calculate historical sales
                        total_sales = product_sales['units_sold'].sum()
                        recent_sales = product_sales[product_sales['date'] >= datetime.now() - timedelta(days=30)]
                        
                        if len(recent_sales) > 0:
                            avg_daily_sales = recent_sales['units_sold'].sum() / 30
                            
                            # Sales by city
                            city_sales = product_sales.groupby('city_name')['units_sold'].sum().sort_values(ascending=False)
                            
                            response += f"📊 **Historical Performance:**\n"
                            response += f"• Total Sales (All Time): {total_sales:,} units\n"
                            response += f"• Recent Sales (30 days): {recent_sales['units_sold'].sum():,} units\n"
                            response += f"• Average Daily Sales: {avg_daily_sales:.1f} units/day\n\n"
                            
                            # Demand predictions
                            response += f"🔮 **Demand Forecasts:**\n"
                            response += f"• Next 7 days: {avg_daily_sales * 7:.0f} units\n"
                            response += f"• Next 14 days: {avg_daily_sales * 14:.0f} units\n"
                            response += f"• Next 30 days: {avg_daily_sales * 30:.0f} units\n\n"
                            
                            # Top selling cities
                            if len(city_sales) > 0:
                                response += f"🏙️ **Top Selling Cities:**\n"
                                for idx, (city, sales) in enumerate(city_sales.head(5).items(), 1):
                                    response += f"{idx}. **{city.title()}**: {sales:,} units\n"
                                response += "\n"
                            
                            # Get current inventory for this product
                            product_inventory = self.inventory_df[self.inventory_df['product_id'] == product_id]
                            if len(product_inventory) > 0:
                                total_stock = product_inventory['stock_quantity'].sum()
                                response += f"📦 **Current Inventory:**\n"
                                response += f"• Total Stock: {total_stock:,} units\n"
                                
                                # Stock by city
                                city_stock = product_inventory.groupby('city_name')['stock_quantity'].sum()
                                city_stock = city_stock[city_stock > 0].sort_values(ascending=False)
                                
                                if len(city_stock) > 0:
                                    response += f"• Available in {len(city_stock)} cities\n"
                                    for idx, (city, stock) in enumerate(city_stock.head(3).items(), 1):
                                        response += f"  {idx}. {city.title()}: {stock:,} units\n"
                                
                                # Calculate days of inventory
                                if avg_daily_sales > 0:
                                    days_of_inventory = total_stock / avg_daily_sales
                                    response += f"• Days of Inventory: {days_of_inventory:.1f} days\n"
                                    
                                    if days_of_inventory < 7:
                                        response += f"⚠️ **Alert:** Low inventory! Restock recommended.\n"
                                    elif days_of_inventory > 60:
                                        response += f"✅ **Status:** Good inventory levels.\n"
                            
                        else:
                            response += f"� **Historical Performance:**\n"
                            response += f"• Total Sales (All Time): {total_sales:,} units\n"
                            response += f"• No recent sales data (last 30 days)\n\n"
                            
                            response += f"⚠️ **Note:** Limited recent sales data. Using historical average for prediction.\n"
                            if total_sales > 0:
                                # Use all-time average
                                days_span = (product_sales['date'].max() - product_sales['date'].min()).days
                                if days_span > 0:
                                    avg_daily_sales = total_sales / days_span
                                    response += f"🔮 **Conservative Forecasts:**\n"
                                    response += f"• Next 7 days: {avg_daily_sales * 7:.0f} units\n"
                                    response += f"• Next 14 days: {avg_daily_sales * 14:.0f} units\n"
                                    response += f"• Next 30 days: {avg_daily_sales * 30:.0f} units\n"
                    else:
                        # Product not found in sales data
                        response += f"❌ **Product ID {product_id} not found in sales history**\n\n"
                        
                        # Check if it exists in inventory
                        product_inventory = self.inventory_df[self.inventory_df['product_id'] == product_id]
                        if len(product_inventory) > 0:
                            product_name = product_inventory['product_name'].iloc[0].strip()
                            category = product_inventory['category'].iloc[0]
                            total_stock = product_inventory['stock_quantity'].sum()
                            
                            response += f"📦 **Found in Inventory:**\n"
                            response += f"**Name:** {product_name}\n"
                            response += f"**Category:** {category}\n"
                            response += f"**Current Stock:** {total_stock:,} units\n\n"
                            
                            response += f"💡 **Recommendation:** This product has no sales history. Consider:\n"
                            response += f"• Market testing before large inventory investment\n"
                            response += f"• Analyzing similar products in the {category} category\n"
                            response += f"• Starting with small quantities in high-traffic locations\n"
                        else:
                            response += f"• Product not found in inventory either\n"
                            response += f"• Please verify the product ID: {product_id}\n"
                    
                    response += "\n" + "="*50 + "\n\n"
                
                return response
                
            else:
                # General demand prediction (existing functionality)
                response = "📈 **General Warehouse Demand Forecast**\n\n"
                    
                # Calculate demand trends
                recent_sales = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
                daily_avg = recent_sales.groupby(recent_sales['date'].dt.date)['units_sold'].sum().mean()
                
                # Predict for next 7, 14, 30 days
                predictions = {
                    '7 days': daily_avg * 7,
                    '14 days': daily_avg * 14,
                    '30 days': daily_avg * 30
                }
                
                response += "📊 **Overall Demand Forecasts:**\n"
                for period, forecast in predictions.items():
                    response += f"• Next {period}: {forecast:,.0f} units\n"
                    
                # Add category-wise predictions
                category_avg = recent_sales.groupby('category')['units_sold'].sum() / 30
                response += f"\n🏷️ **Category-wise Daily Demand:**\n"
                for category, demand in category_avg.head(5).items():
                    response += f"• {category}: {demand:.0f} units/day\n"
                    
                response += f"\n💡 **Tip:** Specify a product ID (e.g., 'Predict demand for product 561099') for detailed product-specific forecasts!\n"
                
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
            return "👋 Hello! I'm your **Advanced Warehouse AI Assistant** powered by sophisticated NLP models. I can help you with:\n\n🏆 **Top seller analysis** - Find your best-performing products\n🚚 **Smart distribution** - Optimize unit allocation across cities\n📦 **Inventory management** - Monitor stock levels and alerts\n🔮 **Demand forecasting** - Predict future requirements\n📊 **Sales analytics** - Comprehensive performance insights\n\nWhat would you like to explore today?"
            
        return "🤖 I'm your **Advanced Warehouse AI Assistant** with sophisticated NLP capabilities. I can analyze sales, optimize distribution, manage inventory, predict demand, and provide comprehensive business insights.\n\n💡 **Try asking me:**\n• What are our top selling products?\n• I have 1000 units, which places should I distribute them?\n• Show me inventory levels\n• Predict demand for next month\n• Which city has the best sales performance?\n\nCould you please be more specific about what you'd like to know?"

# Global instance for the Flask app
nlp_engine = None

def get_nlp_engine():
    """Get or create the NLP engine instance"""
    global nlp_engine
    if nlp_engine is None:
        nlp_engine = SimplifiedNLPWarehouseEngine()
    return nlp_engine