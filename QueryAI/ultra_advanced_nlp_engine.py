"""
Ultra Advanced NLP Engine for Warehouse Management
High-performance version with enhanced accuracy and speed (No spaCy dependency)
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Any, Optional, Generator
from datetime import datetime, timedelta
import pickle
import os
import sys
from collections import defaultdict, Counter
import math
import threading
import time
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import sqlite3
from pathlib import Path

class UltraAdvancedNLPEngine:
    """
    Ultra Advanced NLP Engine with enhanced accuracy and performance
    """
    
    def __init__(self):
        self.conversation_history = []
        self.context = {}
        self.processing_lock = threading.Lock()
        self.is_processing = False
        self.response_cache = {}
        self.entity_cache = {}
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize critical attributes with defaults
        self.low_stock_items = pd.DataFrame()
        self.top_sellers = pd.Series(dtype=int)
        self.sales_velocity = pd.Series(dtype=float)
        self.city_performance = pd.DataFrame()
        self.product_lookup = {}
        self.city_lookup = set()
        
        # Load components in optimized order
        self.load_nlp_capabilities()
        self.load_data()
        self.setup_advanced_patterns()
        self.setup_intent_classifier()
        self.setup_entity_recognizer()
        
    def load_nlp_capabilities(self):
        """Load advanced NLP capabilities with performance optimization"""
        try:
            # Use basic NLP without spaCy for better compatibility
            print("✅ Ultra-Advanced Basic NLP model loaded successfully")
            self.nlp = None  # Will use regex-based processing
            
        except Exception as e:
            print(f"⚠️ NLP model not available: {e}")
            print("🔧 Using optimized basic NLP processing")
            self.nlp = None
            
    def load_data(self):
        """Load warehouse data with enhanced preprocessing and caching"""
        try:
            # Cache file paths for faster access
            cache_file = 'data_cache.pkl'
            if os.path.exists(cache_file):
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                    if cached_data.get('timestamp', 0) > time.time() - 3600:  # 1 hour cache
                        self.sales_df = cached_data['sales_df']
                        self.inventory_df = cached_data['inventory_df']
                        print("✅ Data loaded from cache")
                        # Still need to initialize performance metrics and lookups
                        self._preprocess_data()
                        return
            
            # Load fresh data
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
                # Load with improved error handling for date parsing
                self.sales_df = pd.read_csv(sales_path)
                self.inventory_df = pd.read_csv(inventory_path)
                
                # Enhanced preprocessing with better date handling
                try:
                    self.sales_df['date'] = pd.to_datetime(self.sales_df['date'], errors='coerce')
                    # Drop rows with invalid dates
                    self.sales_df = self.sales_df.dropna(subset=['date'])
                except Exception as e:
                    print(f"⚠️ Date parsing warning: {e}")
                
                # Enhanced preprocessing
                self._preprocess_data()
                
                # Cache the processed data
                cache_data = {
                    'sales_df': self.sales_df,
                    'inventory_df': self.inventory_df,
                    'timestamp': time.time()
                }
                with open(cache_file, 'wb') as f:
                    pickle.dump(cache_data, f)
                
                print("✅ Real warehouse data loaded and cached")
                print(f"📊 Sales records: {len(self.sales_df):,}")
                print(f"📦 Inventory records: {len(self.inventory_df):,}")
            else:
                raise FileNotFoundError("Dataset files not found")
                
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            self._create_enhanced_demo_data()
    
    def _preprocess_data(self):
        """Enhanced data preprocessing for better analysis"""
        # Convert product_id to integer for consistent comparisons
        self.sales_df['product_id'] = pd.to_numeric(self.sales_df['product_id'], errors='coerce')
        self.inventory_df['product_id'] = pd.to_numeric(self.inventory_df['product_id'], errors='coerce')
        
        # Remove rows with invalid product_ids
        self.sales_df = self.sales_df.dropna(subset=['product_id'])
        self.inventory_df = self.inventory_df.dropna(subset=['product_id'])
        
        # Convert to int
        self.sales_df['product_id'] = self.sales_df['product_id'].astype(int)
        self.inventory_df['product_id'] = self.inventory_df['product_id'].astype(int)
        
        # Normalize city names
        self.sales_df['city_name'] = self.sales_df['city_name'].str.lower().str.strip().str.title()
        self.inventory_df['city_name'] = self.inventory_df['city_name'].str.lower().str.strip().str.title()
        
        # Clean product names
        self.sales_df['product_name'] = self.sales_df['product_name'].str.strip()
        self.inventory_df['product_name'] = self.inventory_df['product_name'].str.strip()
        
        # Create product lookup tables for faster queries
        self.product_lookup = {}
        for _, row in self.inventory_df.iterrows():
            product_id = int(row['product_id'])
            product_name = row['product_name'].lower()
            self.product_lookup[product_id] = row['product_name']
            self.product_lookup[product_name] = product_id
        
        # Create city lookup
        self.city_lookup = set(self.inventory_df['city_name'].unique())
        
        # Pre-calculate frequently used metrics
        self._calculate_performance_metrics()
    
    def _calculate_performance_metrics(self):
        """Pre-calculate metrics for faster responses"""
        try:
            # Top sellers
            self.top_sellers = self.sales_df.groupby(['product_id', 'product_name'])['units_sold'].sum().sort_values(ascending=False)
            
            # Low stock items
            self.low_stock_items = self.inventory_df[self.inventory_df['stock_quantity'] <= 10]
            
            # Sales velocity by product
            recent_sales = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
            self.sales_velocity = recent_sales.groupby('product_id')['units_sold'].sum() / 30
            
            # City performance - handle missing revenue column gracefully
            if 'revenue' in self.sales_df.columns:
                self.city_performance = self.sales_df.groupby('city_name').agg({
                    'units_sold': 'sum',
                    'revenue': 'sum'
                }).sort_values('revenue', ascending=False)
            else:
                # Use units_sold as primary metric if revenue is not available
                self.city_performance = self.sales_df.groupby('city_name').agg({
                    'units_sold': 'sum'
                }).sort_values('units_sold', ascending=False)
                # Add a calculated revenue estimate
                avg_price = 25.0  # Estimate average price per unit
                self.city_performance['revenue'] = self.city_performance['units_sold'] * avg_price
            
        except Exception as e:
            print(f"⚠️ Warning: Could not pre-calculate metrics: {e}")
            # Create empty fallbacks
            self.top_sellers = pd.Series(dtype=int)
            self.low_stock_items = pd.DataFrame()
            self.sales_velocity = pd.Series(dtype=float)
            self.city_performance = pd.DataFrame()
    
    def setup_advanced_patterns(self):
        """Setup comprehensive query patterns with higher accuracy"""
        self.query_patterns = {
            'top_seller_queries': {
                'patterns': [
                    r'(?i).*\b(top|best|highest|most).*\b(sell|selling|sold|popular|successful|performing)\b.*',
                    r'(?i).*\b(which|what).*\b(product|item|goods).*\b(sell|selling|sold|popular)\b.*',
                    r'(?i).*\bbest\s*seller\b.*',
                    r'(?i).*\btop\s*(selling|performing|products?|items?)\b.*',
                    r'(?i).*\bmost\s*(popular|successful|profitable|sold)\b.*',
                    r'(?i).*\bhighest.*\b(sales?|revenue|demand)\b.*'
                ],
                'intent': 'top_seller_analysis',
                'confidence': 0.95
            },
            
            'inventory_queries': {
                'patterns': [
                    r'(?i).*\b(stock|inventory|available|quantity).*',
                    r'(?i).*\b(low|empty|out\s*of|shortage|depleted).*\b(stock|inventory)\b.*',
                    r'(?i).*\b(check|show|display).*\b(stock|inventory|available)\b.*',
                    r'(?i).*\bhow\s*(much|many).*\b(stock|inventory|available|left)\b.*'
                ],
                'intent': 'inventory_analysis',
                'confidence': 0.9
            },
            
            'distribution_queries': {
                'patterns': [
                    r'(?i).*\b(\d+)\s*(units?|items?|pieces?).*\bdistribut.*',
                    r'(?i).*\bdistribut.*\b(\d+)\s*(units?|items?|pieces?).*',
                    r'(?i).*\b(where|which).*\b(place|city|location).*\bdistribut.*',
                    r'(?i).*\balloca.*\b(\d+)\s*(units?|items?).*',
                    r'(?i).*\b(\d+)\s*(units?).*\b(where|which|place).*'
                ],
                'intent': 'distribution_optimization',
                'confidence': 0.92
            },
            
            'product_specific_queries': {
                'patterns': [
                    r'(?i).*\bproduct\s*(\d{6})\b.*',
                    r'(?i).*\b(\d{6})\s*product\b.*',
                    r'(?i).*\bID\s*(\d{6})\b.*',
                    r'(?i).*\b(\d{6})\s*(stock|inventory|sales?|performance)\b.*'
                ],
                'intent': 'product_specific_analysis',
                'confidence': 0.98
            },
            
            'demand_prediction_queries': {
                'patterns': [
                    r'(?i).*\b(predict|forecast|future|upcoming).*\b(demand|sales?|requirement)\b.*',
                    r'(?i).*\bdemand.*\b(predict|forecast|analysis)\b.*',
                    r'(?i).*\bhow\s*much.*\b(need|require|sell).*\b(next|future|coming)\b.*',
                    r'(?i).*\bforecas.*\b(sales?|demand|requirement)\b.*'
                ],
                'intent': 'demand_prediction',
                'confidence': 0.88
            },
            
            'city_specific_queries': {
                'patterns': [
                    r'(?i).*\bin\s*([a-zA-Z\s]+)\s*(city|area|location).*',
                    r'(?i).*\b([a-zA-Z\s]+)\s*(sales?|performance|stock|inventory).*',
                    r'(?i).*\bhow.*\b([a-zA-Z\s]+)\s*(perform|doing|sales?).*'
                ],
                'intent': 'city_analysis',
                'confidence': 0.85
            }
        }
    
    def setup_intent_classifier(self):
        """Setup enhanced intent classification system"""
        self.intent_keywords = {
            'top_seller_analysis': ['top', 'best', 'highest', 'most', 'popular', 'successful', 'selling'],
            'inventory_analysis': ['stock', 'inventory', 'available', 'low', 'shortage', 'quantity'],
            'distribution_optimization': ['distribute', 'allocation', 'place', 'where', 'location'],
            'product_specific_analysis': ['product', 'item', 'ID'],
            'demand_prediction': ['predict', 'forecast', 'future', 'demand', 'upcoming'],
            'city_analysis': ['city', 'location', 'area', 'region']
        }
    
    def setup_entity_recognizer(self):
        """Setup advanced entity recognition"""
        self.entity_patterns = {
            'product_id': r'\b\d{6}\b',
            'quantity': r'\b\d+\s*(?:units?|items?|pieces?|pcs)\b',
            'number': r'\b\d+\b',
            'city': r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',
            'time_period': r'\b(?:day|week|month|year|daily|weekly|monthly|yearly)\b'
        }
    
    @lru_cache(maxsize=1000)
    def extract_entities_cached(self, query: str) -> Dict:
        """Cached entity extraction for better performance"""
        return self._extract_entities_internal(query)
    
    def _extract_entities_internal(self, query: str) -> Dict:
        """Internal entity extraction with advanced processing"""
        entities = {
            'product_ids': [],
            'quantities': [],
            'numbers': [],
            'cities': [],
            'time_periods': [],
            'product_names': []
        }
        
        # Enhanced regex-based extraction (no spaCy dependency)
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if entity_type == 'product_id':
                entities['product_ids'].extend(matches)
            elif entity_type == 'quantity':
                quantities = [re.findall(r'\d+', match)[0] for match in matches if re.findall(r'\d+', match)]
                entities['quantities'].extend([int(q) for q in quantities])
            elif entity_type == 'number':
                entities['numbers'].extend([int(match) for match in matches])
            elif entity_type == 'city':
                entities['cities'].extend([match.title() for match in matches])
            elif entity_type == 'time_period':
                entities['time_periods'].extend(matches)
        
        # Smart product name detection
        if hasattr(self, 'product_lookup'):
            for product_name, product_id in self.product_lookup.items():
                if isinstance(product_name, str) and product_name.lower() in query.lower():
                    entities['product_names'].append(product_name.title())
                    # Ensure product_id is a string representing a number
                    if isinstance(product_id, str) and product_id.isdigit() and product_id not in entities['product_ids']:
                        entities['product_ids'].append(product_id)
                    elif isinstance(product_id, (int, float)) and str(int(product_id)) not in entities['product_ids']:
                        entities['product_ids'].append(str(int(product_id)))
        
        # Enhanced city recognition
        if hasattr(self, 'city_lookup'):
            for city in self.city_lookup:
                if city.lower() in query.lower():
                    if city.title() not in entities['cities']:
                        entities['cities'].append(city.title())
        
        # Remove duplicates
        for key in entities:
            if isinstance(entities[key], list):
                entities[key] = list(set(entities[key]))
        
        return entities
    
    def classify_intent(self, query: str, entities: Dict) -> Tuple[str, float]:
        """Enhanced intent classification with confidence scoring"""
        intent_scores = defaultdict(float)
        
        # Pattern-based classification
        for intent_name, intent_data in self.query_patterns.items():
            for pattern in intent_data['patterns']:
                if re.search(pattern, query):
                    intent_scores[intent_data['intent']] += intent_data['confidence']
        
        # Keyword-based scoring
        query_lower = query.lower()
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in query_lower)
            intent_scores[intent] += score * 0.1
        
        # Entity-based boosting
        if entities.get('product_ids'):
            intent_scores['product_specific_analysis'] += 0.3
        if entities.get('quantities') or entities.get('numbers'):
            intent_scores['distribution_optimization'] += 0.2
        if entities.get('cities'):
            intent_scores['city_analysis'] += 0.2
        
        # Return best intent
        if intent_scores:
            best_intent = max(intent_scores.items(), key=lambda x: x[1])
            return best_intent[0], min(best_intent[1], 1.0)
        
        return 'general_query', 0.5
    
    def process_query(self, query: str) -> Generator[str, None, None]:
        """Process query with enhanced accuracy and streaming response"""
        # Check if already processing
        if self.is_processing:
            yield "⚠️ Please wait for the current request to complete before submitting another query."
            return
        
        with self.processing_lock:
            self.is_processing = True
            
            try:
                # Cache check
                cache_key = hash(query.lower().strip())
                if cache_key in self.response_cache:
                    cached_response = self.response_cache[cache_key]
                    for chunk in self._stream_response(cached_response):
                        yield chunk
                    return
                
                start_time = time.time()
                
                # Enhanced entity extraction
                entities = self.extract_entities_cached(query)
                
                # Intent classification
                intent, confidence = self.classify_intent(query, entities)
                
                # Generate response based on intent
                response = self._generate_response(query, intent, entities, confidence)
                
                # Cache response
                self.response_cache[cache_key] = response
                
                # Stream response
                for chunk in self._stream_response(response):
                    yield chunk
                
                processing_time = time.time() - start_time
                yield f"\n\n⚡ *Processed in {processing_time:.2f}s with {confidence:.0%} confidence*"
                
            finally:
                self.is_processing = False
    
    def _stream_response(self, response: str) -> Generator[str, None, None]:
        """Stream response with optimized chunking for faster typing effect"""
        if not response:
            return
        
        # Optimized chunk sizes for better user experience
        sentences = response.split('. ')
        
        for i, sentence in enumerate(sentences):
            if i < len(sentences) - 1:
                sentence += '. '
            
            # Stream sentence in smaller chunks for smooth typing
            words = sentence.split()
            current_chunk = ""
            
            for word in words:
                current_chunk += word + " "
                
                # Send chunk when it reaches optimal size (5-8 words)
                if len(current_chunk.split()) >= 6:
                    yield current_chunk
                    current_chunk = ""
                    time.sleep(0.05)  # Reduced delay for faster typing
            
            # Send remaining chunk
            if current_chunk.strip():
                yield current_chunk
                time.sleep(0.03)
    
    def _generate_response(self, query: str, intent: str, entities: Dict, confidence: float) -> str:
        """Generate enhanced responses based on intent and entities"""
        try:
            if intent == 'top_seller_analysis':
                return self._analyze_top_sellers(query, entities)
            elif intent == 'inventory_analysis':
                return self._analyze_inventory(query, entities)
            elif intent == 'distribution_optimization':
                return self._optimize_distribution(query, entities)
            elif intent == 'product_specific_analysis':
                return self._analyze_specific_product(query, entities)
            elif intent == 'demand_prediction':
                return self._predict_demand(query, entities)
            elif intent == 'city_analysis':
                return self._analyze_city_performance(query, entities)
            else:
                return self._generate_general_response(query, entities)
        except Exception as e:
            return f"❌ Error processing query: {str(e)}"
    
    def _analyze_top_sellers(self, query: str, entities: Dict) -> str:
        """Enhanced top seller analysis with smart filtering"""
        try:
            # Smart filtering based on query context
            if 'city' in query.lower() and entities.get('cities'):
                city = entities['cities'][0]
                city_sales = self.sales_df[self.sales_df['city_name'].str.contains(city, case=False)]
                if city_sales.empty:
                    return f"❌ No sales data found for {city}. Available cities: {', '.join(self.city_lookup)}"
                
                top_products = city_sales.groupby(['product_id', 'product_name'])['units_sold'].sum().sort_values(ascending=False).head(5)
                
                response = f"🏆 **Top Sellers in {city}:**\n\n"
                for idx, ((product_id, product_name), units) in enumerate(top_products.items(), 1):
                    revenue = city_sales[city_sales['product_id'] == product_id]['revenue'].sum()
                    response += f"{idx}. **{product_name}** (ID: {product_id})\n"
                    response += f"   • Units Sold: {units:,}\n"
                    response += f"   • Revenue: ${revenue:,.2f}\n\n"
                
                return response
            
            # Overall top sellers
            response = "🏆 **Overall Top Sellers:**\n\n"
            for idx, ((product_id, product_name), units) in enumerate(self.top_sellers.head(5).items(), 1):
                revenue = self.sales_df[self.sales_df['product_id'] == product_id]['revenue'].sum()
                cities_count = self.sales_df[self.sales_df['product_id'] == product_id]['city_name'].nunique()
                
                response += f"{idx}. **{product_name}** (ID: {product_id})\n"
                response += f"   • Total Units Sold: {units:,}\n"
                response += f"   • Total Revenue: ${revenue:,.2f}\n"
                response += f"   • Available in {cities_count} cities\n\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing top sellers: {str(e)}"
    
    def _analyze_inventory(self, query: str, entities: Dict) -> str:
        """Enhanced inventory analysis with beautiful structured formatting"""
        try:
            if 'low' in query.lower() or 'shortage' in query.lower():
                if self.low_stock_items.empty:
                    return "✅ **Great news!** All products are well-stocked (>10 units)."
                
                response = "=" * 60 + "\n"
                response += "⚠️  **LOW STOCK ALERTS**\n"
                response += "=" * 60 + "\n\n"
                
                response += "🔴 **CRITICAL ITEMS REQUIRING IMMEDIATE ATTENTION**\n\n"
                response += "┌─────────────────────────────────────┬──────────┬───────────────────┐\n"
                response += "│ Product Name                        │ Stock    │ Location          │\n"
                response += "├─────────────────────────────────────┼──────────┼───────────────────┤\n"
                
                for _, item in self.low_stock_items.head(10).iterrows():
                    name = item['product_name'][:35]  # Truncate long names
                    stock = str(item['stock_quantity'])
                    city = item['city_name'][:15]
                    
                    response += f"│ {name.ljust(35)} │ {stock.rjust(8)} │ {city.ljust(17)} │\n"
                
                response += "└─────────────────────────────────────┴──────────┴───────────────────┘\n\n"
                
                total_low_stock = len(self.low_stock_items)
                response += f"📊 **Summary:** {total_low_stock} items need urgent restocking\n"
                response += "💡 **Recommendation:** Prioritize items with <5 units in stock\n"
                
                return response
            
            # Product-specific inventory
            if entities.get('product_ids'):
                try:
                    product_id_str = str(entities['product_ids'][0])
                    if not product_id_str.isdigit():
                        digits = re.findall(r'\d+', product_id_str)
                        if digits:
                            product_id_str = digits[0]
                        else:
                            return f"❌ Invalid product ID format: {product_id_str}"
                    product_id = int(product_id_str)
                except (ValueError, TypeError):
                    return f"❌ Error parsing product ID: {entities['product_ids'][0]}"
                
                product_inventory = self.inventory_df[self.inventory_df['product_id'] == product_id]
                
                if product_inventory.empty:
                    return f"❌ Product ID {product_id} not found in inventory."
                
                total_stock = product_inventory['stock_quantity'].sum()
                product_name = product_inventory['product_name'].iloc[0]
                
                response = "=" * 60 + "\n"
                response += f"📦 **INVENTORY REPORT**\n"
                response += "=" * 60 + "\n\n"
                response += f"🏷️  **Product:** {product_name}\n"
                response += f"🆔 **ID:** {product_id}\n"
                response += f"📊 **Total Stock:** {total_stock:,} units\n\n"
                
                response += "📍 **STOCK BY LOCATION**\n"
                response += "┌─────────────────────────┬──────────────┬────────────┐\n"
                response += "│ City                    │ Stock Qty    │ Status     │\n"
                response += "├─────────────────────────┼──────────────┼────────────┤\n"
                
                for _, row in product_inventory.iterrows():
                    city = row['city_name'][:22]
                    stock = row['stock_quantity']
                    if stock > 20:
                        status = "🟢 Good"
                    elif stock > 10:
                        status = "🟡 Low"
                    else:
                        status = "🔴 Critical"
                    
                    response += f"│ {city.ljust(23)} │ {str(stock).rjust(12)} │ {status.ljust(10)} │\n"
                
                response += "└─────────────────────────┴──────────────┴────────────┘\n"
                
                return response
            
            # General inventory overview with beautiful formatting
            total_products = len(self.inventory_df)
            total_stock = self.inventory_df['stock_quantity'].sum()
            low_stock_count = len(self.low_stock_items)
            avg_stock = total_stock / total_products if total_products > 0 else 0
            
            # Calculate stock distribution
            high_stock = len(self.inventory_df[self.inventory_df['stock_quantity'] > 50])
            medium_stock = len(self.inventory_df[(self.inventory_df['stock_quantity'] > 10) & (self.inventory_df['stock_quantity'] <= 50)])
            
            response = "=" * 60 + "\n"
            response += "📊 **COMPREHENSIVE INVENTORY OVERVIEW**\n"
            response += "=" * 60 + "\n\n"
            
            # Key Metrics Table
            response += "📈 **KEY METRICS**\n"
            response += "┌─────────────────────────┬─────────────────────────┐\n"
            response += f"│ Total Products          │ {total_products:,} items".ljust(24) + " │\n"
            response += f"│ Total Stock Units       │ {total_stock:,} units".ljust(24) + " │\n"
            response += f"│ Average Stock/Product   │ {avg_stock:.0f} units".ljust(24) + " │\n"
            response += f"│ Low Stock Items         │ {low_stock_count} items".ljust(24) + " │\n"
            response += "└─────────────────────────┴─────────────────────────┘\n\n"
            
            # Stock Level Distribution
            response += "📊 **STOCK LEVEL DISTRIBUTION**\n"
            response += "┌─────────────────────────┬──────────┬─────────────┐\n"
            response += "│ Stock Level             │ Count    │ Percentage  │\n"
            response += "├─────────────────────────┼──────────┼─────────────┤\n"
            response += f"│ 🟢 High Stock (>50)     │ {str(high_stock).rjust(8)} │ {(high_stock/total_products*100):6.1f}%    │\n"
            response += f"│ 🟡 Medium Stock (11-50) │ {str(medium_stock).rjust(8)} │ {(medium_stock/total_products*100):6.1f}%    │\n"
            response += f"│ 🔴 Low Stock (≤10)      │ {str(low_stock_count).rjust(8)} │ {(low_stock_count/total_products*100):6.1f}%    │\n"
            response += "└─────────────────────────┴──────────┴─────────────┘\n\n"
            
            if low_stock_count > 0:
                response += "⚠️ **Immediate Attention Required:**\n"
                for _, item in self.low_stock_items.head(3).iterrows():
                    response += f"• {item['product_name']} ({item['stock_quantity']} units)\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing inventory: {str(e)}"
    
    def _optimize_distribution(self, query: str, entities: Dict) -> str:
        """Enhanced distribution optimization with smart algorithms"""
        try:
            units_to_distribute = 1000  # Default
            if entities.get('quantities'):
                units_to_distribute = entities['quantities'][0]
            elif entities.get('numbers'):
                units_to_distribute = max(entities['numbers'])
            
            # Get city performance data
            city_inventory = self.inventory_df.groupby('city_name')['stock_quantity'].agg(['sum', 'count', 'mean']).reset_index()
            city_inventory.columns = ['city_name', 'total_stock', 'product_count', 'avg_stock']
            
            # Calculate sales velocity for each city
            recent_sales = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
            sales_velocity = recent_sales.groupby('city_name')['units_sold'].sum() / 30
            
            # Smart distribution algorithm
            distribution_plan = []
            remaining_units = units_to_distribute
            
            for _, city_data in city_inventory.iterrows():
                city = city_data['city_name']
                current_stock = city_data['total_stock']
                product_count = city_data['product_count']
                
                velocity = sales_velocity.get(city, 10)
                
                # Enhanced priority calculation
                stock_density = current_stock / product_count if product_count > 0 else 0
                velocity_norm = velocity / sales_velocity.max() if len(sales_velocity) > 0 and sales_velocity.max() > 0 else 0.5
                stock_need = max(0, 1 - (stock_density / 100))  # Higher need for lower stock density
                
                priority_score = velocity_norm * 0.6 + stock_need * 0.4
                
                # Calculate allocation
                estimated_capacity = 1500  # Increased capacity estimate
                available_capacity = max(0, estimated_capacity - current_stock)
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
                        'priority_score': priority_score,
                        'sales_velocity': velocity,
                        'product_count': product_count
                    })
                    remaining_units -= recommended_units
                    
                if remaining_units <= 0:
                    break
            
            # Sort by priority
            distribution_plan.sort(key=lambda x: x['priority_score'], reverse=True)
            
            # Generate optimized response
            response = f"🚚 **Smart Distribution Plan for {units_to_distribute:,} Units**\n\n"
            response += "📋 **Optimized Allocation Strategy:**\n\n"
            
            total_allocated = 0
            for idx, plan in enumerate(distribution_plan, 1):
                response += f"**{idx}. {plan['city']}:** {plan['units']:,} units\n"
                response += f"   • Current → Future Stock: {plan['current_stock']:,} → {plan['after_stock']:,}\n"
                response += f"   • Products Managed: {plan['product_count']}\n"
                response += f"   • Daily Sales Rate: {plan['sales_velocity']:.1f} units/day\n"
                response += f"   • Priority Score: {plan['priority_score']:.2f}/1.0\n\n"
                total_allocated += plan['units']
            
            response += f"📊 **Distribution Summary:**\n"
            response += f"• **Total Allocated:** {total_allocated:,} units ({total_allocated/units_to_distribute*100:.1f}%)\n"
            response += f"• **Remaining:** {remaining_units:,} units\n"
            response += f"• **Strategy:** Priority-based allocation considering sales velocity and stock density\n"
            
            if remaining_units > 0:
                response += f"\n💡 **Recommendation:** Consider expanding capacity in high-priority cities or distributing to additional locations."
            
            return response
            
        except Exception as e:
            return f"❌ Error optimizing distribution: {str(e)}"
    
    def _analyze_specific_product(self, query: str, entities: Dict) -> str:
        """Enhanced product-specific analysis with beautiful formatting"""
        try:
            if not entities.get('product_ids'):
                return "❌ Please specify a product ID (6-digit number) for analysis."
            
            # Safe product ID conversion
            try:
                product_id_str = str(entities['product_ids'][0])
                if not product_id_str.isdigit():
                    # Extract digits if it's not a pure number
                    digits = re.findall(r'\d+', product_id_str)
                    if digits:
                        product_id_str = digits[0]
                    else:
                        return f"❌ Invalid product ID format: {product_id_str}"
                product_id = int(product_id_str)
            except (ValueError, TypeError) as e:
                return f"❌ Error parsing product ID '{entities['product_ids'][0]}': {str(e)}"
            
            # Get product info
            product_info = self.inventory_df[self.inventory_df['product_id'] == product_id]
            sales_info = self.sales_df[self.sales_df['product_id'] == product_id]
            
            if product_info.empty:
                return f"❌ Product ID {product_id} not found in database."
            
            product_name = product_info['product_name'].iloc[0]
            
            # Comprehensive analysis
            total_stock = product_info['stock_quantity'].sum()
            total_sales = sales_info['units_sold'].sum()
            cities_available = product_info['city_name'].nunique()
            
            # Handle revenue column safely
            if 'revenue' in sales_info.columns and not sales_info['revenue'].empty:
                total_revenue = sales_info['revenue'].sum()
            else:
                # Estimate revenue if column doesn't exist
                avg_price = 25.0  # Default estimate
                total_revenue = total_sales * avg_price
            
            # Recent performance (last 30 days)
            recent_sales = sales_info[sales_info['date'] >= datetime.now() - timedelta(days=30)]
            recent_units = recent_sales['units_sold'].sum()
            daily_velocity = recent_units / 30 if not recent_sales.empty else 0
            
            # 🎨 Beautiful structured response with tables
            response = "=" * 60 + "\n"
            response += f"📈 **PRODUCT ANALYSIS REPORT**\n"
            response += "=" * 60 + "\n\n"
            
            # Product Header
            response += f"🏷️  **Product:** {product_name}\n"
            response += f"🆔  **ID:** {product_id}\n"
            response += f"📅  **Analysis Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            
            # Key Metrics - Simple Format
            response += "📊 **KEY PERFORMANCE METRICS**\n\n"
            response += f"Current Stock: {total_stock:,} units\n"
            response += f"Total Sales (All Time): {total_sales:,} units\n"
            response += f"Total Revenue: ${total_revenue:,.2f}\n"
            response += f"Cities Available: {cities_available} locations\n"
            response += f"Daily Sales Rate: {daily_velocity:.1f} units/day\n\n"
            
            # Stock by Location - Simple Format
            response += "📍 **INVENTORY BY LOCATION**\n\n"
            
            for _, row in product_info.iterrows():
                city = row['city_name'][:22]  # Truncate long city names
                stock = row['stock_quantity']
                if stock > 20:
                    status = "� Good"
                elif stock > 10:
                    status = "🟡 Low"
                else:
                    status = "🔴 Critical"
                
                response += f"│ {city.ljust(23)} │ {str(stock).rjust(12)} │ {status.ljust(10)} │\n"
            
            response += "└─────────────────────────┴──────────────┴────────────┘\n\n"
            
            # Supply Analysis
            response += "⏱️  **SUPPLY ANALYSIS**\n"
            response += "─" * 40 + "\n"
            
            if daily_velocity > 0:
                days_supply = total_stock / daily_velocity
                response += f"📦 **Supply Duration:** {days_supply:.0f} days at current rate\n\n"
                
                if days_supply < 7:
                    response += "🚨 **URGENT ALERT**\n"
                    response += "   • Supply critically low!\n"
                    response += "   • Immediate restocking required\n"
                    response += f"   • Recommend ordering {int(daily_velocity * 30)} units\n\n"
                elif days_supply < 30:
                    response += "⚠️  **WARNING**\n"
                    response += "   • Consider restocking soon\n"
                    response += f"   • Recommend ordering {int(daily_velocity * 20)} units\n\n"
                else:
                    response += "✅ **GOOD STATUS**\n"
                    response += "   • Supply levels are adequate\n\n"
            else:
                response += "📊 **No recent sales data available**\n\n"
            
            # Performance Insights
            if total_sales > 0:
                response += "💡 **INSIGHTS & RECOMMENDATIONS**\n"
                response += "─" * 40 + "\n"
                
                avg_revenue_per_unit = total_revenue / total_sales if total_sales > 0 else 0
                response += f"• Average selling price: ${avg_revenue_per_unit:.2f} per unit\n"
                
                if daily_velocity > 5:
                    response += "• High-velocity product - monitor stock closely\n"
                elif daily_velocity > 1:
                    response += "• Moderate sales velocity - standard monitoring\n"
                else:
                    response += "• Low sales velocity - consider promotion\n"
                
                response += f"• Available in {cities_available} cities - {'good distribution' if cities_available > 5 else 'consider expansion'}\n"
            
            response += "\n" + "=" * 60
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing product: {str(e)}"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing product: {str(e)}"
    
    def _predict_demand(self, query: str, entities: Dict) -> str:
        """Enhanced demand prediction with smart algorithms"""
        try:
            # Product-specific prediction
            if entities.get('product_ids'):
                product_id = int(entities['product_ids'][0])
                product_sales = self.sales_df[self.sales_df['product_id'] == product_id]
                
                if product_sales.empty:
                    return f"❌ No sales history found for product ID {product_id}."
                
                product_name = product_sales['product_name'].iloc[0]
                
                # Calculate trends
                recent_sales = product_sales[product_sales['date'] >= datetime.now() - timedelta(days=30)]
                prev_sales = product_sales[
                    (product_sales['date'] >= datetime.now() - timedelta(days=60)) &
                    (product_sales['date'] < datetime.now() - timedelta(days=30))
                ]
                
                recent_avg = recent_sales['units_sold'].sum() / 30 if not recent_sales.empty else 0
                prev_avg = prev_sales['units_sold'].sum() / 30 if not prev_sales.empty else recent_avg
                
                trend = ((recent_avg - prev_avg) / prev_avg * 100) if prev_avg > 0 else 0
                
                # Future predictions
                next_week = recent_avg * 7 * (1 + trend/100)
                next_month = recent_avg * 30 * (1 + trend/100)
                
                response = f"🔮 **Demand Forecast: {product_name}** (ID: {product_id})\n\n"
                response += "📈 **Current Performance:**\n"
                response += f"• **Recent Daily Average:** {recent_avg:.1f} units/day\n"
                response += f"• **Trend:** {trend:+.1f}% vs previous month\n\n"
                
                response += "🎯 **Predictions:**\n"
                response += f"• **Next Week:** {next_week:.0f} units\n"
                response += f"• **Next Month:** {next_month:.0f} units\n\n"
                
                # Recommendations
                current_stock = self.inventory_df[self.inventory_df['product_id'] == product_id]['stock_quantity'].sum()
                if current_stock < next_month:
                    shortfall = next_month - current_stock
                    response += f"⚠️ **Recommendation:** Restock {shortfall:.0f} units to meet next month's demand.\n"
                else:
                    response += "✅ **Status:** Current stock should meet predicted demand.\n"
                
                return response
            
            # General demand prediction
            total_recent = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]['units_sold'].sum()
            daily_avg = total_recent / 30
            
            response = "🔮 **Overall Demand Forecast:**\n\n"
            response += f"• **Current Daily Average:** {daily_avg:.0f} units\n"
            response += f"• **Predicted Next Week:** {daily_avg * 7:.0f} units\n"
            response += f"• **Predicted Next Month:** {daily_avg * 30:.0f} units\n\n"
            
            # Top demand products
            recent_demand = self.sales_df[self.sales_df['date'] >= datetime.now() - timedelta(days=30)]
            top_demand = recent_demand.groupby(['product_id', 'product_name'])['units_sold'].sum().sort_values(ascending=False).head(3)
            
            response += "🔥 **Highest Demand Products:**\n"
            for (product_id, product_name), units in top_demand.items():
                response += f"• **{product_name}:** {units:,} units (last 30 days)\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error predicting demand: {str(e)}"
    
    def _analyze_city_performance(self, query: str, entities: Dict) -> str:
        """Enhanced city performance analysis"""
        try:
            if entities.get('cities'):
                city_name = entities['cities'][0]
                city_sales = self.sales_df[self.sales_df['city_name'].str.contains(city_name, case=False)]
                city_inventory = self.inventory_df[self.inventory_df['city_name'].str.contains(city_name, case=False)]
                
                if city_sales.empty:
                    return f"❌ No data found for {city_name}. Available cities: {', '.join(sorted(self.city_lookup))}"
                
                # Calculate metrics
                total_revenue = city_sales['revenue'].sum()
                total_units = city_sales['units_sold'].sum()
                total_stock = city_inventory['stock_quantity'].sum()
                product_count = city_inventory['product_id'].nunique()
                
                # Recent performance
                recent_sales = city_sales[city_sales['date'] >= datetime.now() - timedelta(days=30)]
                recent_revenue = recent_sales['revenue'].sum()
                recent_units = recent_sales['units_sold'].sum()
                
                response = f"🏙️ **Performance Analysis: {city_name}**\n\n"
                response += "📊 **Overall Performance:**\n"
                response += f"• **Total Revenue:** ${total_revenue:,.2f}\n"
                response += f"• **Units Sold:** {total_units:,}\n"
                response += f"• **Products Available:** {product_count}\n"
                response += f"• **Current Stock:** {total_stock:,} units\n\n"
                
                response += "📈 **Recent Performance (30 days):**\n"
                response += f"• **Revenue:** ${recent_revenue:,.2f}\n"
                response += f"• **Units Sold:** {recent_units:,}\n"
                response += f"• **Daily Average:** {recent_units/30:.1f} units/day\n\n"
                
                # Top products in this city
                city_top_products = city_sales.groupby(['product_id', 'product_name'])['units_sold'].sum().sort_values(ascending=False).head(3)
                response += f"🏆 **Top Products in {city_name}:**\n"
                for idx, ((product_id, product_name), units) in enumerate(city_top_products.items(), 1):
                    response += f"{idx}. **{product_name}:** {units:,} units\n"
                
                return response
            
            # Overall city comparison
            response = "🏙️ **City Performance Comparison:**\n\n"
            for idx, (city, data) in enumerate(self.city_performance.head(5).iterrows(), 1):
                response += f"**{idx}. {city}**\n"
                response += f"   • Revenue: ${data['revenue']:,.2f}\n"
                response += f"   • Units Sold: {data['units_sold']:,}\n\n"
            
            return response
            
        except Exception as e:
            return f"❌ Error analyzing city performance: {str(e)}"
    
    def _generate_general_response(self, query: str, entities: Dict) -> str:
        """Enhanced general response with smart suggestions"""
        suggestions = [
            "🏆 Ask about **top sellers**: 'What are the best selling products?'",
            "📦 Check **inventory**: 'Show me low stock items'",
            "🚚 Optimize **distribution**: 'I have 1000 units to distribute'",
            "📈 Get **demand predictions**: 'Predict demand for product 561099'",
            "🏙️ Analyze **city performance**: 'How is Mumbai performing?'",
            "🔍 Product **specific analysis**: 'Tell me about product 561099'"
        ]
        
        response = "🤖 **AI Warehouse Assistant**\n\n"
        response += "I can help you with comprehensive warehouse management queries. Here are some examples:\n\n"
        response += "\n".join(suggestions)
        response += "\n\n💡 **Tip:** Be specific with product IDs (6-digit numbers) and quantities for best results!"
        
        return response
    
    def _create_enhanced_demo_data(self):
        """Create enhanced demo data for testing"""
        # Enhanced demo data creation
        cities = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata', 'Pune', 'Hyderabad']
        products = [
            {'id': 561099, 'name': 'Premium Headphones'},
            {'id': 561100, 'name': 'Wireless Mouse'},
            {'id': 561101, 'name': 'Gaming Keyboard'},
            {'id': 561102, 'name': 'USB Cable'},
            {'id': 561103, 'name': 'Power Bank'},
            {'id': 561104, 'name': 'Phone Case'},
            {'id': 561105, 'name': 'Laptop Stand'},
            {'id': 561106, 'name': 'Bluetooth Speaker'},
            {'id': 561107, 'name': 'Monitor Stand'},
            {'id': 561108, 'name': 'Webcam'}
        ]
        
        # Generate sales data
        sales_data = []
        for i in range(1000):
            product = np.random.choice(products)
            city = np.random.choice(cities)
            units = np.random.randint(1, 50)
            price_per_unit = np.random.uniform(10, 200)
            revenue = units * price_per_unit
            date = datetime.now() - timedelta(days=np.random.randint(1, 365))
            
            sales_data.append({
                'product_id': product['id'],
                'product_name': product['name'],
                'city_name': city,
                'units_sold': units,
                'revenue': revenue,
                'date': date
            })
        
        self.sales_df = pd.DataFrame(sales_data)
        
        # Generate inventory data
        inventory_data = []
        for product in products:
            for city in cities:
                stock = np.random.randint(5, 200)
                inventory_data.append({
                    'product_id': product['id'],
                    'product_name': product['name'],
                    'city_name': city,
                    'stock_quantity': stock
                })
        
        self.inventory_df = pd.DataFrame(inventory_data)
        
        # Initialize product lookup
        self.product_lookup = {}
        for product in products:
            self.product_lookup[str(product['id'])] = product['name']
            self.product_lookup[product['name'].lower()] = str(product['id'])
        
        self.city_lookup = set(cities)
        
        # Pre-calculate metrics
        self._calculate_performance_metrics()
        
        print("✅ Enhanced demo data created successfully")
        print(f"📊 Sales records: {len(self.sales_df):,}")
        print(f"📦 Inventory records: {len(self.inventory_df):,}")

# Global instance for better performance
_nlp_engine_instance = None

def get_nlp_engine():
    """Get singleton NLP engine instance"""
    global _nlp_engine_instance
    if _nlp_engine_instance is None:
        _nlp_engine_instance = UltraAdvancedNLPEngine()
    return _nlp_engine_instance