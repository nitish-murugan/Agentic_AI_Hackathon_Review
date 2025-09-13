"""
Database Interface for Quick Commerce Agentic AI System
"""
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import os

class QuickCommerceDB:
    """
    Simple database interface for Quick Commerce operations
    """
    
    def __init__(self, db_path='quickcommerce.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create sales table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                product_id INTEGER,
                product_name TEXT,
                category TEXT,
                sub_category TEXT,
                city_name TEXT,
                units_sold INTEGER,
                mrp REAL,
                selling_price REAL,
                gross_merchandise_value REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create inventory table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                product_name TEXT,
                category TEXT,
                sub_category TEXT,
                city_name TEXT,
                store_name TEXT,
                store_type TEXT,
                stock_quantity INTEGER,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id INTEGER,
                city_name TEXT,
                prediction_date DATE,
                predicted_demand INTEGER,
                actual_demand INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create alerts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                product_id INTEGER,
                city_name TEXT,
                message TEXT,
                severity TEXT,
                status TEXT DEFAULT 'ACTIVE',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print(f"Database initialized: {self.db_path}")
    
    def load_csv_data(self):
        """Load data from CSV files into the database"""
        
        print("Loading CSV data into database...")
        
        # Load sales data
        sales_df = pd.read_csv('dataset/salesData.csv')
        sales_df['date'] = pd.to_datetime(sales_df['date']).dt.date
        
        # Normalize city names to lowercase for consistency
        sales_df['city_name'] = sales_df['city_name'].str.lower()
        
        # Convert numeric columns
        sales_df['gross_merchandise_value'] = pd.to_numeric(sales_df['gross_merchandise_value'], errors='coerce')
        
        # Select relevant columns for database
        sales_cols = ['date', 'product_id', 'product_name', 'category', 'sub_category', 
                     'city_name', 'units_sold', 'mrp', 'selling_price', 'gross_merchandise_value']
        
        # Load inventory data
        inventory_df = pd.read_csv('dataset/inventoryData.csv')
        inventory_df['product_id'] = pd.to_numeric(inventory_df['product_id'], errors='coerce')
        
        # Normalize city names to lowercase for consistency
        inventory_df['city_name'] = inventory_df['city_name'].str.lower()
        
        inventory_cols = ['product_id', 'product_name', 'category', 'sub_category',
                         'city_name', 'store_name', 'store_type', 'stock_quantity']
        
        # Insert into database
        conn = sqlite3.connect(self.db_path)
        
        sales_df[sales_cols].to_sql('sales', conn, if_exists='replace', index=False)
        inventory_df[inventory_cols].to_sql('inventory', conn, if_exists='replace', index=False)
        
        conn.close()
        
        print(f"Loaded {len(sales_df)} sales records and {len(inventory_df)} inventory records")
    
    def get_sales_data(self, city_name=None, product_id=None, start_date=None, end_date=None):
        """Query sales data with filters"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM sales WHERE 1=1"
        params = []
        
        if city_name:
            query += " AND city_name = ?"
            params.append(city_name)
        
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
        
        query += " ORDER BY date DESC"
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_inventory_data(self, city_name=None, product_id=None):
        """Query inventory data with filters"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = "SELECT * FROM inventory WHERE 1=1"
        params = []
        
        if city_name:
            query += " AND city_name = ?"
            params.append(city_name)
        
        if product_id:
            query += " AND product_id = ?"
            params.append(product_id)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def get_low_stock_products(self, threshold=50):
        """Get products with low stock"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT product_id, product_name, city_name, 
                   SUM(stock_quantity) as total_stock
            FROM inventory 
            GROUP BY product_id, city_name
            HAVING total_stock < ?
            ORDER BY total_stock ASC
        '''
        
        df = pd.read_sql_query(query, conn, params=[threshold])
        conn.close()
        
        return df
    
    def get_top_selling_products(self, city_name=None, days=7, limit=10):
        """Get top selling products in the last N days"""
        
        conn = sqlite3.connect(self.db_path)
        
        start_date = (datetime.now() - timedelta(days=days)).date()
        
        query = '''
            SELECT product_id, product_name, city_name,
                   SUM(units_sold) as total_sales
            FROM sales 
            WHERE date >= ?
        '''
        params = [start_date]
        
        if city_name:
            query += " AND city_name = ?"
            params.append(city_name)
        
        query += '''
            GROUP BY product_id, city_name
            ORDER BY total_sales DESC
            LIMIT ?
        '''
        params.append(limit)
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        return df
    
    def add_prediction(self, product_id, city_name, prediction_date, predicted_demand):
        """Add a demand prediction to the database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (product_id, city_name, prediction_date, predicted_demand)
            VALUES (?, ?, ?, ?)
        ''', (product_id, city_name, prediction_date, predicted_demand))
        
        conn.commit()
        conn.close()
    
    def add_alert(self, alert_type, product_id, city_name, message, severity='MEDIUM'):
        """Add an alert to the database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO alerts (alert_type, product_id, city_name, message, severity)
            VALUES (?, ?, ?, ?, ?)
        ''', (alert_type, product_id, city_name, message, severity))
        
        conn.commit()
        conn.close()
    
    def get_active_alerts(self):
        """Get all active alerts"""
        
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM alerts 
            WHERE status = 'ACTIVE' 
            ORDER BY created_at DESC
        '''
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        return df
    
    def get_city_performance(self, days=7):
        """Get city performance metrics"""
        
        conn = sqlite3.connect(self.db_path)
        
        start_date = (datetime.now() - timedelta(days=days)).date()
        
        query = '''
            SELECT city_name,
                   SUM(units_sold) as total_units,
                   SUM(gross_merchandise_value) as total_gmv,
                   COUNT(DISTINCT product_id) as unique_products,
                   AVG(units_sold) as avg_units_per_transaction
            FROM sales 
            WHERE date >= ?
            GROUP BY city_name
            ORDER BY total_units DESC
        '''
        
        df = pd.read_sql_query(query, conn, params=[start_date])
        conn.close()
        
        return df
    
    def update_stock(self, product_id, city_name, new_stock_quantity):
        """Update stock quantity for a product in a city"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE inventory 
            SET stock_quantity = ?, updated_at = CURRENT_TIMESTAMP
            WHERE product_id = ? AND city_name = ?
        ''', (new_stock_quantity, product_id, city_name))
        
        conn.commit()
        conn.close()
    
    def close(self):
        """Close database connection"""
        pass

def main():
    """Initialize database and load data"""
    
    print("Quick Commerce Database Setup")
    print("=" * 30)
    
    # Initialize database
    db = QuickCommerceDB()
    
    # Load CSV data
    db.load_csv_data()
    
    # Test queries
    print("\n=== Testing Database Queries ===")
    
    # Top selling products
    top_products = db.get_top_selling_products(days=30, limit=5)
    print(f"\nTop 5 selling products (last 30 days):")
    print(top_products)
    
    # Low stock products
    low_stock = db.get_low_stock_products(threshold=10)
    print(f"\nProducts with low stock (< 10 units):")
    print(low_stock.head())
    
    # City performance
    city_perf = db.get_city_performance(days=30)
    print(f"\nTop 5 cities by performance:")
    print(city_perf.head())
    
    # Add sample alert
    db.add_alert('LOW_STOCK', 445285, 'delhi', 'Stock below threshold', 'HIGH')
    
    # Get alerts
    alerts = db.get_active_alerts()
    print(f"\nActive alerts:")
    print(alerts)
    
    print(f"\nDatabase setup completed successfully!")

if __name__ == "__main__":
    main()