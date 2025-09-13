"""
Data Exploration and Analysis for Quick Commerce Demand Prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data():
    """Load and perform basic exploration of sales and inventory data"""
    
    # Load datasets
    print("Loading datasets...")
    sales_df = pd.read_csv('dataset/salesData.csv')
    inventory_df = pd.read_csv('dataset/inventoryData.csv')
    
    print(f"Sales data shape: {sales_df.shape}")
    print(f"Inventory data shape: {inventory_df.shape}")
    
    # Basic info about sales data
    print("\n=== SALES DATA ANALYSIS ===")
    print("Columns:", sales_df.columns.tolist())
    print("\nData types:")
    print(sales_df.dtypes)
    
    # Convert date column
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Convert numeric columns
    sales_df['gross_merchandise_value'] = pd.to_numeric(sales_df['gross_merchandise_value'], errors='coerce')
    sales_df['gross_selling_value'] = pd.to_numeric(sales_df['gross_selling_value'], errors='coerce')
    
    print(f"\nDate range: {sales_df['date'].min()} to {sales_df['date'].max()}")
    print(f"Unique products: {sales_df['product_id'].nunique()}")
    print(f"Unique cities: {sales_df['city_name'].nunique()}")
    print(f"Unique categories: {sales_df['category'].nunique()}")
    
    # Sales summary
    print(f"\nTotal units sold: {sales_df['units_sold'].sum():,}")
    print(f"Average units per transaction: {sales_df['units_sold'].mean():.2f}")
    print(f"Total GMV: ₹{sales_df['gross_merchandise_value'].sum():,.2f}")
    
    # Top cities by sales
    print("\nTop 10 cities by units sold:")
    city_sales = sales_df.groupby('city_name')['units_sold'].sum().sort_values(ascending=False)
    print(city_sales.head(10))
    
    # Top products by sales
    print("\nTop 10 products by units sold:")
    product_sales = sales_df.groupby(['product_id', 'product_name'])['units_sold'].sum().sort_values(ascending=False)
    print(product_sales.head(10))
    
    # Basic info about inventory data
    print("\n=== INVENTORY DATA ANALYSIS ===")
    print("Columns:", inventory_df.columns.tolist())
    
    print(f"\nUnique products in inventory: {inventory_df['product_id'].nunique()}")
    print(f"Unique cities in inventory: {inventory_df['city_name'].nunique()}")
    print(f"Total stock quantity: {inventory_df['stock_quantity'].sum():,}")
    
    # Stock distribution by city
    print("\nStock distribution by top 10 cities:")
    city_stock = inventory_df.groupby('city_name')['stock_quantity'].sum().sort_values(ascending=False)
    print(city_stock.head(10))
    
    # Products with zero stock
    zero_stock = inventory_df[inventory_df['stock_quantity'] == 0]
    print(f"\nProducts with zero stock: {len(zero_stock)} out of {len(inventory_df)} ({len(zero_stock)/len(inventory_df)*100:.1f}%)")
    
    return sales_df, inventory_df

def analyze_demand_patterns(sales_df):
    """Analyze demand patterns for prediction model"""
    
    print("\n=== DEMAND PATTERN ANALYSIS ===")
    
    # Daily sales trend
    daily_sales = sales_df.groupby('date').agg({
        'units_sold': 'sum',
        'gross_merchandise_value': 'sum'
    }).reset_index()
    
    print(f"Average daily sales: {daily_sales['units_sold'].mean():.2f} units")
    print(f"Peak daily sales: {daily_sales['units_sold'].max()} units")
    print(f"Minimum daily sales: {daily_sales['units_sold'].min()} units")
    
    # City-wise demand patterns
    city_demand = sales_df.groupby(['city_name', 'date'])['units_sold'].sum().reset_index()
    
    # Product-wise demand patterns
    product_demand = sales_df.groupby(['product_id', 'date'])['units_sold'].sum().reset_index()
    
    # Weekly patterns
    sales_df['day_of_week'] = sales_df['date'].dt.day_name()
    weekly_pattern = sales_df.groupby('day_of_week')['units_sold'].sum()
    print(f"\nWeekly demand pattern:")
    print(weekly_pattern)
    
    return daily_sales, city_demand, product_demand

def create_features_for_prediction(sales_df, inventory_df):
    """Create features for demand prediction model"""
    
    print("\n=== FEATURE ENGINEERING ===")
    
    # Convert product_id to consistent type
    inventory_df['product_id'] = pd.to_numeric(inventory_df['product_id'], errors='coerce')
    
    # Merge sales and inventory data
    merged_df = sales_df.merge(
        inventory_df[['product_id', 'city_name', 'stock_quantity']], 
        on=['product_id', 'city_name'], 
        how='left'
    )
    
    # Create time-based features
    merged_df['day_of_week'] = merged_df['date'].dt.dayofweek
    merged_df['month'] = merged_df['date'].dt.month
    merged_df['day_of_month'] = merged_df['date'].dt.day
    merged_df['week_of_year'] = merged_df['date'].dt.isocalendar().week
    
    # Create lag features (previous day sales)
    city_product_daily = merged_df.groupby(['city_name', 'product_id', 'date'])['units_sold'].sum().reset_index()
    city_product_daily = city_product_daily.sort_values(['city_name', 'product_id', 'date'])
    
    # Calculate rolling averages
    city_product_daily['sales_lag_1'] = city_product_daily.groupby(['city_name', 'product_id'])['units_sold'].shift(1)
    city_product_daily['sales_rolling_7'] = city_product_daily.groupby(['city_name', 'product_id'])['units_sold'].rolling(7).mean().reset_index(0, drop=True)
    city_product_daily['sales_rolling_30'] = city_product_daily.groupby(['city_name', 'product_id'])['units_sold'].rolling(30).mean().reset_index(0, drop=True)
    
    # Price features
    merged_df['discount_percentage'] = ((merged_df['mrp'] - merged_df['selling_price']) / merged_df['mrp']) * 100
    
    print(f"Feature engineered dataset shape: {merged_df.shape}")
    print("Created features:")
    print("- Time-based: day_of_week, month, day_of_month, week_of_year")
    print("- Lag features: sales_lag_1, sales_rolling_7, sales_rolling_30")
    print("- Price features: discount_percentage")
    print("- Stock features: stock_quantity")
    
    return merged_df, city_product_daily

def main():
    """Main function to run data exploration"""
    
    print("Quick Commerce Demand Prediction - Data Exploration")
    print("=" * 60)
    
    # Load and explore data
    sales_df, inventory_df = load_and_explore_data()
    
    # Analyze demand patterns
    daily_sales, city_demand, product_demand = analyze_demand_patterns(sales_df)
    
    # Create features
    merged_df, city_product_daily = create_features_for_prediction(sales_df, inventory_df)
    
    # Save processed data for model training
    merged_df.to_csv('processed_sales_data.csv', index=False)
    city_product_daily.to_csv('city_product_daily_sales.csv', index=False)
    
    print(f"\nData exploration complete!")
    print(f"Processed data saved to 'processed_sales_data.csv'")
    print(f"Daily aggregated data saved to 'city_product_daily_sales.csv'")

if __name__ == "__main__":
    main()