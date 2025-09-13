"""
Demand Prediction Module for Quick Commerce Agentic AI System
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import sqlite3
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DemandPredictor:
    """
    Demand prediction model for Quick Commerce operations
    """
    
    def __init__(self):
        self.model = None
        self.label_encoders = {}
        self.feature_columns = []
        self.is_trained = False
        
    def load_and_preprocess_data(self):
        """Load and preprocess sales and inventory data"""
        
        print("Loading and preprocessing data...")
        
        # Load datasets
        sales_df = pd.read_csv('dataset/salesData.csv')
        inventory_df = pd.read_csv('dataset/inventoryData.csv')
        
        # Convert date column
        sales_df['date'] = pd.to_datetime(sales_df['date'])
        
        # Convert numeric columns
        sales_df['gross_merchandise_value'] = pd.to_numeric(sales_df['gross_merchandise_value'], errors='coerce')
        sales_df['gross_selling_value'] = pd.to_numeric(sales_df['gross_selling_value'], errors='coerce')
        
        # Convert product_id to consistent type
        inventory_df['product_id'] = pd.to_numeric(inventory_df['product_id'], errors='coerce')
        
        # Aggregate inventory by city and product (sum across stores)
        inventory_agg = inventory_df.groupby(['product_id', 'city_name'])['stock_quantity'].sum().reset_index()
        
        # Create daily aggregated sales data
        daily_sales = sales_df.groupby(['date', 'city_name', 'product_id']).agg({
            'units_sold': 'sum',
            'mrp': 'mean',
            'selling_price': 'mean',
            'category': 'first',
            'sub_category': 'first'
        }).reset_index()
        
        # Merge with inventory data
        merged_df = daily_sales.merge(
            inventory_agg,
            on=['product_id', 'city_name'],
            how='left'
        )
        
        # Fill missing stock quantities with 0
        merged_df['stock_quantity'] = merged_df['stock_quantity'].fillna(0)
        
        print(f"Merged dataset shape: {merged_df.shape}")
        return merged_df
    
    def create_features(self, df):
        """Create features for machine learning model"""
        
        print("Creating features...")
        
        # Sort by date for lag features
        df = df.sort_values(['city_name', 'product_id', 'date']).reset_index(drop=True)
        
        # Time-based features
        df['day_of_week'] = df['date'].dt.dayofweek
        df['month'] = df['date'].dt.month
        df['day_of_month'] = df['date'].dt.day
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Price features
        df['discount_percentage'] = ((df['mrp'] - df['selling_price']) / df['mrp']) * 100
        df['price_ratio'] = df['selling_price'] / df['mrp']
        
        # Create lag features using a more robust approach
        print("Creating lag features...")
        
        # Initialize lag columns
        df['sales_lag_1'] = np.nan
        df['sales_lag_7'] = np.nan
        df['sales_avg_7'] = np.nan
        df['sales_avg_30'] = np.nan
        
        # Group by city and product to create lag features
        for (city, product), group in df.groupby(['city_name', 'product_id']):
            group = group.sort_values('date')
            indices = group.index
            
            # 1-day lag
            df.loc[indices, 'sales_lag_1'] = group['units_sold'].shift(1)
            
            # 7-day lag
            df.loc[indices, 'sales_lag_7'] = group['units_sold'].shift(7)
            
            # 7-day rolling average
            df.loc[indices, 'sales_avg_7'] = group['units_sold'].rolling(window=7, min_periods=1).mean().shift(1)
            
            # 30-day rolling average
            df.loc[indices, 'sales_avg_30'] = group['units_sold'].rolling(window=30, min_periods=1).mean().shift(1)
        
        # Fill NaN values for lag features with 0
        lag_columns = ['sales_lag_1', 'sales_lag_7', 'sales_avg_7', 'sales_avg_30']
        df[lag_columns] = df[lag_columns].fillna(0)
        
        # City and product popularity features
        city_popularity = df.groupby('city_name')['units_sold'].mean()
        product_popularity = df.groupby('product_id')['units_sold'].mean()
        
        df['city_avg_sales'] = df['city_name'].map(city_popularity)
        df['product_avg_sales'] = df['product_id'].map(product_popularity)
        
        print(f"Features created. Dataset shape: {df.shape}")
        return df
    
    def prepare_training_data(self, df):
        """Prepare data for training"""
        
        # Define feature columns
        self.feature_columns = [
            'day_of_week', 'month', 'day_of_month', 'week_of_year', 'is_weekend',
            'mrp', 'selling_price', 'discount_percentage', 'price_ratio',
            'stock_quantity', 'sales_lag_1', 'sales_lag_7', 'sales_avg_7', 'sales_avg_30',
            'city_avg_sales', 'product_avg_sales'
        ]
        
        # Encode categorical features
        categorical_features = ['city_name', 'product_id', 'category', 'sub_category']
        
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            df[f'{feature}_encoded'] = self.label_encoders[feature].fit_transform(df[feature].astype(str))
            self.feature_columns.append(f'{feature}_encoded')
        
        # Prepare feature matrix and target
        X = df[self.feature_columns].fillna(0)
        y = df['units_sold']
        
        print(f"Training data prepared. Features: {len(self.feature_columns)}")
        return X, y, df
    
    def train_model(self, X, y):
        """Train the demand prediction model"""
        
        print("Training demand prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=True
        )
        
        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model Performance:")
        print(f"- Mean Absolute Error: {mae:.2f}")
        print(f"- Mean Squared Error: {mse:.2f}")
        print(f"- R² Score: {r2:.3f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        
        self.is_trained = True
        return mae, mse, r2
    
    def predict_demand(self, city_name, product_id, date=None, stock_quantity=0, db_handler=None):
        """Predict demand for a specific city and product"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        if date is None:
            date = datetime.now()
        
        # Get actual product data from database if available
        if db_handler:
            try:
                # Get product details from sales data
                sales_data = db_handler.get_sales_data(city_name=city_name, product_id=product_id)
                if not sales_data.empty:
                    # Use actual product data
                    avg_mrp = sales_data['mrp'].mean()
                    avg_selling_price = sales_data['selling_price'].mean()
                    category = sales_data['category'].iloc[0]
                    sub_category = sales_data['sub_category'].iloc[0]
                    
                    # Calculate derived features
                    discount_percentage = ((avg_mrp - avg_selling_price) / avg_mrp) * 100
                    price_ratio = avg_selling_price / avg_mrp
                    
                    # Get recent sales for lag features
                    recent_sales = sales_data.tail(30)['units_sold']
                    sales_lag_1 = recent_sales.iloc[-1] if len(recent_sales) >= 1 else 0
                    sales_lag_7 = recent_sales.iloc[-7] if len(recent_sales) >= 7 else 0
                    sales_avg_7 = recent_sales.tail(7).mean() if len(recent_sales) >= 7 else 0
                    sales_avg_30 = recent_sales.mean() if len(recent_sales) > 0 else 0
                    
                    # Get city and product averages
                    city_sales = db_handler.get_sales_data(city_name=city_name)
                    city_avg_sales = city_sales['units_sold'].mean() if not city_sales.empty else 100
                    
                    product_sales = db_handler.get_sales_data(product_id=product_id)
                    product_avg_sales = product_sales['units_sold'].mean() if not product_sales.empty else 50
                    
                else:
                    # Fallback to defaults if no data found
                    avg_mrp, avg_selling_price = 300, 250
                    category, sub_category = 'Baby Care', 'Baby Wipes'
                    discount_percentage, price_ratio = 16.67, 0.83
                    sales_lag_1, sales_lag_7, sales_avg_7, sales_avg_30 = 0, 0, 0, 0
                    city_avg_sales, product_avg_sales = 100, 50
                    
            except Exception as e:
                print(f"Error getting product data: {e}")
                # Fallback to defaults
                avg_mrp, avg_selling_price = 300, 250
                category, sub_category = 'Baby Care', 'Baby Wipes'
                discount_percentage, price_ratio = 16.67, 0.83
                sales_lag_1, sales_lag_7, sales_avg_7, sales_avg_30 = 0, 0, 0, 0
                city_avg_sales, product_avg_sales = 100, 50
        else:
            # Use defaults if no database handler provided
            avg_mrp, avg_selling_price = 300, 250
            category, sub_category = 'Baby Care', 'Baby Wipes'
            discount_percentage, price_ratio = 16.67, 0.83
            sales_lag_1, sales_lag_7, sales_avg_7, sales_avg_30 = 0, 0, 0, 0
            city_avg_sales, product_avg_sales = 100, 50
        
        # Create feature vector for prediction
        features = pd.DataFrame({
            'day_of_week': [date.weekday()],
            'month': [date.month],
            'day_of_month': [date.day],
            'week_of_year': [date.isocalendar().week],
            'is_weekend': [1 if date.weekday() >= 5 else 0],
            'mrp': [avg_mrp],
            'selling_price': [avg_selling_price],
            'discount_percentage': [discount_percentage],
            'price_ratio': [price_ratio],
            'stock_quantity': [stock_quantity],
            'sales_lag_1': [sales_lag_1],
            'sales_lag_7': [sales_lag_7],
            'sales_avg_7': [sales_avg_7],
            'sales_avg_30': [sales_avg_30],
            'city_avg_sales': [city_avg_sales],
            'product_avg_sales': [product_avg_sales],
        })
        
        # Encode categorical features
        categorical_values = {
            'city_name': city_name,
            'product_id': str(product_id),
            'category': category,
            'sub_category': sub_category
        }
        
        for feature, value in categorical_values.items():
            if feature in self.label_encoders:
                try:
                    encoded_value = self.label_encoders[feature].transform([str(value)])[0]
                except ValueError:
                    # Handle unseen categories
                    encoded_value = 0
                features[f'{feature}_encoded'] = [encoded_value]
            else:
                features[f'{feature}_encoded'] = [0]
        
        # Make prediction
        prediction = self.model.predict(features[self.feature_columns])[0]
        return max(0, round(prediction))
    
    def save_model(self, filename='demand_predictor.pkl'):
        """Save the trained model"""
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoders': self.label_encoders,
                'feature_columns': self.feature_columns,
                'is_trained': self.is_trained
            }, f)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='demand_predictor.pkl'):
        """Load a pre-trained model"""
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.label_encoders = data['label_encoders']
            self.feature_columns = data['feature_columns']
            self.is_trained = data['is_trained']
        print(f"Model loaded from {filename}")

def train_demand_predictor():
    """Main function to train the demand prediction model"""
    
    print("Quick Commerce Demand Prediction Model Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = DemandPredictor()
    
    # Load and preprocess data
    df = predictor.load_and_preprocess_data()
    
    # Create features
    df = predictor.create_features(df)
    
    # Prepare training data
    X, y, processed_df = predictor.prepare_training_data(df)
    
    # Train model
    mae, mse, r2 = predictor.train_model(X, y)
    
    # Save model
    predictor.save_model('demand_predictor.pkl')
    
    # Save processed data for reference
    processed_df.to_csv('processed_training_data.csv', index=False)
    
    print(f"\nTraining completed successfully!")
    print(f"Model saved as 'demand_predictor.pkl'")
    
    # Test prediction
    print(f"\nTesting prediction...")
    test_prediction = predictor.predict_demand('delhi', 445285, datetime.now(), 100)
    print(f"Predicted demand for product 445285 in Delhi: {test_prediction} units")
    
    return predictor

if __name__ == "__main__":
    predictor = train_demand_predictor()