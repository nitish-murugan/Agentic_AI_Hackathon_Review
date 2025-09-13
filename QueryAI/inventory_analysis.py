import pandas as pd

# Load inventory data
df = pd.read_csv('E:/Projects/Agentic_AI_Business_model/dataset/inventoryData.csv')

print('📊 COMPREHENSIVE INVENTORY OVERVIEW')
print('=' * 60)
print()

# Basic statistics
total_products = len(df)
total_stock = df['stock_quantity'].sum()
cities = df['city_name'].nunique()
avg_stock = df['stock_quantity'].mean()

print(f'🏢 **WAREHOUSE SUMMARY:**')
print(f'• Total Products: {total_products:,}')
print(f'• Total Stock: {total_stock:,} units')
print(f'• Operating Cities: {cities}')
print(f'• Average Stock per Product: {avg_stock:.1f} units')
print()

# Stock distribution
print('📦 **STOCK DISTRIBUTION:**')
print(df['stock_quantity'].describe())
print()

# Low stock analysis
low_stock = df[df['stock_quantity'] <= 10]
critical_stock = df[df['stock_quantity'] <= 5]
zero_stock = df[df['stock_quantity'] == 0]

print(f'⚠️ **STOCK ALERTS:**')
print(f'• Low Stock (≤10 units): {len(low_stock):,} items ({len(low_stock)/total_products*100:.1f}%)')
print(f'• Critical Stock (≤5 units): {len(critical_stock):,} items ({len(critical_stock)/total_products*100:.1f}%)')
print(f'• Out of Stock (0 units): {len(zero_stock):,} items ({len(zero_stock)/total_products*100:.1f}%)')
print()

# Top cities by product count
print('🏙️ **TOP CITIES BY PRODUCT AVAILABILITY:**')
city_counts = df['city_name'].value_counts().head(10)
for i, (city, count) in enumerate(city_counts.items(), 1):
    total_city_stock = df[df['city_name'] == city]['stock_quantity'].sum()
    print(f'{i:2d}. {city}: {count:,} products, {total_city_stock:,} total units')
print()

# Category analysis
print('📈 **CATEGORY BREAKDOWN:**')
if 'category' in df.columns:
    category_stats = df.groupby('category').agg({
        'stock_quantity': ['count', 'sum', 'mean']
    }).round(1)
    category_stats.columns = ['Products', 'Total_Stock', 'Avg_Stock']
    print(category_stats.head(10))
print()

# Critical items needing immediate attention
print('🚨 **CRITICAL ITEMS NEEDING IMMEDIATE RESTOCKING:**')
critical_items = df[df['stock_quantity'] <= 5][['product_name', 'city_name', 'stock_quantity']]
if len(critical_items) > 0:
    print(critical_items.head(15).to_string(index=False))
else:
    print('✅ No critical stock issues found!')