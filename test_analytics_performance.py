import time
from prediction_api import DemandPredictionAPI

print("Testing analytics page performance...")

api = DemandPredictionAPI()

# Test restock alerts performance
start = time.time()
alerts = api.get_restock_alerts(days_threshold=7, limit=20)
end = time.time()
print(f'Restock alerts completed in {end-start:.2f} seconds')
print(f'Found {len(alerts.get("restock_alerts", []))} alerts')

# Test underperforming cities performance
start = time.time()
underperforming = api.get_underperforming_cities(days=7)
end = time.time()
print(f'Underperforming cities analysis completed in {end-start:.2f} seconds')
print(f'Found {len(underperforming.get("underperforming_cities", []))} underperforming cities')

print("Analytics performance test completed!")