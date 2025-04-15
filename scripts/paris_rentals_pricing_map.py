import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import json

# ------------------------------
# 1. Load Neighborhoods Data
# ------------------------------
df_neigh = pd.read_csv("../data/paris_neighborhoods.csv", encoding='utf-8')
df_neigh.rename(columns={df_neigh.columns[0]: 'neigh_id'}, inplace=True)

def convert_geojson(geo_str):
    try:
        geo_dict = json.loads(geo_str)
        return shape(geo_dict)
    except Exception as e:
        print("Error converting geometry:", e)
        return None

df_neigh['geometry'] = df_neigh['geo_shape'].apply(convert_geojson)
gdf_neigh = gpd.GeoDataFrame(df_neigh, geometry='geometry', crs="EPSG:4326")

# ------------------------------
# 2. Load Rentals Data and Filter
# ------------------------------
df_rentals = pd.read_csv("../data/paris_rentals.csv", sep=';', encoding='utf-8', on_bad_lines='skip')

# Keep only rows where the first column (0-indexed = 0) is 2024
df_rentals = df_rentals[df_rentals.iloc[:, 0] == 2024]

# ------------------------------
# 3. Parse Coordinates and Rental Price
# ------------------------------
def parse_coords(coord_str):
    try:
        parts = coord_str.split(',')
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return Point(lon, lat)
    except Exception as e:
        print("Error parsing coordinates:", e)
        return None

# Column 13 (0-indexed) has "latitude, longitude" as text
df_rentals['geometry'] = df_rentals.iloc[:, 13].apply(parse_coords)

# Column 7 (0-indexed) has the rental price per square meter
df_rentals['rental_price'] = pd.to_numeric(df_rentals.iloc[:, 7], errors='coerce')

# Drop rows missing geometry or price
df_rentals = df_rentals[df_rentals['geometry'].notnull() & df_rentals['rental_price'].notnull()]

gdf_rentals = gpd.GeoDataFrame(df_rentals, geometry='geometry', crs="EPSG:4326")

# ------------------------------
# 4. Spatial Join
# ------------------------------
gdf_join = gpd.sjoin(gdf_rentals, gdf_neigh, how='left', predicate='within')

# ------------------------------
# 5. Aggregate: Average Rental Price per Neighborhood
# ------------------------------
avg_prices = gdf_join.groupby('neigh_id')['rental_price'].mean().reset_index(name='avg_rental_price')
gdf_neigh = gdf_neigh.merge(avg_prices, on='neigh_id', how='left')
gdf_neigh['avg_rental_price'] = gdf_neigh['avg_rental_price'].fillna(0)

# ------------------------------
# 6. Visualization
# ------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf_neigh.plot(column='avg_rental_price', cmap='YlOrRd', legend=True, ax=ax, edgecolor='black')
ax.set_title("Average Rental Price per Square Meter by Neighborhood in Paris (Only 2024)")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()
