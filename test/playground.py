import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import json

# ------------------------------
# 1. Load Neighborhoods Data
# ------------------------------
# Read the neighborhoods CSV and rename the first column to 'neigh_id'.
df_neigh = pd.read_csv("paris_neighborhoods.csv", encoding='utf-8')
df_neigh.rename(columns={df_neigh.columns[0]: 'neigh_id'}, inplace=True)

# Convert the geo_shape column (GeoJSON-like string) into Shapely geometries.
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
# 2. Load Rentals Data
# ------------------------------
# Load the rentals CSV.
# Note: paris_rentals.csv is semicolon-separated.
df_rentals = pd.read_csv("../data/paris_rentals.csv", sep=';', encoding='utf-8', on_bad_lines='skip')

# ------------------------------
# 3. Parse Coordinates and Rental Price
# ------------------------------
# In column 13 (0-indexed) we have a string like: "48.88504369140323, 2.302909824651906"
def parse_coords(coord_str):
    try:
        parts = coord_str.split(',')
        # Extract latitude and longitude (first and second values)
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        return Point(lon, lat)  # Create Point(longitude, latitude)
    except Exception as e:
        print("Error parsing coordinates:", e)
        return None

# Create a new geometry column by parsing the coordinate string from column index 13.
df_rentals['geometry'] = df_rentals.iloc[:, 13].apply(parse_coords)

# The rental price per square meter is in column 7 (0-indexed).
# Convert it to a numeric column.
df_rentals['rental_price'] = pd.to_numeric(df_rentals.iloc[:, 7], errors='coerce')

# Drop rows with missing geometry or rental price.
df_rentals = df_rentals[df_rentals['geometry'].notnull() & df_rentals['rental_price'].notnull()]
gdf_rentals = gpd.GeoDataFrame(df_rentals, geometry='geometry', crs="EPSG:4326")

# ------------------------------
# 4. Spatial Join (Point in Polygon)
# ------------------------------
# Assign each rental (point) the attributes of the neighborhood polygon that contains it.
gdf_join = gpd.sjoin(gdf_rentals, gdf_neigh, how='left', predicate='within')

# ------------------------------
# 5. Aggregate: Average Rental Price per Neighborhood
# ------------------------------
# Group by the neighborhood identifier ('neigh_id') and compute the average rental price.
avg_prices = gdf_join.groupby('neigh_id')['rental_price'].mean().reset_index(name='avg_rental_price')

# Merge these averages back into the neighborhoods GeoDataFrame.
gdf_neigh = gdf_neigh.merge(avg_prices, on='neigh_id', how='left')
gdf_neigh['avg_rental_price'] = gdf_neigh['avg_rental_price'].fillna(0)

# ------------------------------
# 6. Visualization
# ------------------------------
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# Plot neighborhoods, coloring each by the average rental price per square meter.
gdf_neigh.plot(column='avg_rental_price', cmap='YlOrRd', legend=True, ax=ax, edgecolor='black')
ax.set_title("Average Rental Price per Square Meter by Neighborhood in Paris")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()
