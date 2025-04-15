import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import json

# --------------
# 1. Load Neighborhoods
# --------------
# Assume "paris_neighborhoods.csv" has:
#   - A neighborhood identifier in the first column (e.g. "Numéro du quartier")
#   - A column "geo_shape" holding the GeoJSON-like polygon geometry as a string.
df_neigh = pd.read_csv("../data/paris_neighborhoods.csv", encoding='utf-8')

# Function to convert a GeoJSON string to a Shapely geometry
def convert_geojson(geo_str):
    try:
        geo_dict = json.loads(geo_str)
        return shape(geo_dict)
    except Exception as e:
        print("Error converting geometry:", e)
        return None

df_neigh['geometry'] = df_neigh['geo_shape'].apply(convert_geojson)
gdf_neigh = gpd.GeoDataFrame(df_neigh, geometry='geometry', crs="EPSG:4326")

# --------------
# 2. Load Airbnb Listings
# --------------
# Assume "paris_airbnb.csv" uses semicolon as the delimiter.
# We expect that:
#   - Latitude is in column 6
#   - Longitude is in column 7
df_airbnb = pd.read_csv("../data/paris_airbnb.csv", delimiter=',', on_bad_lines='skip', encoding='utf-8')

# (Optional) Print DataFrame shape and columns for debugging:
print("Airbnb DataFrame shape:", df_airbnb.shape)
print("Airbnb columns:", list(df_airbnb.columns))

# Define a function that creates a Point from a row.
def create_point(row):
    # Check if the row has at least 7 columns
    if len(row) >= 8:
        try:
            # Note: Point(longitude, latitude)
            return Point(row.iloc[7], row.iloc[6])
        except Exception as e:
            print("Error creating point for row:", e)
            return None
    else:
        return None

# Create a new column "geometry". Rows that fail will be None.
df_airbnb['geometry'] = df_airbnb.apply(create_point, axis=1)

# Filter out rows where point creation failed
df_airbnb = df_airbnb[df_airbnb['geometry'].notnull()]
gdf_airbnb = gpd.GeoDataFrame(df_airbnb, geometry='geometry', crs="EPSG:4326")

# --------------
# 3. Point in Polygon (Spatial Join)
# --------------
# Each Airbnb listing (point) is assigned the attributes of the neighborhood (polygon) in which it falls.
gdf_join = gpd.sjoin(gdf_airbnb, gdf_neigh, how='left', predicate='within')

# --------------
# 4. Count Airbnb Listings per Neighborhood
# --------------
# We assume the neighborhood identifier is in the first column of paris_neighborhoods.csv.
# For clarity, let’s assign this column a name. If it's not already named, you might rename it.
neigh_id_col = gdf_neigh.columns[0]  # assuming first column contains the neighborhood identifier

# Group by the neighborhood identifier and count the number of listings.
counts = gdf_join.groupby(neigh_id_col).size().reset_index(name="airbnb_count")

# Merge these counts back into the neighborhoods GeoDataFrame.
gdf_neigh = gdf_neigh.merge(counts, on=neigh_id_col, how='left')
gdf_neigh['airbnb_count'] = gdf_neigh['airbnb_count'].fillna(0)

# --------------
# 5. Visualization
# --------------
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
# Plot the neighborhoods colored by the number of Airbnb listings.
gdf_neigh.plot(column='airbnb_count', cmap='OrRd', legend=True, ax=ax, edgecolor='black')
# Overlay the Airbnb points.
#gdf_airbnb.plot(ax=ax, color='blue', markersize=5, alpha=0.5)

ax.set_title("Airbnb Listings by Neighborhood in Paris")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")
plt.show()
