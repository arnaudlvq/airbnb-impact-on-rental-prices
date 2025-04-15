import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import json

# --------------
# 1. Load Neighborhoods
# --------------
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
df_airbnb = pd.read_csv("../data/paris_airbnb.csv", delimiter=',', on_bad_lines='skip', encoding='utf-8')
print(f"Original Airbnb data rows: {len(df_airbnb)}")

# Print DataFrame shape and columns for debugging:
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
gdf_join = gpd.sjoin(gdf_airbnb, gdf_neigh, how='left', predicate='within')

# --------------
# 4. Count Airbnb Listings per Neighborhood
# --------------
neigh_id_col = gdf_neigh.columns[0]  # assuming first column contains the neighborhood identifier
neigh_name_col = None

# Try to find a column with neighborhood names (helpful for display)
potential_name_cols = ['nom', 'name', 'n_q', 'nom_quartier', 'nom_qu', 'libelle', 'l_qu']
for col in potential_name_cols:
    if col in gdf_neigh.columns:
        neigh_name_col = col
        break

# Group by the neighborhood identifier and count the number of listings.
counts = gdf_join.groupby(neigh_id_col).size().reset_index(name="airbnb_count")

# Merge these counts back into the neighborhoods GeoDataFrame.
gdf_neigh = gdf_neigh.merge(counts, on=neigh_id_col, how='left')
gdf_neigh['airbnb_count'] = gdf_neigh['airbnb_count'].fillna(0).astype(int)

# Print the total count of Airbnb listings
total_airbnbs = len(gdf_airbnb)
print(f"Total number of Airbnb listings in Paris: {total_airbnbs}")

# --------------
# 5. Visualization
# --------------
# Calculate area in square kilometers and density
gdf_neigh['area_km2'] = gdf_neigh.geometry.to_crs('EPSG:3857').area / 1_000_000
gdf_neigh['density'] = gdf_neigh['airbnb_count'] / gdf_neigh['area_km2']

# Create a plot showing density
fig, ax = plt.subplots(1, 1, figsize=(10, 10))
gdf_neigh.plot(column='density', cmap='OrRd', legend=True, ax=ax, edgecolor='black')
ax.set_title("Airbnb Density (listings per km²) by Neighborhood in Paris")



plt.show()

# Print summary of the densities
print("\nTop 5 neighborhoods by Airbnb density:")
density_cols = [neigh_id_col, 'airbnb_count', 'area_km2', 'density']
print(gdf_neigh.sort_values('density', ascending=False).head(5)[density_cols])