import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import json

# ============================================================================
# Step 1: Load rentals and extract unique neighborhoods (in memory, no CSV)
# ============================================================================
df_rentals_initial = pd.read_csv("../data/paris_rentals.csv", 
                                 delimiter=';', 
                                 on_bad_lines='skip', 
                                 encoding='utf-8')

# Drop duplicates based on “Numéro du quartier”
df_neigh = df_rentals_initial.drop_duplicates(subset="Numéro du quartier")

# Keep only the neighborhood ID and its GeoJSON geometry
df_neigh = df_neigh[["Numéro du quartier", "geo_shape"]]
print(f"Number of unique neighborhoods: {len(df_neigh)}")

# Rename for convenience
df_neigh.rename(columns={"Numéro du quartier": "neigh_id"}, inplace=True)

# Convert the geo_shape (GeoJSON) to Shapely geometry
def convert_geojson(geo_str):
    try:
        geo_dict = json.loads(geo_str)
        return shape(geo_dict)
    except Exception as e:
        print("Error converting geometry:", e)
        return None

df_neigh["geometry"] = df_neigh["geo_shape"].apply(convert_geojson)
gdf_neigh = gpd.GeoDataFrame(df_neigh, geometry="geometry", crs="EPSG:4326")

# ============================================================================
# Step 2: Load Airbnb listings and compute Airbnb density by neighborhood
# ============================================================================
df_airbnb = pd.read_csv("../data/paris_airbnb.csv", 
                        delimiter=',', 
                        on_bad_lines='skip', 
                        encoding='utf-8')

print(f"Original Airbnb data rows: {len(df_airbnb)}")
print("Airbnb DataFrame shape:", df_airbnb.shape)
print("Airbnb columns:", list(df_airbnb.columns))

# Create geometry points from columns 6 (latitude) and 7 (longitude)
def create_point(row):
    # Ensure the row has at least 8 columns
    if len(row) >= 8:
        try:
            lon, lat = row.iloc[7], row.iloc[6]
            return Point(lon, lat)
        except Exception as e:
            print("Error creating point for row:", e)
            return None
    else:
        return None

df_airbnb['geometry'] = df_airbnb.apply(create_point, axis=1)
df_airbnb = df_airbnb[df_airbnb['geometry'].notnull()]  # filter out invalid points

gdf_airbnb = gpd.GeoDataFrame(df_airbnb, geometry='geometry', crs="EPSG:4326")

# Spatial join with neighborhoods to identify which listing falls in which neighborhood
gdf_join_airbnb = gpd.sjoin(gdf_airbnb, gdf_neigh, how='left', predicate='within')

# Count the number of listings per neighborhood
counts = gdf_join_airbnb.groupby('neigh_id').size().reset_index(name='airbnb_count')
gdf_neigh = gdf_neigh.merge(counts, on='neigh_id', how='left')
gdf_neigh['airbnb_count'] = gdf_neigh['airbnb_count'].fillna(0).astype(int)

# Calculate area in km² (by projecting to EPSG:3857) and density
gdf_neigh['area_km2'] = gdf_neigh.to_crs(epsg=3857).area / 1e6
gdf_neigh['airbnb_density'] = gdf_neigh['airbnb_count'] / gdf_neigh['area_km2']
print(f"Total number of Airbnb listings in Paris: {len(gdf_airbnb)}")

# Plot Airbnb density
fig, ax = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column='airbnb_density', cmap='OrRd', legend=True, ax=ax, edgecolor='black')
ax.set_title("Airbnb Density (listings per km²) by Neighborhood in Paris (2024)")
plt.show()

# Show top 5 neighborhoods by Airbnb density
print("\nTop 5 neighborhoods by Airbnb density:")
density_cols = ['neigh_id', 'airbnb_count', 'area_km2', 'airbnb_density']
print(gdf_neigh.sort_values('airbnb_density', ascending=False).head(5)[density_cols])

# ============================================================================
# Step 3: Filter rentals for year 2024 and compute average price by neighborhood
# ============================================================================
# Keep only rows where the first column == 2024
df_rentals_2024 = df_rentals_initial[df_rentals_initial.iloc[:, 0] == 2024].copy()

# Parse "latitude, longitude" from column index 13
def parse_coords(coord_str):
    try:
        lat_str, lon_str = coord_str.split(',')
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return Point(lon, lat)
    except Exception as e:
        print("Error parsing coordinates:", e)
        return None

df_rentals_2024['geometry'] = df_rentals_2024.iloc[:, 13].apply(parse_coords)

# Rental price is in column index 7
df_rentals_2024['rental_price'] = pd.to_numeric(df_rentals_2024.iloc[:, 7], errors='coerce')

# Drop rows missing geometry or rental price
df_rentals_2024 = df_rentals_2024[df_rentals_2024['geometry'].notnull() & 
                                  df_rentals_2024['rental_price'].notnull()]

gdf_rentals_2024 = gpd.GeoDataFrame(df_rentals_2024, geometry='geometry', crs="EPSG:4326")

# Spatial join to find which 2024 rentals fall in which neighborhood
gdf_join_rentals_2024 = gpd.sjoin(gdf_rentals_2024, gdf_neigh, how='left', predicate='within')

# Average rental price per neighborhood
avg_prices_2024 = gdf_join_rentals_2024.groupby('neigh_id')['rental_price'].mean().reset_index(name='avg_rental_price')
gdf_neigh = gdf_neigh.merge(avg_prices_2024, on='neigh_id', how='left')
gdf_neigh['avg_rental_price'] = gdf_neigh['avg_rental_price'].fillna(0)

# Plot the average rental price (2024)
fig2, ax2 = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column='avg_rental_price', cmap='YlOrRd', legend=True, ax=ax2, edgecolor='black')
ax2.set_title("Average Rental Price per Square Meter by Neighborhood in Paris (2024)")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.show()


# -------------------------------
# Step 4: Compute 2019 Average Rental Price
# -------------------------------
# Filter rentals for year 2019
df_rentals_2019 = df_rentals_initial[df_rentals_initial.iloc[:, 0] == 2019].copy()

# Parse "latitude, longitude" from column index 13
df_rentals_2019['geometry'] = df_rentals_2019.iloc[:, 13].apply(parse_coords)

# Rental price is in column index 7 (same as for 2024)
df_rentals_2019['rental_price'] = pd.to_numeric(df_rentals_2019.iloc[:, 7], errors='coerce')

# Drop rows missing geometry or rental price
df_rentals_2019 = df_rentals_2019[df_rentals_2019['geometry'].notnull() & 
                                  df_rentals_2019['rental_price'].notnull()]

# Create a GeoDataFrame for 2019 rentals
gdf_rentals_2019 = gpd.GeoDataFrame(df_rentals_2019, geometry='geometry', crs="EPSG:4326")

# Spatial join to find which 2019 rentals fall in which neighborhood
gdf_join_rentals_2019 = gpd.sjoin(gdf_rentals_2019, gdf_neigh, how='left', predicate='within')

# Compute average rental price per neighborhood for 2019
avg_prices_2019 = gdf_join_rentals_2019.groupby('neigh_id')['rental_price'].mean().reset_index(name='avg_rental_price_2019')

# Merge the 2019 average prices into the gdf_neigh GeoDataFrame
gdf_neigh = gdf_neigh.merge(avg_prices_2019, on='neigh_id', how='left')
gdf_neigh['avg_rental_price_2019'] = gdf_neigh['avg_rental_price_2019'].fillna(0)

# -------------------------------
# Step 5: Compute the Price Increase (2024 - 2019)
# -------------------------------
# 'avg_rental_price' in gdf_neigh corresponds to 2024 rental prices from the previous processing
gdf_neigh['price_increase'] = gdf_neigh['avg_rental_price'] - gdf_neigh['avg_rental_price_2019']

# -------------------------------
# Step 6: Visualize Price Increase & Compare with Airbnb Density
# -------------------------------
# Map: Display neighborhoods with a color gradient reflecting the price increase
fig, ax = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column='price_increase', cmap='RdYlGn', legend=True, ax=ax, edgecolor='black')
ax.set_title("Rental Price Increase (2024 vs. 2019) by Neighborhood in Paris")
plt.show()

# Optional: Scatter plot to compare Airbnb density and price increase
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(gdf_neigh['airbnb_density'], gdf_neigh['price_increase'], alpha=0.7, edgecolors='w')
ax2.set_xlabel("Airbnb Density (listings per km²)")
ax2.set_ylabel("Price Increase (2024 - 2019) [€/m²]")
ax2.set_title("Airbnb Density vs. Rental Price Increase")
plt.show()
