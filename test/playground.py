import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
import json

use_fine_grid = True

if (use_fine_grid):
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

else:
    # Read the GeoJSON file that contains your neighborhood polygons
    gdf_neigh = gpd.read_file("../data/neighbourhoods.geojson")
    
    # (Optional) If you need to match the original code later on,
    # rename a key property (for example, if the GeoJSON uses "neighbourhood")
    # to "neigh_id" so that downstream merges or joins work as before.
    gdf_neigh.rename(columns={"neighbourhood": "neigh_id"}, inplace=True)
    
    # If needed, drop duplicate geometries based on the neighborhood identifier.
    gdf_neigh = gdf_neigh.drop_duplicates(subset="neigh_id")
    
    # Now you have 'gdf_neigh' in memory with the proper CRS:
    gdf_neigh.set_crs("EPSG:4326", inplace=True)
    
    print(f"Number of unique neighborhoods: {len(gdf_neigh)}")




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



# Keep only rows where the first column == 2024
df_rentals_2024 = df_rentals_initial[df_rentals_initial.iloc[:, 0] == 2024].copy()
df_rentals_2019 = df_rentals_initial[df_rentals_initial.iloc[:, 0] == 2019].copy()

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
df_rentals_2019['geometry'] = df_rentals_2019.iloc[:, 13].apply(parse_coords)


# Rental price is in column index 7
df_rentals_2024['rental_price'] = pd.to_numeric(df_rentals_2024.iloc[:, 7], errors='coerce')
df_rentals_2019['rental_price'] = pd.to_numeric(df_rentals_2019.iloc[:, 7], errors='coerce')

# Drop rows missing geometry or rental price
df_rentals_2024 = df_rentals_2024[df_rentals_2024['geometry'].notnull() & 
                                  df_rentals_2024['rental_price'].notnull()]
df_rentals_2019 = df_rentals_2019[df_rentals_2019['geometry'].notnull() & 
                                  df_rentals_2019['rental_price'].notnull()]

gdf_rentals_2024 = gpd.GeoDataFrame(df_rentals_2024, geometry='geometry', crs="EPSG:4326")
gdf_rentals_2019 = gpd.GeoDataFrame(df_rentals_2019, geometry='geometry', crs="EPSG:4326")

# Spatial join to find which 2024 rentals fall in which neighborhood
gdf_join_rentals_2024 = gpd.sjoin(gdf_rentals_2024, gdf_neigh, how='left', predicate='within')
gdf_join_rentals_2019 = gpd.sjoin(gdf_rentals_2019, gdf_neigh, how='left', predicate='within')


# Average rental price per neighborhood 2024 and 2019
avg_prices_2024 = gdf_join_rentals_2024.groupby('neigh_id')['rental_price'].mean().reset_index(name='avg_rental_price_2024')
gdf_neigh = gdf_neigh.merge(avg_prices_2024, on='neigh_id', how='left')
gdf_neigh['avg_rental_price_2024'] = gdf_neigh['avg_rental_price_2024'].fillna(0)

avg_prices_2019 = gdf_join_rentals_2019.groupby('neigh_id')['rental_price'].mean().reset_index(name='avg_rental_price_2019')
gdf_neigh = gdf_neigh.merge(avg_prices_2019, on='neigh_id', how='left')
gdf_neigh['avg_rental_price_2019'] = gdf_neigh['avg_rental_price_2019'].fillna(0)


# -------------------------------
# Step 5: Compute the Price Increase (2029 -> 2014)
# -------------------------------
gdf_neigh['price_increase'] = gdf_neigh['avg_rental_price_2024'] - gdf_neigh['avg_rental_price_2019']

# Plot the average rental price (2024)
fig2, ax2 = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column='price_increase', cmap='RdYlGn', legend=True, ax=ax2, edgecolor='black')
ax2.set_title("Rental Price Increase (2024 vs. 2019) by Neighborhood in Paris")
ax2.set_xlabel("Longitude")
ax2.set_ylabel("Latitude")
plt.show()



fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(gdf_neigh['price_increase'], gdf_neigh['airbnb_density'], alpha=0.7, edgecolors='w')
ax2.set_xlabel("Airbnb Density (listings per km²)")
ax2.set_ylabel("Price Increase (2024 - 2019) [€/m²]")
ax2.set_title("Airbnb Density vs. Rental Price Increase")
plt.show()




import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold


# ------------------------
# 1. Bias-Variance Analysis using Price Increase as Predictor
# ------------------------
# Predictor x: price_increase, Target y: airbnb_density
x = gdf_neigh['price_increase'].values.reshape(-1, 1)
y = gdf_neigh['airbnb_density'].values

degrees = range(1, 6)  # Evaluate polynomial degrees 1 to 7
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []

# Loop over each polynomial degree to compute cross-validated MSEs
for degree in degrees:
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    x_poly = poly_features.fit_transform(x)
    
    model = LinearRegression()
    cv_results = cross_validate(
        model, x_poly, y, cv=kf, 
        scoring='neg_mean_squared_error', 
        return_train_score=True
    )
    
    # Convert negative MSE scores to positive values
    train_mse = -np.mean(cv_results['train_score'])
    test_mse = -np.mean(cv_results['test_score'])
    
    results.append({
         'degree': degree,
         'train_mse': train_mse,
         'test_mse': test_mse
    })

results_df = pd.DataFrame(results)
print(results_df)

# Select the best model (lowest validation MSE)
best_model_row = results_df.loc[results_df['test_mse'].idxmin()]
best_degree = int(best_model_row['degree'])
print(f"\nBest model: Degree {best_degree}")
print(f"Train MSE (bias indicator): {best_model_row['train_mse']:.4f}")
print(f"Validation MSE (overall error): {best_model_row['test_mse']:.4f}")

# ------------------------
# 2. Plot Bias-Variance Trade-Off
# ------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(results_df['degree'], results_df['train_mse'], marker='o', label='Training MSE (Bias)')
ax.plot(results_df['degree'], results_df['test_mse'], marker='o', label='Validation MSE (Variance)')
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('Mean Squared Error')
ax.set_title('Bias-Variance Trade-Off (Price Increase vs. Airbnb Density)')
ax.legend()
plt.show()

# ------------------------
# 3. Fit and Plot the Best Degree Model
# ------------------------
# Fit the selected best model using the entire dataset
poly_features_best = PolynomialFeatures(degree=best_degree, include_bias=False)
x_poly_best = poly_features_best.fit_transform(x)
best_model = LinearRegression()
best_model.fit(x_poly_best, y)

# Generate a smooth regression curve for plotting
x_range = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_range_poly = poly_features_best.transform(x_range)
y_pred = best_model.predict(x_range_poly)

# Create scatter plot with best-fit polynomial regression curve
fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.scatter(x, y, alpha=0.7, edgecolors='w', label='Neighborhood Data')
ax2.plot(x_range, y_pred, color='red', linestyle='--',
         label=f'Polynomial Regression (degree {best_degree})')
ax2.set_xlabel("Price Increase (2024 - 2019)")
ax2.set_ylabel("Airbnb Density (listings per km²)")
ax2.set_title("Best Polynomial Regression Fit: Airbnb Density vs. Price Increase")
ax2.legend()

plt.show()
