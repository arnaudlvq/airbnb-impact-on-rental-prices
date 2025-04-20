import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
import seaborn as sns

# ----------------------------------------------------------------------
# PARAMETERS ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
START_YEAR = 2020    # rentals comparison start year
END_YEAR   = 2023    # rentals comparison end year
COMPARE_DIFF = True  # True: x-axis = price change; False: x-axis = price at END_YEAR
XLS_SHEET = 'scraped'  # Excel sheet name for rentals data

# ----------------------------------------------------------------------
# 1. File paths –––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
excel_rentals = "../data/london_rentals.xls"        # rentals data
csv_airbnb    = "../data/london_airbnb.csv"         # Airbnb listings
geojson_neigh = "../data/london_neighbourhoods.geojson"  # neighbourhood shapes

# ----------------------------------------------------------------------
# 2. Load Airbnb listings ––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
df_airbnb = pd.read_csv(csv_airbnb, encoding="utf-8")

def make_point(row):
    try:
        lon, lat = row.iloc[7], row.iloc[6]
        return Point(lon, lat)
    except Exception:
        return None

# build GeoDataFrame
DF = df_airbnb.copy()
DF['geometry'] = DF.apply(make_point, axis=1)
DF = DF[DF['geometry'].notnull()]
gdf_airbnb = gpd.GeoDataFrame(DF, geometry='geometry', crs='EPSG:4326')

# ----------------------------------------------------------------------
# 3. Load and process rental data ––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
# positional column indices in the Excel
YEAR_COL, QUARTER_COL, NEIGH_COL, CATEGORY_COL, PRICE_COL = 0, 1, 3, 4, 6
NEIGH_NAME = 'neighbourhood'

# read raw rentals sheet
raw = pd.read_excel(excel_rentals, sheet_name=XLS_SHEET, header=None)
raw.rename(columns={NEIGH_COL: NEIGH_NAME}, inplace=True)

# filter by years, Q1, all categories
df_filt = raw[(raw.iloc[:, YEAR_COL].isin([START_YEAR, END_YEAR])) &
               (raw.iloc[:, QUARTER_COL]=='Q1') &
               (raw.iloc[:, CATEGORY_COL]=='All categories')].copy()

# parse price and drop NAs
df_filt[PRICE_COL] = pd.to_numeric(df_filt.iloc[:, PRICE_COL], errors='coerce')
df_filt.dropna(subset=[NEIGH_NAME, PRICE_COL], inplace=True)

# average price per neighbourhood per year
avg_start = df_filt[df_filt.iloc[:, YEAR_COL]==START_YEAR]
avg_start = avg_start.groupby(NEIGH_NAME)[PRICE_COL].mean().reset_index(name=f"avg_price_{START_YEAR}")
avg_end = df_filt[df_filt.iloc[:, YEAR_COL]==END_YEAR]
avg_end = avg_end.groupby(NEIGH_NAME)[PRICE_COL].mean().reset_index(name=f"avg_price_{END_YEAR}")
# merge average prices
df_rentals = pd.merge(avg_start, avg_end, on=NEIGH_NAME, how='outer').fillna(0)
# compute price change always
df_rentals['price_change'] = df_rentals[f"avg_price_{END_YEAR}"] - df_rentals[f"avg_price_{START_YEAR}"]
# x_value based on user choice
if COMPARE_DIFF:
    df_rentals['x_value'] = df_rentals['price_change']
    x_label = f"Price Change ({END_YEAR}–{START_YEAR})"
else:
    df_rentals['x_value'] = df_rentals[f"avg_price_{END_YEAR}"]
    x_label = f"Average Price ({END_YEAR})"
print(df_rentals.head())
# ----------------------------------------------------------------------
# 4. Merge with neighbourhood geometries and compute Airbnb density ––––
# ----------------------------------------------------------------------
gdf_neigh = gpd.read_file(geojson_neigh)
gdf_neigh = gdf_neigh.merge(df_rentals, on=NEIGH_NAME, how='left').fillna(0)
joined = gpd.sjoin(gdf_airbnb, gdf_neigh, how='left', predicate='within')
counts = joined.groupby('neighbourhood_right').size().reset_index(name='airbnb_count')
counts.rename(columns={'neighbourhood_right':'neighbourhood'}, inplace=True)
gdf_neigh = gdf_neigh.merge(counts, on='neighbourhood', how='left').fillna({'airbnb_count':0})
gdf_neigh['area_km2'] = gdf_neigh.to_crs(epsg=3857).area / 1e6
gdf_neigh['airbnb_density'] = gdf_neigh['airbnb_count'] / gdf_neigh['area_km2']

print(f"Total Airbnb listings: {len(gdf_airbnb)}")

# ----------------------------------------------------------------------
# 5. Maps: Airbnb density & price change –––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column='airbnb_density', cmap='RdPu', legend=True, ax=ax, edgecolor='black')
ax.set_title('Airbnb Density (listings/km²) – London')
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column='price_change', cmap='YlOrRd', legend=True, ax=ax, edgecolor='black')
ax.set_title(f'Price Change ({END_YEAR}–{START_YEAR}) – London')
ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
plt.show()

# ----------------------------------------------------------------------
# 6. Scatter plot: density vs. price –––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(gdf_neigh['price_change'], gdf_neigh['airbnb_density'], alpha=0.7, edgecolors='w')
ax.set_xlabel(f"Price Change ({END_YEAR}–{START_YEAR})")
ax.set_ylabel('Airbnb Density (listings/km²)')
ax.set_title('Density vs. Price Change – London')
plt.show()

# ----------------------------------------------------------------------
# 7. Polynomial regression and bias-variance trade-off ––––––––––––––––––
# ----------------------------------------------------------------------
X = gdf_neigh['x_value'].values.reshape(-1,1)
Y = gdf_neigh['airbnb_density'].values
degrees = range(1,6)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = []
for d in degrees:
    Xp = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    lr = LinearRegression()
    cv = cross_validate(lr, Xp, Y, cv=kf, scoring='neg_mean_squared_error', return_train_score=True)
    results.append({'degree':d,'train_mse':-cv['train_score'].mean(),'test_mse':-cv['test_score'].mean()})
cv_df = pd.DataFrame(results)
print(cv_df)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cv_df['degree'], cv_df['train_mse'], marker='o', label='Train MSE')
ax.plot(cv_df['degree'], cv_df['test_mse'], marker='o', label='Val MSE')
ax.set_xlabel('Polynomial Degree'); ax.set_ylabel('MSE (log scale)')
ax.set_yscale('log'); ax.set_title('Bias-Variance Trade-Off')
ax.legend(); plt.show()

# best model fit and plot
best_deg = int(cv_df.loc[cv_df['test_mse'].idxmin(), 'degree'])
poly = PolynomialFeatures(best_deg, include_bias=False)
model = LinearRegression().fit(poly.fit_transform(X), Y)
r2 = model.score(poly.fit_transform(X), Y)
r = np.sqrt(r2) if r2>0 else 0

X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
Y_plot = model.predict(poly.transform(X_plot))
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, Y, alpha=0.7, edgecolors='w', label='Data')
ax.plot(X_plot, Y_plot, 'r--', label=f'Deg {best_deg}, R={r:.2f}')
ax.set_xlabel(x_label)
ax.set_ylabel('Airbnb Density (listings/km²)')
ax.set_title('Polynomial Regression – London')
ax.legend(); plt.show()

# ----------------------------------------------------------------------
# 8. Binning & quadratic fit on medians –––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
gdf_neigh['price_bin'] = pd.qcut(gdf_neigh['price_change'], q=5, duplicates='drop')

plt.figure(figsize=(12,6))
sns.boxplot(x='price_bin', y='airbnb_density', data=gdf_neigh)
plt.xlabel('Price Change Quintile')
plt.ylabel('Airbnb Density')
plt.title('Density by Price Change Quintile')
plt.xticks(rotation=45); plt.show()

# quadratic fit of bin medians
bin_stats = gdf_neigh.groupby('price_bin', observed=False).agg(
    price_change=('price_change','mean'),
    airbnb_density=('airbnb_density','median')
).reset_index()
Xb = bin_stats['price_change'].values.reshape(-1,1)
yb = bin_stats['airbnb_density'].values
poly2 = PolynomialFeatures(2, include_bias=False)
model2 = LinearRegression().fit(poly2.fit_transform(Xb), yb)
X_curve = np.linspace(gdf_neigh['price_change'].min(), gdf_neigh['price_change'].max(), 100).reshape(-1,1)
y_curve = model2.predict(poly2.transform(X_curve))

fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(Xb, yb, s=80, label='Bin Medians')
ax.plot(X_curve, y_curve, 'r--', label='Quadratic Fit')
for edge in gdf_neigh['price_bin'].cat.categories.right[:-1]: ax.axvline(edge, linestyle=':', alpha=0.5)
ax.set_xlabel('Mean Price Change')
ax.set_ylabel('Median Airbnb Density')
ax.set_title('Quadratic Regression on Binned Data')
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()