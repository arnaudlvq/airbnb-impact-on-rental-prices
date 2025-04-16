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
# 1.  File paths ––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
excel_rentals = "../data/london_rentals.xls"        # sheet “Raw data”
csv_airbnb    = "../data/london_airbnb.csv"         # identical format to Paris file
geojson_neigh = "../data/london_neighbourhoods.geojson"

# ----------------------------------------------------------------------
# 2.  Airbnb listings  ––––––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
df_airbnb = pd.read_csv(csv_airbnb, delimiter=",", encoding="utf-8")

def make_point(row):
    try:
        lon, lat = row.iloc[7], row.iloc[6]          # same columns as Paris file
        return Point(lon, lat)
    except Exception:
        return None

df_airbnb["geometry"] = df_airbnb.apply(make_point, axis=1)
df_airbnb = df_airbnb[df_airbnb["geometry"].notnull()]
gdf_airbnb = gpd.GeoDataFrame(df_airbnb, geometry="geometry", crs="EPSG:4326")

# ----------------------------------------------------------------------
# 3.  Rentals (2011 vs 2019)  –––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
# Read sheet WITHOUT headers so we can use positional indices
df_raw = pd.read_excel(excel_rentals, sheet_name="Raw data", header=None)

# Rename the neighbourhood column (index 2 → “Neighbourhood”)
name_col = 2
df_raw.rename(columns={name_col: "Neighbourhood"}, inplace=True)

# Filter: year, quarter, category
df_filt = df_raw[
    (df_raw.iloc[:, 0].isin([2011, 2019])) &      # year
    (df_raw.iloc[:, 1] == "Q1") &                 # quarter
    (df_raw.iloc[:, 4] == "All categories")       # category
].copy()

# Rental price in column 6
df_filt["rental_price"] = pd.to_numeric(df_filt.iloc[:, 6], errors="coerce")
df_filt = df_filt.dropna(subset=["Neighbourhood", "rental_price"])

# Average by neighbourhood & year
avg_2011 = (
    df_filt[df_filt.iloc[:, 0] == 2011]
    .groupby("Neighbourhood")["rental_price"]
    .mean()
    .reset_index(name="avg_rental_price_2011")
)
avg_2019 = (
    df_filt[df_filt.iloc[:, 0] == 2019]
    .groupby("Neighbourhood")["rental_price"]
    .mean()
    .reset_index(name="avg_rental_price_2019")
)

df_rentals = pd.merge(avg_2011, avg_2019, on="Neighbourhood", how="outer").fillna(0)
df_rentals["price_increase"] = (
    df_rentals["avg_rental_price_2019"] - df_rentals["avg_rental_price_2011"]
)

# ----------------------------------------------------------------------
# 4.  Neighbourhood polygons  –––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
gdf_neigh = gpd.read_file(geojson_neigh)

# Standardise name column to “Neighbourhood”
if "neighbourhood" in gdf_neigh.columns:
    gdf_neigh.rename(columns={"neighbourhood": "Neighbourhood"}, inplace=True)

# Merge rental attributes
gdf_neigh = gdf_neigh.merge(df_rentals, on="Neighbourhood", how="left").fillna(0)

# ----------------------------------------------------------------------
# 5.  Airbnb density  –––––––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
gdf_joined = gpd.sjoin(gdf_airbnb, gdf_neigh, how="left", predicate="within")
counts = gdf_joined.groupby("Neighbourhood").size().reset_index(name="airbnb_count")

gdf_neigh = gdf_neigh.merge(counts, on="Neighbourhood", how="left")
gdf_neigh["airbnb_count"] = gdf_neigh["airbnb_count"].fillna(0).astype(int)
gdf_neigh["area_km2"] = gdf_neigh.to_crs(epsg=3857).area / 1e6
gdf_neigh["airbnb_density"] = gdf_neigh["airbnb_count"] / gdf_neigh["area_km2"]

print(f"Total Airbnb listings in London: {len(gdf_airbnb)}")

# ----------------------------------------------------------------------
# 6.  Maps  –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column="airbnb_density", cmap="RdPu", legend=True, ax=ax, edgecolor="black")
ax.set_title("Airbnb Density (listings / km²) – London (2024)")
plt.show()

fig, ax = plt.subplots(figsize=(10, 10))
gdf_neigh.plot(column="price_increase", cmap="YlOrRd", legend=True, ax=ax, edgecolor="black")
ax.set_title("Rental Price Increase (2019 – 2011) – London")
ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
plt.show()

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(gdf_neigh["price_increase"], gdf_neigh["airbnb_density"],
           alpha=0.7, edgecolors="w")
ax.set_xlabel("Rental Price Increase (2019 – 2011)")
ax.set_ylabel("Airbnb Density (listings / km²)")
ax.set_title("Airbnb Density vs Rental Price Increase – London")
plt.show()

# ----------------------------------------------------------------------
# 7.  Polynomial regression (degrees 1–5)  –––––––––––––––––––––––––––––
# ----------------------------------------------------------------------
X = gdf_neigh["price_increase"].values.reshape(-1, 1)
Y = gdf_neigh["airbnb_density"].values
degrees = range(1, 6)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = []
for d in degrees:
    X_poly = PolynomialFeatures(d, include_bias=False).fit_transform(X)
    lr = LinearRegression()
    cv = cross_validate(lr, X_poly, Y, cv=kf,
                        scoring="neg_mean_squared_error",
                        return_train_score=True)
    results.append({
        "degree": d,
        "train_mse": -cv["train_score"].mean(),
        "test_mse":  -cv["test_score"].mean()
    })

cv_df = pd.DataFrame(results)
print(cv_df)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(cv_df["degree"], cv_df["train_mse"], marker="o", label="Training MSE")
ax.plot(cv_df["degree"], cv_df["test_mse"],  marker="o", label="Validation MSE")
ax.set_xlabel("Polynomial degree"); ax.set_ylabel("MSE (log‐scale)")
ax.set_yscale("log"); ax.set_title("Bias–Variance Trade‑Off")
ax.legend(); plt.show()

best_deg = cv_df.loc[cv_df["test_mse"].idxmin(), "degree"]
print(f"Best degree = {best_deg}")

# Fit and plot the chosen model
poly = PolynomialFeatures(int(best_deg), include_bias=False)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, Y)
r2 = model.score(X_poly, Y)
r = np.sqrt(r2) if r2 >= 0 else 0

X_plot = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
Y_plot = model.predict(poly.transform(X_plot))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(X, Y, alpha=0.7, edgecolors="w", label="Neighbourhoods")
ax.plot(X_plot, Y_plot, "r--", label=f"Poly deg {best_deg}, R={r:.2f}")
ax.set_xlabel("Rental Price Increase (2019 – 2011)")
ax.set_ylabel("Airbnb Density (listings / km²)")
ax.set_title("Polynomial Regression – London")
ax.legend(); plt.show()

# ----------------------------------------------------------------------
# 8.  Binning and quadratic fit of bin medians  ––––––––––––––––––––––––
# ----------------------------------------------------------------------
gdf_neigh["price_bin"] = pd.qcut(gdf_neigh["price_increase"],
                                 q=5, duplicates="drop")

plt.figure(figsize=(12, 6))
sns.boxplot(x="price_bin", y="airbnb_density", data=gdf_neigh)
plt.xlabel("Rental Price Increase (binned)"); plt.ylabel("Airbnb Density")
plt.title("Airbnb Density by Price Increase Quintile – London")
plt.xticks(rotation=45); plt.show()

bin_stats = (
    gdf_neigh.groupby("price_bin", observed=False)
    .agg(price_increase=("price_increase", "mean"),
         airbnb_density=("airbnb_density", "median"))
    .reset_index()
)

Xb = bin_stats["price_increase"].values.reshape(-1, 1)
yb = bin_stats["airbnb_density"].values
poly2 = PolynomialFeatures(2, include_bias=False)
model2 = LinearRegression().fit(poly2.fit_transform(Xb), yb)

X_curve = np.linspace(gdf_neigh["price_increase"].min(),
                      gdf_neigh["price_increase"].max(), 100).reshape(-1, 1)
y_curve = model2.predict(poly2.transform(X_curve))

fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(Xb, yb, s=80, label="Bin medians")
ax.plot(X_curve, y_curve, "r--", label="Quadratic fit")
for edge in gdf_neigh["price_bin"].cat.categories.right[:-1]:
    ax.axvline(edge, color="g", linestyle=":", alpha=0.5)
ax.set_xlabel("Mean Rental Price Increase (2019 – 2011)")
ax.set_ylabel("Median Airbnb Density (listings / km²)")
ax.set_title("Quadratic Regression on Binned Data – London")
ax.legend(); ax.grid(alpha=0.3); plt.tight_layout(); plt.show()
