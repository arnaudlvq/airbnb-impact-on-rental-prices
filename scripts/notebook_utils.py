import json
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import shape, Point
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate, KFold
from sklearn.metrics import r2_score
import seaborn as sns

# ----------------------------------------------------------------------
# UTILITY FUNCTIONS
# ----------------------------------------------------------------------
def create_point_from_coords(row, lon_idx, lat_idx):
    """Create a shapely Point from longitude and latitude columns"""
    try:
        lon, lat = row.iloc[lon_idx], row.iloc[lat_idx]
        return Point(lon, lat)
    except Exception as e:
        print(f"Error creating point: {e}")
        return None

def parse_paris_coords(coord_str):
    """Parse coordinate string from Paris data format"""
    try:
        lat_str, lon_str = coord_str.split(',')
        lat = float(lat_str.strip())
        lon = float(lon_str.strip())
        return Point(lon, lat)
    except Exception as e:
        print(f"Error parsing coordinates: {e}")
        return None
        
def convert_geojson_to_shape(geo_str):
    """Convert GeoJSON string to shapely geometry object"""
    try:
        geo_dict = json.loads(geo_str)
        return shape(geo_dict)
    except Exception as e:
        print(f"Error converting geometry: {e}")
        return None

def fit_polynomial_models(X, Y, max_degree=5):
    """Fit polynomial models of different degrees and perform cross-validation"""
    degrees = range(1, max_degree + 1)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_results_list = []

    for deg in degrees:
        poly = PolynomialFeatures(degree=deg, include_bias=False)
        X_poly = poly.fit_transform(X)
        lr = LinearRegression()
        cv_results = cross_validate(lr, X_poly, Y, cv=kf, 
                                   scoring='neg_mean_squared_error', 
                                   return_train_score=True)
        train_mse = -np.mean(cv_results['train_score'])
        test_mse = -np.mean(cv_results['test_score'])
        cv_results_list.append({
            'degree': deg, 
            'train_mse': train_mse, 
            'test_mse': test_mse
        })
    
    return pd.DataFrame(cv_results_list)

# For the scatter plot of Airbnb densities
def plot_airbnb_density_scatter_comparison(paris_data, london_data):
    """Create side-by-side scatter plots comparing Airbnb density vs price changes"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Paris scatter plot
    axes[0].scatter(paris_data['price_increase'], paris_data['airbnb_density'], 
                alpha=0.7, color=PARIS_COLOR)
    axes[0].set_xlabel('Rental Price Increase (2024 - 2019) [€/m²]')
    axes[0].set_ylabel('Airbnb Density (listings/km²)')
    axes[0].set_title('Paris: Airbnb Density vs. Rental Price Increase')
    axes[0].grid(True, alpha=0.3)
    
    # London scatter plot
    axes[1].scatter(london_data['price_change'], london_data['airbnb_density'], 
                alpha=0.7, color=LONDON_COLOR)
    axes[1].set_xlabel(f'Rental Price Change ({LONDON_END_YEAR}–{LONDON_START_YEAR})')
    axes[1].set_ylabel('Airbnb Density (listings/km²)')
    axes[1].set_title('London: Airbnb Density vs. Rental Price Change')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

# For the bias-variance trade-off plots
def plot_bias_variance_tradeoff_comparison(paris_cv_results, london_cv_results):
    """Plot bias-variance tradeoff curves for both cities side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Paris curves
    axes[0].plot(paris_cv_results['degree'], paris_cv_results['train_mse'], 
             marker='o', color=PARIS_COLOR, linestyle='-', label='Training MSE')
    axes[0].plot(paris_cv_results['degree'], paris_cv_results['test_mse'], 
             marker='o', color=PARIS_COLOR, linestyle='--', label='Validation MSE')
    axes[0].set_xlabel('Polynomial Degree')
    axes[0].set_ylabel('Mean Squared Error')
    axes[0].set_title('Paris: Bias-Variance Trade-Off')
    axes[0].set_yscale('log')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # London curves
    axes[1].plot(london_cv_results['degree'], london_cv_results['train_mse'], 
             marker='s', color=LONDON_COLOR, linestyle='-', label='Training MSE')
    axes[1].plot(london_cv_results['degree'], london_cv_results['test_mse'], 
             marker='s', color=LONDON_COLOR, linestyle='--', label='Validation MSE')
    axes[1].set_xlabel('Polynomial Degree')
    axes[1].set_ylabel('Mean Squared Error')
    axes[1].set_title('London: Bias-Variance Trade-Off')
    axes[1].set_yscale('log')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

# For polynomial regression plots
def plot_polynomial_regression_comparison(paris_X, paris_Y, paris_poly, paris_model, paris_r, paris_deg,
                                         london_X, london_Y, london_poly, london_model, london_r, london_deg):
    """Plot polynomial regression curves for both cities side by side"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Paris regression
    paris_X_range = np.linspace(paris_X.min(), paris_X.max(), 100).reshape(-1, 1)
    paris_X_range_poly = paris_poly.transform(paris_X_range)
    paris_Y_pred = paris_model.predict(paris_X_range_poly)
    
    axes[0].scatter(paris_X, paris_Y, alpha=0.7, color=PARIS_COLOR, label='Neighborhoods')
    axes[0].plot(paris_X_range, paris_Y_pred, color=PARIS_COLOR, linestyle='--', 
             label=f'Polynomial Regression (deg {paris_deg}, R = {paris_r:.2f})')
    axes[0].set_xlabel('Rental Price Increase (2024 - 2019) [€/m²]')
    axes[0].set_ylabel('Airbnb Density (listings/km²)')
    axes[0].set_title('Paris: Airbnb Density vs. Rental Price Increase')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # London regression
    london_X_range = np.linspace(london_X.min(), london_X.max(), 100).reshape(-1, 1)
    london_X_range_poly = london_poly.transform(london_X_range)
    london_Y_pred = london_model.predict(london_X_range_poly)
    
    axes[1].scatter(london_X, london_Y, alpha=0.7, color=LONDON_COLOR, label='Neighborhoods')
    axes[1].plot(london_X_range, london_Y_pred, color=LONDON_COLOR, linestyle='--', 
             label=f'Polynomial Regression (deg {london_deg}, R = {london_r:.2f})')
    axes[1].set_xlabel(f'Rental Price Change ({LONDON_END_YEAR}–{LONDON_START_YEAR})')
    axes[1].set_ylabel('Airbnb Density (listings/km²)')
    axes[1].set_title('London: Airbnb Density vs. Rental Price Change')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

# For density distribution as bar chart instead of KDE
def plot_density_distribution_comparison(paris_data, london_data):
    """Create side-by-side histograms with KDE curves for Airbnb density distributions"""
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # Paris density distribution
    sns.histplot(
        paris_data['airbnb_density'], 
        kde=True, 
        ax=axes[0], 
        color=PARIS_COLOR,
        stat='density',
        bins=15,
        alpha=0.7
    )
    axes[0].set_title('Paris: Airbnb Density Distribution', fontsize=14)
    axes[0].set_xlabel('Airbnb Density (listings/km²)', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # London density distribution
    sns.histplot(
        london_data['airbnb_density'], 
        kde=True, 
        ax=axes[1], 
        color=LONDON_COLOR,
        stat='density',
        bins=15,
        alpha=0.7
    )
    axes[1].set_title('London: Airbnb Density Distribution', fontsize=14)
    axes[1].set_xlabel('Airbnb Density (listings/km²)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    # Improve layout
    plt.tight_layout()
    return fig, axes

def fit_best_model(X, Y, cv_results_df):
    """Fit the best polynomial model based on cross-validation results"""
    best_row = cv_results_df.loc[cv_results_df['test_mse'].idxmin()]
    best_deg = int(best_row['degree'])
    print(f"Best Polynomial Degree: {best_deg}")
    
    poly_best = PolynomialFeatures(degree=best_deg, include_bias=False)
    X_poly_best = poly_best.fit_transform(X)
    lr_best = LinearRegression()
    lr_best.fit(X_poly_best, Y)
    r2_best = lr_best.score(X_poly_best, Y)
    r_best = np.sqrt(r2_best) if r2_best >= 0 else 0
    
    return best_deg, poly_best, lr_best, r_best


def plot_boxplot_comparison(paris_df, london_df):
    """Create side-by-side boxplots for both cities"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Paris boxplot
    sns.boxplot(x='price_increase_bin', y='airbnb_density', data=paris_df, ax=axes[0], color=PARIS_COLOR)
    axes[0].set_title("Paris: Distribution of Airbnb Density by Price Increase Bins")
    axes[0].set_xlabel("Rental Price Increase (2024 - 2019) [€/m²] (Binned)")
    axes[0].set_ylabel("Airbnb Density (listings per km²)")
    axes[0].tick_params(axis='x', rotation=45)
    
    # London boxplot
    sns.boxplot(x='price_bin', y='airbnb_density', data=london_df, ax=axes[1], color=LONDON_COLOR)
    axes[1].set_title(f"London: Distribution of Airbnb Density by Price Change Bins")
    axes[1].set_xlabel(f"Price Change ({LONDON_END_YEAR}–{LONDON_START_YEAR}) Quintile")
    axes[1].set_ylabel("Airbnb Density (listings per km²)")
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig, axes

def plot_quadratic_fit_comparison(paris_df, london_df):
    """Plot quadratic fits on binned data for both cities with R² values"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Paris quadratic fit
    paris_bin_stats = paris_df.groupby('price_increase_bin', observed=False).agg({
        'price_increase': 'mean', 
        'airbnb_density': 'median'
    }).reset_index()
    
    paris_X = paris_bin_stats['price_increase'].values.reshape(-1, 1)
    paris_y = paris_bin_stats['airbnb_density'].values
    
    paris_poly = PolynomialFeatures(degree=2, include_bias=False)
    paris_X_poly = paris_poly.fit_transform(paris_X)
    paris_model = LinearRegression()
    paris_model.fit(paris_X_poly, paris_y)
    
    # Calculate R² for Paris model
    paris_y_pred = paris_model.predict(paris_X_poly)
    paris_r2 = r2_score(paris_y, paris_y_pred)
    
    paris_X_curve = np.linspace(paris_df['price_increase'].min(), paris_df['price_increase'].max(), 100).reshape(-1, 1)
    paris_y_curve = paris_model.predict(paris_poly.transform(paris_X_curve))
    
    axes[0].scatter(paris_X, paris_y, color=PARIS_COLOR, s=80, label='Bin Medians')
    axes[0].plot(paris_X_curve, paris_y_curve, color=PARIS_COLOR, linestyle='--', label='Quadratic Fit')
    
    # Add bin edge vertical lines for Paris
    paris_bin_edges = pd.unique(paris_df['price_increase_bin'].cat.categories.right.values[:-1])
    for edge in paris_bin_edges:
        axes[0].axvline(x=edge, color='gray', linestyle=':', alpha=0.5)
    
    axes[0].set_title(f"Paris: Quadratic Regression on Binned Data (R² = {paris_r2:.3f})")
    axes[0].set_xlabel("Mean Price Increase (2024 - 2019) (€/m²)")
    axes[0].set_ylabel("Median Airbnb Density (listings/km²)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # London quadratic fit
    london_bin_stats = london_df.groupby('price_bin', observed=False).agg({
        'price_change': 'mean', 
        'airbnb_density': 'median'
    }).reset_index()
    
    london_X = london_bin_stats['price_change'].values.reshape(-1, 1)
    london_y = london_bin_stats['airbnb_density'].values
    
    london_poly = PolynomialFeatures(degree=2, include_bias=False)
    london_X_poly = london_poly.fit_transform(london_X)
    london_model = LinearRegression()
    london_model.fit(london_X_poly, london_y)
    
    # Calculate R² for London model
    london_y_pred = london_model.predict(london_X_poly)
    london_r2 = r2_score(london_y, london_y_pred)
    
    london_X_curve = np.linspace(london_df['price_change'].min(), london_df['price_change'].max(), 100).reshape(-1, 1)
    london_y_curve = london_model.predict(london_poly.transform(london_X_curve))
    
    axes[1].scatter(london_X, london_y, color=LONDON_COLOR, s=80, label='Bin Medians')
    axes[1].plot(london_X_curve, london_y_curve, color=LONDON_COLOR, linestyle='--', label='Quadratic Fit')
    
    # Add bin edge vertical lines for London
    london_bin_edges = pd.unique(london_df['price_bin'].cat.categories.right.values[:-1])
    for edge in london_bin_edges:
        axes[1].axvline(x=edge, color='gray', linestyle=':', alpha=0.5)
    
    axes[1].set_title(f"London: Quadratic Regression on Binned Data (R² = {london_r2:.3f})")
    axes[1].set_xlabel(f"Mean Price Change ({LONDON_END_YEAR}–{LONDON_START_YEAR})")
    axes[1].set_ylabel("Median Airbnb Density (listings/km²)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, axes

# For correlation bar chart
def plot_correlation_bar_chart(paris_corr, london_corr, paris_pvalue, london_pvalue):
    """Plot a bar chart comparing Pearson correlations for Paris and London with p-value annotations."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        ['Paris', 'London'],
        [paris_corr, london_corr],
        color=[PARIS_COLOR, LONDON_COLOR],
        edgecolor='black',
        alpha=0.85
    )
    plt.axhline(y=0, color='red', linestyle='--', alpha=0.4, linewidth=1)
    plt.ylim(-1, 1)
    plt.title('Correlation between Rental Price Changes and Airbnb Density', fontsize=16, fontweight='bold')
    plt.ylabel('Pearson Correlation Coefficient', fontsize=13)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Annotate correlation values on bars
    for idx, (corr, pval) in enumerate(zip([paris_corr, london_corr], [paris_pvalue, london_pvalue])):
        plt.text(
            idx, corr + (0.07 if corr > 0 else -0.13),
            f"r={corr:.2f}\np={pval:.4f}",
            ha='center', va='bottom' if corr > 0 else 'top',
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.7)
        )
    
    plt.tight_layout()