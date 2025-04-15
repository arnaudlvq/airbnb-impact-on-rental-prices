import pandas as pd

# Load the CSV file (adjust the path as necessary)
df = pd.read_csv("../data/paris_rentals.csv", delimiter=';', on_bad_lines='skip', encoding='utf-8')

# Drop duplicate rows based on the neighborhood identifier "Numéro du quartier" (column C)
unique_neighborhoods = df.drop_duplicates(subset="Numéro du quartier")

# Select only the columns for the neighborhood number and its geo_shape
result = unique_neighborhoods[["Numéro du quartier", "geo_shape"]]

# Optionally, check the number of unique neighborhoods (should be 80)
print("Number of unique neighborhoods:", result.shape[0])

# Save the result to a new CSV file
result.to_csv("paris_neighborhoods.csv", index=False, encoding='utf-8')
