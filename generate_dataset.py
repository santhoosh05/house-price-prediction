import pandas as pd
import numpy as np
import os

def generate_housing_data(num_samples=2500, random_seed=42):
    np.random.seed(random_seed)
    
    # Generate feature arrays
    sqft = np.random.randint(800, 5000, size=num_samples)
    bedrooms = np.random.randint(1, 6, size=num_samples)
    bathrooms = np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0], size=num_samples, 
                                 p=[0.1, 0.15, 0.3, 0.2, 0.15, 0.05, 0.05])
    
    neighborhoods = ['Rural', 'Suburb', 'Urban', 'Downtown', 'Waterfront']
    neighborhood = np.random.choice(neighborhoods, size=num_samples, 
                                    p=[0.15, 0.35, 0.25, 0.15, 0.10])
    
    year_built = np.random.randint(1950, 2026, size=num_samples)
    has_garden = np.random.choice([0, 1], size=num_samples, p=[0.4, 0.6])
    floors = np.random.choice([1, 2, 3], size=num_samples, p=[0.5, 0.4, 0.1])
    garage_capacity = np.random.choice([0, 1, 2, 3], size=num_samples, p=[0.2, 0.3, 0.4, 0.1])
    condition = np.random.choice([1, 2, 3, 4, 5], size=num_samples, p=[0.05, 0.15, 0.45, 0.25, 0.10])
    
    # Neighborhood multiplier for base value
    neighborhood_multipliers = {
        'Rural': 0.8,
        'Suburb': 1.0,
        'Urban': 1.25,
        'Downtown': 1.5,
        'Waterfront': 1.95
    }
    
    # Initialize dataframe
    df = pd.DataFrame({
        'sqft': sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'neighborhood': neighborhood,
        'year_built': year_built,
        'has_garden': has_garden,
        'floors': floors,
        'garage_capacity': garage_capacity,
        'condition': condition
    })
    
    # Compute base price before age depreciation and condition adjustments
    # Note: Using interaction terms (e.g. sqft price depends on neighborhood) makes the dataset non-linear,
    # which highlights the benefits of Random Forest / XGBoost over basic Linear Regression.
    base_sqft_price = 150.0
    prices = []
    
    for i in range(num_samples):
        n_type = df.loc[i, 'neighborhood']
        mult = neighborhood_multipliers[n_type]
        
        # Core structural price
        structural_price = (
            (df.loc[i, 'sqft'] * base_sqft_price) +
            (df.loc[i, 'bedrooms'] * 22000) +
            (df.loc[i, 'bathrooms'] * 30000)
        )
        
        # Apply neighborhood multiplier
        loc_price = structural_price * mult
        
        # Add values for amenities
        amenities = (
            (df.loc[i, 'has_garden'] * 12000) +
            ((df.loc[i, 'floors'] - 1) * 15000) +
            (df.loc[i, 'garage_capacity'] * 10000)
        )
        
        # House age depreciation
        age = 2026 - df.loc[i, 'year_built']
        depreciation = max(0, age * 750) # Depreciation up to a certain point
        
        # Condition effect (1 to 5, where 3 is baseline, 5 adds value, 1 subtracts)
        cond_effect = (df.loc[i, 'condition'] - 3) * 20000
        
        # Total price
        price = loc_price + amenities - depreciation + cond_effect + 25000 # baseline shift
        
        # Ensure price is positive and has some random variance (noise)
        noise = np.random.normal(0, 18000)
        final_price = max(50000, price + noise)
        prices.append(round(final_price, -2)) # round to nearest hundred
        
    df['price'] = prices
    
    return df

if __name__ == '__main__':
    print("Generating synthetic housing data...")
    df = generate_housing_data()
    output_path = 'housing_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset generated with {len(df)} samples and saved to '{output_path}'.")
    print(df.head())
