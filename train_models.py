import pandas as pd
import numpy as np
import json
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_and_save_models():
    # 1. Load Data
    data_path = 'housing_data.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Please run generate_dataset.py first.")
        
    df = pd.read_csv(data_path)
    
    # 2. Split Features and Target
    X = df.drop(columns=['price'])
    y = df['price']
    
    # Define features
    num_features = ['sqft', 'bedrooms', 'bathrooms', 'year_built', 'floors', 'garage_capacity', 'condition']
    cat_features = ['neighborhood']
    
    # 3. Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_features)
        ])
    
    # Fit preprocessor on training data
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)
    
    # Get feature names after preprocessing
    feature_names = preprocessor.get_feature_names_out()
    # Clean feature names (remove prefixes like 'num__' and 'cat__')
    clean_feature_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]
    
    # 5. Initialize and Train Models
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=150, max_depth=6, learning_rate=0.07, random_state=42, n_jobs=-1)
    }
    
    metrics = {}
    trained_models = {}
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_preprocessed, y_train)
        
        # Predictions
        y_pred = model.predict(X_test_preprocessed)
        
        # Calculate Metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        metrics[name] = {
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r2': round(float(r2), 4)
        }
        
        # Extract Feature Importance (if applicable)
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            feature_imp_list = [
                {'feature': clean_feature_names[i], 'importance': round(float(importances[i]), 4)}
                for i in range(len(clean_feature_names))
            ]
            # Sort by importance descending
            feature_imp_list = sorted(feature_imp_list, key=lambda x: x['importance'], reverse=True)
            metrics[name]['feature_importances'] = feature_imp_list
        elif hasattr(model, 'coef_'):
            # For linear regression, we can use absolute coefficients as a proxy for "importance"
            # note: features are scaled so coefficients are comparable in magnitude
            coefs = np.abs(model.coef_)
            total_coef = np.sum(coefs)
            normalized_coefs = coefs / total_coef if total_coef > 0 else coefs
            feature_imp_list = [
                {'feature': clean_feature_names[i], 'importance': round(float(normalized_coefs[i]), 4)}
                for i in range(len(clean_feature_names))
            ]
            feature_imp_list = sorted(feature_imp_list, key=lambda x: x['importance'], reverse=True)
            metrics[name]['feature_importances'] = feature_imp_list
            
        # Save Model
        model_filename = f"models/{name.lower().replace(' ', '_')}.joblib"
        joblib.dump(model, model_filename)
        print(f"Saved {name} to {model_filename}")
        
    # Save Preprocessor
    preprocessor_filename = "models/preprocessor.joblib"
    joblib.dump(preprocessor, preprocessor_filename)
    print(f"Saved Preprocessor to {preprocessor_filename}")
    
    # Save Metrics JSON
    with open('models/model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print("Saved evaluation metrics to models/model_metrics.json")
    
    print("\n--- Model Training Completed ---")
    for name, m in metrics.items():
        print(f"{name}: R2 = {m['r2']:.4f} | MAE = {m['mae']:.2f} | RMSE = {m['rmse']:.2f}")

if __name__ == '__main__':
    train_and_save_models()
