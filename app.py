from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os
import json
import joblib

app = Flask(__name__)

# Global variables for models and preprocessor
preprocessor = None
model_lr = None
model_rf = None
model_xgb = None
model_metrics = {}

def load_ml_components():
    global preprocessor, model_lr, model_rf, model_xgb, model_metrics
    
    models_dir = 'models'
    
    # Path verification
    preprocessor_path = os.path.join(models_dir, 'preprocessor.joblib')
    lr_path = os.path.join(models_dir, 'linear_regression.joblib')
    rf_path = os.path.join(models_dir, 'random_forest.joblib')
    xgb_path = os.path.join(models_dir, 'xgboost.joblib')
    metrics_path = os.path.join(models_dir, 'model_metrics.json')
    
    if not (os.path.exists(preprocessor_path) and os.path.exists(lr_path) 
            and os.path.exists(rf_path) and os.path.exists(xgb_path)):
        print("WARNING: ML model files are missing. Please run 'train_models.py' first to train and serialize the models.")
        return False
        
    try:
        # Load Joblib objects
        preprocessor = joblib.load(preprocessor_path)
        model_lr = joblib.load(lr_path)
        model_rf = joblib.load(rf_path)
        model_xgb = joblib.load(xgb_path)
        
        # Load Metrics JSON
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                model_metrics = json.load(f)
        
        print("All machine learning models and preprocessors loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading ML components: {str(e)}")
        return False

# Load models at startup
load_ml_components()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    # If metrics are empty (models were not loaded), reload them
    if not model_metrics:
        load_ml_components()
    return jsonify(model_metrics)

@app.route('/api/predict', methods=['POST'])
def predict():
    global preprocessor, model_lr, model_rf, model_xgb
    
    # Ensure models are loaded
    if preprocessor is None or model_lr is None or model_rf is None or model_xgb is None:
        success = load_ml_components()
        if not success:
            return jsonify({'error': 'Models not loaded. Model files are missing on server.'}), 500
            
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
            
        # Extract features and verify correct order/schema matching training pipeline
        feature_dict = {
            'sqft': [int(data.get('sqft', 2000))],
            'bedrooms': [int(data.get('bedrooms', 3))],
            'bathrooms': [float(data.get('bathrooms', 2.0))],
            'neighborhood': [str(data.get('neighborhood', 'Suburb'))],
            'year_built': [int(data.get('year_built', 2000))],
            'has_garden': [int(data.get('has_garden', 1))],
            'floors': [int(data.get('floors', 1))],
            'garage_capacity': [int(data.get('garage_capacity', 1))],
            'condition': [int(data.get('condition', 3))]
        }
        
        # Create DataFrame
        input_df = pd.DataFrame(feature_dict)
        
        # Apply data preprocessor pipeline (Scales numerical, Encodes categorical)
        X_preprocessed = preprocessor.transform(input_df)
        
        # Make predictions
        pred_lr = model_lr.predict(X_preprocessed)[0]
        pred_rf = model_rf.predict(X_preprocessed)[0]
        pred_xgb = model_xgb.predict(X_preprocessed)[0]
        
        # Format response, rounding predictions to nearest $100
        predictions = {
            'linear_regression': max(50000, round(float(pred_lr), -2)),
            'random_forest': max(50000, round(float(pred_rf), -2)),
            'xgboost': max(50000, round(float(pred_xgb), -2))
        }
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': f'Prediction execution error: {str(e)}'}), 400

if __name__ == '__main__':
    # Start development server on port 5000
    app.run(debug=True, host='127.0.0.1', port=5000)
