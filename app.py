from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

class LifeExpectancyPredictor:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = None
        self.label_encoder = None
        self.feature_columns = []
        self.is_trained = False

    def load_model(self, filepath):
        """Load the trained model from train_model.py"""
        try:
            print(f"Attempting to load model from: {filepath}")
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            # Extract all components from the saved model
            self.ensemble_model = model_data['ensemble_model']
            self.scaler = model_data['scaler']
            self.label_encoder = model_data['label_encoder']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = True
            
            print(f"✅ Model loaded successfully!")
            print(f"📊 Feature columns: {self.feature_columns}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.is_trained = False
            return False

    def predict(self, input_data):
        """Make prediction using the loaded model"""
        if not self.is_trained:
            raise ValueError("Model not trained or loaded!")
        
        try:
            # Create DataFrame from input data
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            for col in self.feature_columns:
                if col not in input_df.columns:
                    if col == 'Status_encoded':
                        input_df[col] = input_data.get('Status_encoded', 1)  # Default to developing
                    else:
                        input_df[col] = 0.0
            
            # Select only the features used during training
            input_features = input_df[self.feature_columns]
            
            # Handle any missing values
            input_features = input_features.fillna(0)
            
            # Scale the features using the same scaler from training
            input_scaled = self.scaler.transform(input_features)
            
            # Make prediction
            prediction = self.ensemble_model.predict(input_scaled)[0]
            
            return prediction
            
        except Exception as e:
            print(f"Prediction error: {str(e)}")
            raise e

    def categorize_life_expectancy(self, life_exp):
        """Categorize life expectancy into ranges"""
        if life_exp < 50:
            return "Very Low Life Expectancy", "danger"
        elif life_exp < 60:
            return "Low Life Expectancy", "warning"
        elif life_exp < 70:
            return "Average Life Expectancy", "info"
        elif life_exp < 75:
            return "Good Life Expectancy", "success"
        else:
            return "Excellent Life Expectancy", "primary"

# Initialize predictor
predictor = LifeExpectancyPredictor()

# Load the trained model on startup
model_path = 'models/life_model.pkl'
if os.path.exists(model_path):
    model_loaded = predictor.load_model(model_path)
    if not model_loaded:
        print("⚠️ Failed to load model. Please retrain the model.")
else:
    print("❌ No trained model found at 'models/life_model.pkl'. Please run train_model.py first.")

@app.route('/')
def index():
    """Home page"""
    return render_template('homepagetemplate.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page"""
    if request.method == 'GET':
        # Show the prediction form
        return render_template('base.html')
    
    try:
        # Extract form data
        input_data = {
            'Adult Mortality': float(request.form.get('adult_mortality', 0)),
            'infant deaths': float(request.form.get('infant_deaths', 0)),
            'Alcohol': float(request.form.get('alcohol', 0)),
            'percentage expenditure': float(request.form.get('percentage_expenditure', 0)),
            'Hepatitis B': float(request.form.get('hepatitis_b', 0)),
            'Measles': float(request.form.get('measles', 0)),
            'BMI': float(request.form.get('bmi', 0)),
            'under-five deaths': float(request.form.get('under_five_deaths', 0)),
            'Polio': float(request.form.get('polio', 0)),
            'Total expenditure': float(request.form.get('total_expenditure', 0)),
            'Diphtheria': float(request.form.get('diphtheria', 0)),
            'HIV/AIDS': float(request.form.get('hiv_aids', 0)),
            'GDP': float(request.form.get('gdp', 0)),
            'Population': float(request.form.get('population', 0)),
            'thinness  1-19 years': float(request.form.get('thinness_1_19', 0)),
            'thinness 5-9 years': float(request.form.get('thinness_5_9', 0)),
            'Income composition of resources': float(request.form.get('income_composition', 0)),
            'Schooling': float(request.form.get('schooling', 0)),
            'Status_encoded': 0 if request.form.get('status', '').lower() == 'developed' else 1
        }

        print(f"📝 Input data received: {input_data}")

        if predictor.is_trained:
            # Make prediction
            life_expectancy = predictor.predict(input_data)
            category, alert_type = predictor.categorize_life_expectancy(life_expectancy)

            result = {
                'life_expectancy': round(life_expectancy, 2),
                'category': category,
                'alert_type': alert_type,
                'success': True
            }
            
            print(f"🎯 Prediction made: {life_expectancy:.2f} years")
        else:
            result = {
                'error': 'Model not loaded. Please ensure the model is trained and saved properly.',
                'success': False
            }

        return render_template('result.html', result=result, input_data=input_data)

    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        result = {
            'error': f'Error making prediction: {str(e)}',
            'success': False
        }
        return render_template('result.html', result=result, input_data={})

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    """Training page"""
    if request.method == 'GET':
        return render_template('train.html')

    try:
        # Import and run the training script
        import subprocess
        import sys
        
        # Run the training script
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            # Try to reload the model after training
            if predictor.load_model(model_path):
                return render_template('train.html', 
                                     success="Model trained and loaded successfully!")
            else:
                return render_template('train.html', 
                                     error="Model trained but failed to load.")
        else:
            return render_template('train.html', 
                                 error=f"Training failed: {result.stderr}")

    except Exception as e:
        return render_template('train.html', 
                             error=f"Error during training: {str(e)}")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()

        if predictor.is_trained:
            life_expectancy = predictor.predict(data)
            category, alert_type = predictor.categorize_life_expectancy(life_expectancy)

            return jsonify({
                'life_expectancy': round(life_expectancy, 2),
                'category': category,
                'alert_type': alert_type,
                'success': True
            })
        else:
            return jsonify({
                'error': 'Model not loaded',
                'success': False
            }), 400

    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/status')
def status():
    """Check model status"""
    return jsonify({
        'model_loaded': predictor.is_trained,
        'feature_count': len(predictor.feature_columns),
        'features': predictor.feature_columns if predictor.is_trained else []
    })

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 LIFE EXPECTANCY PREDICTION SERVER")
    print("="*50)
    print(f"📊 Model Status: {'✅ Loaded' if predictor.is_trained else '❌ Not Loaded'}")
    if predictor.is_trained:
        print(f"🔧 Features: {len(predictor.feature_columns)} columns")
    print("🌐 Server starting...")
    print("="*50)
    
    # Fix for Render deployment
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
