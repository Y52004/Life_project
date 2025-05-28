import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class LifeExpectancyModelTrainer:
    def __init__(self):
        self.ensemble_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.feature_importance = None
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the WHO Life Expectancy dataset"""
        print("Loading dataset...")
        df = pd.read_csv(file_path)
        print(f"Dataset shape: {df.shape}")
        
        # Display basic info
        print("\nDataset Info:")
        print(df.info())
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Handle missing values
        print("\nHandling missing values...")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Handle categorical variables
        if 'Status' in df.columns:
            df['Status_encoded'] = self.label_encoder.fit_transform(df['Status'])
        
        # Remove outliers using IQR method for key columns
        outlier_cols = ['Life expectancy', 'GDP', 'Population']
        for col in outlier_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                before_count = len(df)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                after_count = len(df)
                print(f"Removed {before_count - after_count} outliers from {col}")
        
        return df
    
    def select_features(self, df):
        """Select relevant features for prediction"""
        # Define feature columns based on WHO health indicators
        potential_features = [
            'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure',
            'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio',
            'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
            'thinness  1-19 years', 'thinness 5-9 years', 
            'Income composition of resources', 'Schooling', 'Status_encoded'
        ]
        
        # Select only available features
        self.feature_columns = [col for col in potential_features if col in df.columns]
        print(f"\nSelected features: {self.feature_columns}")
        
        return df[self.feature_columns + ['Life expectancy']]
    
    def create_ensemble_model(self):
        """Create ensemble model with multiple algorithms"""
        # Individual models with tuned parameters
        rf_model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        lr_model = LinearRegression()
        
        # Create voting regressor (ensemble)
        ensemble_model = VotingRegressor([
            ('random_forest', rf_model),
            ('gradient_boosting', gb_model),
            ('linear_regression', lr_model)
        ])
        
        return ensemble_model
    
    def train_and_evaluate(self, df):
        """Train the ensemble model and evaluate performance"""
        print("\nPreparing data for training...")
        
        # Prepare features and target
        X = df[self.feature_columns]
        y = df['Life expectancy']
        
        # Remove any remaining NaN values
        mask = ~(X.isnull().any(axis=1) | y.isnull())
        X, y = X[mask], y[mask]
        
        print(f"Final dataset shape: {X.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Create and train ensemble model
        print("Training ensemble model...")
        self.ensemble_model = self.create_ensemble_model()
        self.ensemble_model.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = self.ensemble_model.predict(X_train_scaled)
        y_test_pred = self.ensemble_model.predict(X_test_scaled)
        
        # Evaluate performance
        print("\n" + "="*50)
        print("MODEL PERFORMANCE EVALUATION")
        print("="*50)
        
        # Training metrics
        train_mse = mean_squared_error(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_r2 = r2_score(y_train, y_train_pred)
        
        print(f"Training Metrics:")
        print(f"  MSE: {train_mse:.2f}")
        print(f"  MAE: {train_mae:.2f}")
        print(f"  R² Score: {train_r2:.4f}")
        
        # Testing metrics
        test_mse = mean_squared_error(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        
        print(f"\nTesting Metrics:")
        print(f"  MSE: {test_mse:.2f}")
        print(f"  MAE: {test_mae:.2f}")
        print(f"  R² Score: {test_r2:.4f}")
        
        # Feature importance from RandomForest only
        if hasattr(self.ensemble_model.estimators_[0], 'feature_importances_'):
            self.feature_importance = self.ensemble_model.estimators_[0].feature_importances_
            self.plot_feature_importance()
        
        # Cross-validation scores
        print("\nPerforming 5-fold cross-validation...")
        cv_scores = cross_val_score(
            self.ensemble_model,
            self.scaler.transform(X),
            y,
            cv=5,
            scoring='r2',
            n_jobs=-1
        )
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Average CV R² score: {cv_scores.mean():.4f}")
    
    def plot_feature_importance(self):
        """Plot feature importance based on RandomForest model"""
        if self.feature_importance is None:
            print("No feature importance available.")
            return
        
        feat_imp_df = pd.DataFrame({
            'Feature': self.feature_columns,
            'Importance': self.feature_importance
        }).sort_values(by='Importance', ascending=False)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Importance', y='Feature', data=feat_imp_df)
        plt.title('Feature Importance from Random Forest')
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """Save model and preprocessing objects to file"""
        print(f"Saving model to {filepath} ...")
        model_data = {
            'ensemble_model': self.ensemble_model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print("Model saved successfully.")
    
    def load_model(self, filepath):
        """Load model and preprocessing objects from file"""
        print(f"Loading model from {filepath} ...")
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        self.ensemble_model = model_data['ensemble_model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        print("Model loaded successfully.")


if __name__ == "__main__":
    trainer = LifeExpectancyModelTrainer()
    data_path = "Life Expectancy Data (1).csv"  # Update path if needed

    df = trainer.load_and_preprocess_data(data_path)
    df_selected = trainer.select_features(df)
    trainer.train_and_evaluate(df_selected)

    # Save model and scaler to correct paths
    import os
    os.makedirs('models', exist_ok=True)
    trainer.save_model('models/life_model.pkl')
    
    with open('models/scaler.pkl', 'wb') as f:
        pickle.dump(trainer.scaler, f)

    print("✅ Model and scaler saved to 'models/' folder")
