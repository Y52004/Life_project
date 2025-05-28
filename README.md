Life Expectancy Prediction Web Application
A machine learning-powered web application that predicts life expectancy based on various health, economic, and social factors using WHO (World Health Organization) data.
📊 Dataset Overview
This project uses the Life Expectancy (WHO) dataset for statistical analysis on factors influencing Life Expectancy. The dataset contains comprehensive health and socio-economic indicators from 193 countries spanning from 2000 to 2015.
Key Features (19 Variables):

Health Factors: Adult Mortality, Infant Deaths, HIV/AIDS, Hepatitis B, Measles, Polio, Diphtheria
Lifestyle Indicators: BMI, Alcohol consumption, Thinness prevalence (1-19 years, 5-9 years)
Economic Indicators: GDP, Percentage expenditure on health, Total expenditure on health
Social Factors: Schooling years, Income composition of resources
Demographics: Population, Under-five deaths
Development Status: Developed vs Developing countries

🎯 Project Features

Interactive Web Interface: User-friendly forms for data input
Real-time Predictions: Instant life expectancy predictions with confidence intervals
Categorized Results: Classifications from "Very Low" to "Excellent" life expectancy
Model Training Interface: Web-based model retraining functionality
REST API: JSON endpoints for programmatic access
Responsive Design: Works on desktop and mobile devices

🏗️ Project Structure
├── models/
│   ├── life_model.pkl      # Trained ensemble model
│   └── scaler.pkl          # Feature scaler
├── templates/
│   ├── base.html           # Prediction form
│   ├── homepagetemplate.html   # Landing page
│   └── result.html         # Results display
├── app.py                  # Flask web application
├── train_model.py          # Model training script
├── Life Expectancy Data (1).csv   # Training dataset
└── requirements.txt        # Python dependencies
🤖 Machine Learning Approach
Model Architecture:

Ensemble Method: Combines multiple algorithms for improved accuracy
Feature Engineering: Handles missing values and categorical encoding
Scaling: StandardScaler for feature normalization
Cross-validation: Ensures robust model performance

Key Features Used:

Adult Mortality Rate
GDP per capita
Schooling years
BMI indicators
Disease immunization rates
Health expenditure metrics

🚀 Installation & Setup
Prerequisites:

Python 3.8+
pip package manager

Local Installation:

Clone the repository:

bashgit clone https://github.com/yourusername/life-expectancy-prediction.git
cd life-expectancy-prediction

Create virtual environment:

bashpython -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:

bashpip install -r requirements.txt

Train the model (if needed):

bashpython train_model.py

Run the application:

bashpython app.py

Access the app:

Open browser to http://localhost:5000



🌐 Deployment
Render Deployment:

Push to GitHub
Connect Render to your repository
Set build settings:

Build Command: pip install -r requirements.txt
Start Command: python app.py
Environment: Python 3



Environment Variables:

PORT: Automatically set by Render
PYTHON_VERSION: 3.9.18 (recommended)

📱 API Usage
Prediction Endpoint:
bashPOST /api/predict
Content-Type: application/json

{
  "Adult Mortality": 263,
  "infant deaths": 62,
  "Alcohol": 0.01,
  "percentage expenditure": 71.279624,
  "Hepatitis B": 65,
  "Measles": 1154,
  "BMI": 19.1,
  "GDP": 584.259210,
  "Population": 33736494,
  "Schooling": 10.1,
  "Status_encoded": 1
}
Response:
json{
  "life_expectancy": 65.2,
  "category": "Average Life Expectancy",
  "alert_type": "info",
  "success": true
}
📈 Model Performance

Accuracy: Achieves high R² score on test data
Features: 19 input variables for comprehensive prediction
Validation: Cross-validated for reliability
Robustness: Handles missing values and outliers

🎨 Life Expectancy Categories
RangeCategoryAlert Type< 50 yearsVery LowDanger50-60 yearsLowWarning60-70 yearsAverageInfo70-75 yearsGoodSuccess> 75 yearsExcellentPrimary
🔧 Technologies Used

Backend: Flask (Python)
ML Libraries: scikit-learn, pandas, numpy
Frontend: HTML5, CSS3, Bootstrap
Deployment: Render, GitHub
Data Processing: pickle, pandas

📊 Data Sources

Primary: WHO's Global Health Estimates provide latest available data on causes of death globally, by region, by sex and by income group
Coverage: 193 countries, 2000-2015
Quality: Cleaned and preprocessed for ML applications
