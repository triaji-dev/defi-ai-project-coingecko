import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import sys
import json
import warnings
import os

warnings.filterwarnings("ignore")

def load_and_prepare_data(csv_file='market_data.csv'):
    """
    Load market data and prepare features for training
    
    Args:
        csv_file: Path to the CSV file containing market data
    
    Returns:
        X, y: Feature matrix and target vector
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"Data file '{csv_file}' not found. Run data_collector.py first.")
    
    # Load historical market data
    data = pd.read_csv(csv_file)
    
    # Feature Engineering: Use previous price as predictor
    data['prev_price'] = data['price'].shift(1)
    data['price_change'] = data['price'] - data['prev_price']
    data['volume_ma'] = data['volume'].rolling(window=3).mean()
    
    # Drop rows with NaN values
    data = data.dropna()
    
    if len(data) < 10:
        raise ValueError("Insufficient data for training. Need at least 10 data points.")
    
    # Prepare features and target
    X = data[['prev_price', 'volume']]
    y = data['price']
    
    return X, y, data

def train_model(X, y):
    """
    Train the linear regression model
    
    Args:
        X: Feature matrix
        y: Target vector
    
    Returns:
        model: Trained LinearRegression model
        metrics: Dictionary containing model performance metrics
    """
    # Split data for validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    metrics = {
        'mse': float(mse),
        'rmse': float(np.sqrt(mse)),
        'r2_score': float(r2)
    }
    
    return model, metrics

def predict_price(model, latest_price, latest_volume):
    """
    Make price prediction using the trained model
    
    Args:
        model: Trained model
        latest_price: Current price
        latest_volume: Current trading volume
    
    Returns:
        Predicted price
    """
    prediction = model.predict([[latest_price, latest_volume]])
    return float(prediction[0])

def main():
    """
    Main execution function
    """
    try:
        # Load and prepare data
        X, y, data = load_and_prepare_data()
        
        # Train model
        model, metrics = train_model(X, y)
        
        # Get input from command line or use latest known values
        if len(sys.argv) >= 3:
            latest_price = float(sys.argv[1])
            latest_volume = float(sys.argv[2])
        else:
            latest_price = float(X.iloc[-1]['prev_price'])
            latest_volume = float(X.iloc[-1]['volume'])
        
        # Make prediction
        predicted_price = predict_price(model, latest_price, latest_volume)
        
        # Calculate confidence metrics
        price_change_pct = ((predicted_price - latest_price) / latest_price) * 100
        
        # Output results as JSON
        result = {
            'prediction': predicted_price,
            'current_price': latest_price,
            'predicted_change_pct': round(price_change_pct, 2),
            'model_metrics': metrics,
            'confidence': 'high' if abs(price_change_pct) < 2 else 'moderate' if abs(price_change_pct) < 5 else 'low'
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'prediction': None
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == '__main__':
    main()