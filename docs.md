# AI Crypto Price Prediction System - Complete Implementation Guide

## Project Overview
Build a complete AI-powered cryptocurrency price prediction system using Node.js, Python, and machine learning. This system fetches historical market data from CoinGecko API, trains a linear regression model, and provides real-time price predictions through a REST API endpoint.

## Prerequisites

### Required Software
- **Node.js** (v14 or higher) and npm
- **Python 3.7+** and pip
- **CoinGecko API Key** (Free Demo plan available at https://www.coingecko.com/en/api)

### Skills Required
- Basic understanding of JavaScript/Node.js
- Basic Python knowledge
- Familiarity with REST APIs
- Command line/terminal usage

## Project Setup

### Step 1: Initialize Project Directory
```bash
mkdir defi-ai-project
cd defi-ai-project
```

### Step 2: Python Environment Setup

#### Create Virtual Environment
**Windows:**
```bash
python -m venv venv
venv\Scripts\activate (powershell or cmd)
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Install Python Dependencies
```bash
pip install numpy pandas scikit-learn requests
```

```bash
python.exe -m pip install --upgrade pip
```

```bash
pip install requests pandas python-dotenv (di bash)
```

**Libraries explanation:**
- `numpy`: Numerical computing
- `pandas`: Data manipulation and analysis
- `scikit-learn`: Machine learning models
- `requests`: HTTP API requests

### Step 3: Node.js Project Setup

#### Initialize Node.js Project
```bash
npm init -y
```

#### Install Node.js Dependencies
```bash
npm install express axios dotenv
```

**Packages explanation:**
- `express`: Web server framework
- `axios`: HTTP client for API requests
- `dotenv`: Environment variable management

### Step 4: Final Project Structure
```
defi-ai-project/
├── data_collector.py      # Fetches historical market data
├── ai_model.py           # AI prediction model
├── index.js              # Express server with API endpoints
├── package.json          # Node.js dependencies
├── .env                  # Environment variables (create this)
├── market_data.csv       # Generated data file
└── venv/                 # Python virtual environment
```

## Implementation

### File 1: Environment Configuration (.env)
Create a `.env` file in the root directory:

```env
COINGECKO_API_KEY=your_actual_api_key_here
PORT=3000
```

### File 2: Data Collection Script (data_collector.py)

```python
import requests
import pandas as pd
import os
from dotenv import load_dotenv

# Load environment variables (optional untuk free API)
load_dotenv()
COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')

def fetch_market_data(coin_id='bitcoin', days=30, vs_currency='usd'):
    """
    Fetch historical market data from CoinGecko API
    
    Args:
        coin_id: Cryptocurrency identifier (default: 'bitcoin')
        days: Number of days of historical data (default: 30)
        vs_currency: Target currency (default: 'usd')
    
    Returns:
        DataFrame with timestamp, price, and volume data
    """
    url = 'https://api.coingecko.com/api/v3/coins/{}/market_chart'.format(coin_id)
    
    # Untuk FREE API, tidak perlu x_cg_pro_api_key
    params = {
        'vs_currency': vs_currency,
        'days': str(days),
    }
    
    try:
        print(f"Fetching {days} days of {coin_id} market data...")
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        # Extract price and volume data
        prices = data.get('prices', [])
        volumes = data.get('total_volumes', [])
        
        if not prices or not volumes:
            raise ValueError("No data received from API")
        
        print(f"Data points received: {len(prices)}")

        # Create DataFrames
        df_prices = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df_volumes = pd.DataFrame(volumes, columns=['timestamp', 'volume'])

        # Merge on timestamp
        df = pd.merge(df_prices, df_volumes, on='timestamp')

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Save to CSV
        output_file = 'market_data.csv'
        df.to_csv(output_file, index=False)
        print(f"✓ Market data saved to {output_file}")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
        
        return df

    except requests.exceptions.HTTPError as http_err:
        print(f"✗ HTTP error: {http_err}")
        print(f"  Response: {response.text if 'response' in locals() else 'No response'}")
    except Exception as err:
        print(f"✗ Error: {err}")
    
    return None

if __name__ == '__main__':
    df = fetch_market_data()
    if df is not None:
        print("\nData collection completed successfully!")
    else:
        print("\nData collection failed. Please check your API key and connection.")
```

### File 3: AI Prediction Model (ai_model.py)

```python
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
```

### File 4: Express Server (index.js)

```javascript
const express = require('express');
const axios = require('axios');
const { execFile } = require('child_process');
const path = require('path');
require('dotenv').config();

const app = express();
app.use(express.json());

// Configuration
const CONFIG = {
  PORT: process.env.PORT || 3000,
  COINGECKO_API_KEY: process.env.COINGECKO_API_KEY,
  PYTHON_PATH: process.platform === 'win32' ? 'python' : 'python3',
  COIN_ID: 'bitcoin',
  VS_CURRENCY: 'usd'
};

/**
 * Execute Python AI model and get prediction
 * @param {number} latestPrice - Current cryptocurrency price
 * @param {number} latestVolume - Current trading volume
 * @returns {Promise<Object>} Prediction results
 */
function getAIPrediction(latestPrice, latestVolume) {
  return new Promise((resolve, reject) => {
    const pythonScript = path.join(__dirname, 'ai_model.py');
    const args = [pythonScript, latestPrice.toString(), latestVolume.toString()];
    
    execFile(CONFIG.PYTHON_PATH, args, { cwd: __dirname }, (error, stdout, stderr) => {
      if (error) {
        console.error('❌ Error executing Python script:', error);
        return reject(new Error(`Python execution failed: ${error.message}`));
      }
      
      if (stderr) {
        console.warn('⚠️  Python stderr:', stderr);
      }
      
      try {
        const result = JSON.parse(stdout);
        if (result.error) {
          return reject(new Error(result.error));
        }
        resolve(result);
      } catch (parseError) {
        console.error('❌ Error parsing Python output:', stdout);
        reject(new Error(`Failed to parse prediction: ${parseError.message}`));
      }
    });
  });
}

/**
 * Fetch current market data from CoinGecko
 * @returns {Promise<Object>} Market data with price and volume
 */
async function getCurrentMarketData() {
  try {
    const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
      params: {
        ids: CONFIG.COIN_ID,
        vs_currencies: CONFIG.VS_CURRENCY,
        include_24hr_vol: true
      }
    });
    
    const data = response.data[CONFIG.COIN_ID];
    return {
      price: data[CONFIG.VS_CURRENCY],
      volume: data[`${CONFIG.VS_CURRENCY}_24h_vol`] || 0
    };
  } catch (error) {
    throw new Error(`Failed to fetch market data: ${error.message}`);
  }
}

/**
 * Generate trading signal based on prediction
 * @param {number} currentPrice - Current price
 * @param {number} predictedPrice - Predicted price
 * @returns {Object} Trading signal and reasoning
 */
function generateTradingSignal(currentPrice, predictedPrice) {
  const changePercent = ((predictedPrice - currentPrice) / currentPrice) * 100;
  
  let signal, reasoning, confidence;
  
  if (changePercent > 1.5) {
    signal = 'BUY';
    reasoning = `Price predicted to increase by ${changePercent.toFixed(2)}%`;
    confidence = 'STRONG';
  } else if (changePercent > 0.5) {
    signal = 'BUY';
    reasoning = `Moderate price increase expected (${changePercent.toFixed(2)}%)`;
    confidence = 'MODERATE';
  } else if (changePercent < -1.5) {
    signal = 'SELL';
    reasoning = `Price predicted to decrease by ${Math.abs(changePercent).toFixed(2)}%`;
    confidence = 'STRONG';
  } else if (changePercent < -0.5) {
    signal = 'SELL';
    reasoning = `Moderate price decrease expected (${Math.abs(changePercent).toFixed(2)}%)`;
    confidence = 'MODERATE';
  } else {
    signal = 'HOLD';
    reasoning = `Price expected to remain stable (${changePercent.toFixed(2)}% change)`;
    confidence = 'LOW';
  }
  
  return { signal, reasoning, confidence, changePercent: changePercent.toFixed(2) };
}

// ============================================
// API ENDPOINTS
// ============================================

/**
 * GET /predict_price
 * Returns current price and AI prediction with trading signal
 */
app.get('/predict_price', async (req, res) => {
  try {
    console.log('📊 Processing prediction request...');
    
    // Fetch current market data
    const marketData = await getCurrentMarketData();
    console.log(`💰 Current ${CONFIG.COIN_ID} price: $${marketData.price}`);
    
    // Get AI prediction
    const aiResult = await getAIPrediction(marketData.price, marketData.volume);
    
    // Generate trading signal
    const tradingSignal = generateTradingSignal(
      marketData.price,
      aiResult.prediction
    );
    
    // Prepare response
    const response = {
      cryptocurrency: CONFIG.COIN_ID,
      timestamp: new Date().toISOString(),
      current_price: {
        value: marketData.price,
        currency: CONFIG.VS_CURRENCY.toUpperCase()
      },
      predicted_price: {
        value: parseFloat(aiResult.prediction.toFixed(2)),
        currency: CONFIG.VS_CURRENCY.toUpperCase()
      },
      prediction_change: aiResult.predicted_change_pct,
      trading_signal: tradingSignal,
      model_performance: aiResult.model_metrics,
      confidence: aiResult.confidence
    };
    
    console.log(`✅ Prediction: $${response.predicted_price.value} (${tradingSignal.signal})`);
    
    res.json(response);
    
  } catch (error) {
    console.error('❌ Prediction error:', error.message);
    res.status(500).json({
      error: 'Prediction failed',
      message: error.message,
      timestamp: new Date().toISOString()
    });
  }
});

/**
 * GET /health
 * Health check endpoint
 */
app.get('/health', (req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

/**
 * GET /
 * API documentation
 */
app.get('/', (req, res) => {
  res.json({
    name: 'Crypto AI Prediction API',
    version: '1.0.0',
    endpoints: {
      '/predict_price': 'Get AI price prediction and trading signal',
      '/health': 'Health check',
      '/': 'API documentation'
    },
    usage: 'GET http://localhost:' + CONFIG.PORT + '/predict_price'
  });
});

// ============================================
// SERVER INITIALIZATION
// ============================================

/**
 * Initialize server and run initial prediction
 */
async function initializeServer() {
  try {
    console.log('\n🚀 Starting Crypto AI Prediction Server...\n');
    
    // Fetch initial market data
    const marketData = await getCurrentMarketData();
    console.log(`💰 Current ${CONFIG.COIN_ID.toUpperCase()} Price: $${marketData.price}`);
    console.log(`📈 24h Volume: $${marketData.volume.toLocaleString()}\n`);
    
    // Get initial AI prediction
    console.log('🤖 Running AI prediction model...');
    const aiResult = await getAIPrediction(marketData.price, marketData.volume);
    
    console.log(`🎯 AI Predicted Price: $${aiResult.prediction.toFixed(2)}`);
    console.log(`📊 Price Change: ${aiResult.predicted_change_pct}%`);
    console.log(`🔍 Model R² Score: ${aiResult.model_metrics.r2_score.toFixed(4)}`);
    console.log(`📉 Model RMSE: $${aiResult.model_metrics.rmse.toFixed(2)}\n`);
    
    // Generate and display trading signal
    const signal = generateTradingSignal(marketData.price, aiResult.prediction);
    console.log('🎲 TRADING SIGNAL:', signal.signal);
    console.log('💡 Reasoning:', signal.reasoning);
    console.log('⚡ Confidence:', signal.confidence);
    console.log('\n' + '='.repeat(60) + '\n');
    
  } catch (error) {
    console.error('❌ Initialization error:', error.message);
    console.error('💡 Make sure to run data_collector.py first!\n');
  }
}

// Start server
app.listen(CONFIG.PORT, async () => {
  console.log(`\n${'='.repeat(60)}`);
  console.log(`  🌟 Server running on http://localhost:${CONFIG.PORT}`);
  console.log(`${'='.repeat(60)}\n`);
  
  await initializeServer();
  
  console.log('📡 API Endpoints Ready:');
  console.log(`   • http://localhost:${CONFIG.PORT}/predict_price`);
  console.log(`   • http://localhost:${CONFIG.PORT}/health\n`);
});

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('\n👋 Shutting down gracefully...');
  process.exit(0);
});
```

## Execution Instructions

### Step 1: Collect Historical Data
First, activate your Python virtual environment and collect market data:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Run data collector
python data_collector.py
```

**Expected Output:**
```
Fetching 30 days of bitcoin market data...
Data points received: 720
✓ Market data saved to market_data.csv
  Date range: 2024-09-29 to 2024-10-29
  Price range: $58423.21 - $69850.43

Data collection completed successfully!
```

### Step 2: Test AI Model
Verify the AI model works correctly:

```bash
python ai_model.py
```

**Expected Output:**
```json
{
  "prediction": 67234.56,
  "current_price": 67180.32,
  "predicted_change_pct": 0.08,
  "model_metrics": {
    "mse": 123456.78,
    "rmse": 351.36,
    "r2_score": 0.9876
  },
  "confidence": "high"
}
```

### Step 3: Start the Server
```bash
node index.js
```

**Expected Output:**
```
============================================================
  🌟 Server running on http://localhost:3000
============================================================

🚀 Starting Crypto AI Prediction Server...

💰 Current BITCOIN Price: $67180.32
📈 24h Volume: $32,456,789,012

🤖 Running AI prediction model...
🎯 AI Predicted Price: $67234.56
📊 Price Change: 0.08%
🔍 Model R² Score: 0.9876
📉 Model RMSE: $351.36

🎲 TRADING SIGNAL: HOLD
💡 Reasoning: Price expected to remain stable (0.08% change)
⚡ Confidence: LOW

============================================================

📡 API Endpoints Ready:
   • http://localhost:3000/predict_price
   • http://localhost:3000/health
```

### Step 4: Test the API
Open a new terminal and test the prediction endpoint:

```bash
curl http://localhost:3000/predict_price
```

**Expected Response:**
```json
{
  "cryptocurrency": "bitcoin",
  "timestamp": "2024-10-29T10:30:45.123Z",
  "current_price": {
    "value": 67180.32,
    "currency": "USD"
  },
  "predicted_price": {
    "value": 67234.56,
    "currency": "USD"
  },
  "prediction_change": "0.08",
  "trading_signal": {
    "signal": "HOLD",
    "reasoning": "Price expected to remain stable (0.08% change)",
    "confidence": "LOW",
    "changePercent": "0.08"
  },
  "model_performance": {
    "mse": 123456.78,
    "rmse": 351.36,
    "r2_score": 0.9876
  },
  "confidence": "high"
}
```

## Trading Signal Logic

The system generates trading signals based on predicted price changes:

| Signal | Condition | Confidence |
|--------|-----------|------------|
| **STRONG BUY** | Predicted increase > 1.5% | STRONG |
| **BUY** | Predicted increase 0.5% - 1.5% | MODERATE |
| **HOLD** | Predicted change -0.5% to +0.5% | LOW |
| **SELL** | Predicted decrease -1.5% to -0.5% | MODERATE |
| **STRONG SELL** | Predicted decrease < -1.5% | STRONG |

## Troubleshooting

### Common Issues and Solutions

1. **"Python script execution failed"**
   - Ensure Python is in your PATH
   - Try changing `PYTHON_PATH` in index.js to `python3` or the full path

2. **"Data file not found"**
   - Run `python data_collector.py` before starting the server
   - Ensure `market_data.csv` exists in the project directory

3. **"API Key invalid"**
   - Verify your CoinGecko API key in `.env` file
   - Ensure no extra spaces or quotes around the key

4. **"Module not found" errors**
   - Reinstall dependencies: `pip install -r requirements.txt`
   - For Node.js: `npm install`

5. **Port already in use**
   - Change PORT in `.env` file
   - Or kill the process using port 3000

## Enhancement Ideas

### Beginner Level
1. Add support for multiple cryptocurrencies (Ethereum, BNB, etc.)
2. Store predictions in a database for historical tracking
3. Create a simple web dashboard with HTML/CSS

### Intermediate Level
4. Implement more sophisticated ML models (Random Forest, LSTM)
5. Add technical indicators (RSI, MACD, Bollinger Bands)
6. Create automated trading bot integration
7. Add email/Telegram notifications for signals

### Advanced Level
8. Implement real-time WebSocket data streaming
9. Add sentiment analysis from news/social media
10. Create portfolio optimization algorithms
11. Implement backtesting framework
12. Add risk management and position sizing

## Security Considerations

⚠️ **Important Security Notes:**
- Never commit `.env` file to version control
- Use environment variables for all sensitive data
- Implement rate limiting for production APIs
- Add authentication for API endpoints
- Validate all user inputs
- Use HTTPS in production
- This is for educational purposes - do not use for actual trading without proper risk management

## Resources and Documentation

- **CoinGecko API Docs**: https://docs.coingecko.com/
- **scikit-learn Documentation**: https://scikit-learn.org/
- **Express.js Guide**: https://expressjs.com/
- **Pandas Documentation**: https://pandas.pydata.org/

## License and Disclaimer

This project is for educational purposes only. Cryptocurrency trading carries significant risk. Never invest more than you can afford to lose. Past performance does not guarantee future results. Always do your own research (DYOR) before making trading decisions.

---

**Project Complete! You now have a fully functional AI crypto prediction system.** 🎉