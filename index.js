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