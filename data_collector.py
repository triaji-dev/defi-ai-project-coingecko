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