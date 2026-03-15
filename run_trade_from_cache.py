import os
import json
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

def execute_from_cache():
    # Load discovery results
    with open('data/latest_discovery.json', 'r') as f:
        discovery = json.load(f)
    
    long_pick = discovery.get('long_pick')
    short_pick = discovery.get('short_pick')
    
    # Initialize trading client
    api_key = os.environ.get('APCA_LIVE_API_KEY_ID')
    api_secret = os.environ.get('APCA_LIVE_API_SECRET_KEY')
    trading_client = TradingClient(api_key, api_secret, paper=False)
    
    # Place orders
    if long_pick:
        print(f"Buying {long_pick}")
        trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=long_pick,
                qty=1,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY
            )
        )
    
    if short_pick:
        print(f"Selling {short_pick}")
        trading_client.submit_order(
            order_data=MarketOrderRequest(
                symbol=short_pick,
                qty=1,
                side=OrderSide.SELL,
                time_in_force=TimeInForce.DAY
            )
        )

if __name__ == '__main__':
    execute_from_cache()