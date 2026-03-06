import json
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

warnings.filterwarnings("ignore")

CACHE_PATH = Path(os.environ.get("DISCOVERY_CACHE_PATH", "data/latest_discovery.json"))
MAX_CACHE_AGE_DAYS = int(os.environ.get("DISCOVERY_MAX_AGE_DAYS", "10"))


def load_mode():
    return os.environ.get("TRADING_MODE", "PAPER").upper()


def load_credentials(mode: str):
    if mode == "LIVE":
        api_key = os.environ.get("APCA_LIVE_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("APCA_LIVE_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        is_paper = False
    elif mode == "MAX-PAPER":
        api_key = os.environ.get("MAX_APCA_PAPER_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("MAX_APCA_PAPER_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        is_paper = True
    else:
        api_key = os.environ.get("APCA_PAPER_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
        api_secret = os.environ.get("APCA_PAPER_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
        is_paper = True

    if not api_key or not api_secret:
        print(f"ERROR: {mode} API credentials not found; skipping trading run.")
        raise SystemExit(0)

    return api_key, api_secret, is_paper


def load_discovery_cache():
    if not CACHE_PATH.exists():
        raise RuntimeError(f"Discovery cache not found at {CACHE_PATH}")

    cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
    generated_at = datetime.fromisoformat(cache["generated_at_utc"].replace("Z", "+00:00"))
    cache_age = datetime.now(timezone.utc) - generated_at
    if cache_age > timedelta(days=MAX_CACHE_AGE_DAYS):
        raise RuntimeError(
            f"Discovery cache is stale ({cache_age.days} days old). "
            "Refresh discovery before trading."
        )

    return cache, cache_age


def latest_price(symbol: str, cache: dict):
    cached_price = cache.get("latest_prices", {}).get(symbol)
    if cached_price:
        return float(cached_price)

    hist = yf.Ticker(symbol).history(period="5d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch a current price for {symbol}")
    return float(hist["Close"].dropna().iloc[-1])


def long_slippage_cap(symbol: str):
    hist = yf.Ticker(symbol).history(period="1mo")
    if hist.empty:
        return 10000.0
    return float((hist["Volume"] * hist["Close"]).tail(20).mean() * 0.01)


def should_trade_today(trading_client: TradingClient):
    today = datetime.now()
    day = today.day
    positions = trading_client.get_all_positions()
    has_equity = any(position.asset_class == AssetClass.US_EQUITY for position in positions)

    should_trade = False
    trade_reason = ""
    if day == 1:
        should_trade = True
        trade_reason = "1st of the month standard rebalance."
    elif 2 <= day <= 25:
        if not has_equity:
            should_trade = True
            trade_reason = f"Forced trade (Day {day} with empty portfolio)."
        else:
            trade_reason = f"Halted: Positions already exist (Day {day})."
    else:
        trade_reason = f"Halted: Day {day} is in cooldown."

    return should_trade, trade_reason, positions, has_equity


def append_trade_log(mode: str, current_pv: float, current_cash: float, current_bp: float, cache: dict):
    log_path = Path("trading_history_logs.csv")
    header = "Date,Mode,Portfolio_Value,Cash,Buying_Power,Long_Pick,Long_Score,Short_Pick,Short_Score\n"
    log_entry = (
        f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')},{mode},{current_pv},{current_cash},{current_bp},"
        f"{cache['long_pick']},{cache['long_score']:.4f},{cache['short_pick']},{cache['short_score']:.4f}\n"
    )

    if not log_path.exists():
        log_path.write_text(header, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(log_entry)


def main():
    mode = load_mode()
    api_key, api_secret, is_paper = load_credentials(mode)
    cache, cache_age = load_discovery_cache()

    trading_client = TradingClient(api_key, api_secret, paper=is_paper)
    account = trading_client.get_account()
    print(f"{mode} ENGINE ONLINE | Account: {account.account_number}")
    print(f"Buying Power: ${float(account.buying_power):,.2f} | Cash: ${float(account.cash):,.2f}")
    print(
        f"Using discovery cache from {cache['generated_at_utc']} "
        f"({cache_age.days}d {cache_age.seconds // 3600}h old)"
    )
    print(
        f"Cached long/short picks: {cache['long_pick']} ({cache['long_score']:.4f}) / "
        f"{cache['short_pick']} ({cache['short_score']:.4f})"
    )

    should_trade, trade_reason, positions, has_equity = should_trade_today(trading_client)
    print(f"Status: {trade_reason}")
    if not should_trade:
        return

    print("Rebalance triggered: liquidating existing equity positions...")
    for position in positions:
        if position.asset_class == AssetClass.US_EQUITY:
            print(f"Closing {position.symbol}...")
            trading_client.close_position(position.symbol)

    if has_equity:
        print("Waiting 15 seconds for settlements...")
        time.sleep(15)

    account = trading_client.get_account()
    current_bp = float(account.buying_power)
    current_cash = float(account.cash)
    current_pv = float(account.portfolio_value)
    top_pick_long = cache["long_pick"]
    top_pick_short = cache["short_pick"]

    slippage_cap_long = long_slippage_cap(top_pick_long)
    long_budget = min(current_bp * 0.80, slippage_cap_long)
    short_budget = current_bp * 0.20
    order_val_long = round(long_budget * 0.98, 2)
    order_val_short = round(short_budget * 0.98, 2)
    print(
        f"Refreshed BP: ${current_bp:,.2f} | Planning Long: ${order_val_long:,.2f} | "
        f"Planning Short: ${order_val_short:,.2f}"
    )

    if order_val_long >= 1.0:
        try:
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol=top_pick_long,
                    notional=order_val_long,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            print(f"LONG submitted: {top_pick_long} (${order_val_long:,.2f})")
        except Exception as exc:
            print(f"LONG trade failed: {exc}")

    if order_val_short >= 1.0:
        try:
            short_price = latest_price(top_pick_short, cache)
            qty = round(order_val_short / short_price, 2)
            if qty > 0.1:
                trading_client.submit_order(
                    MarketOrderRequest(
                        symbol=top_pick_short,
                        qty=qty,
                        side=OrderSide.SELL,
                        time_in_force=TimeInForce.DAY,
                    )
                )
                print(f"SHORT submitted: {top_pick_short} ({qty} shares)")
        except Exception as exc:
            print(f"SHORT trade failed: {exc}")

    time.sleep(2)
    account = trading_client.get_account()
    rem_cash = float(account.cash)
    if rem_cash >= 5.0:
        try:
            trading_client.submit_order(
                MarketOrderRequest(
                    symbol="ETH/USD",
                    notional=round(rem_cash * 0.98, 2),
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.GTC,
                )
            )
            print("Remaining cash swept to ETH vault")
        except Exception:
            pass

    append_trade_log(mode, current_pv, current_cash, current_bp, cache)
    print("Trading run complete.")


if __name__ == "__main__":
    main()
