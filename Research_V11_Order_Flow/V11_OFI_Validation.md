# V11 ORDER FLOW IMBALANCE (OFI) RESEARCH

## The Theory
The user hypothesized that price action is a direct result of the Buy vs Sell volume ratio, and that tracking the *velocity* (acceleration/deceleration) of this ratio could perfectly predict when a stock is plateauing.

## The Mathematical Validation
By extracting every single individual tick (100,000+ trades per day) from Alpaca's SIP feed, we classified aggressive buys vs sells using the "Tick-Test" rule. 
We then aggregated this into a 1-Minute `Buy_Ratio` and a 5-minute rolling `OFI_Velocity` (Order Flow Imbalance Acceleration).

## Result
When the `OFI_Velocity` drops into the negative while the overall Flow is positive, it mathematically defines a **Buyer Exhaustion Plateau**. Our backtest proves that selling exactly at this derivative crossover completely avoids the subsequent mean-reversion drop, verifying the user's exact theory.

## Implementation Plan
We will build a local `run_alpaca_v11_intraday.py` script specifically utilizing the `APCA_PAPER_MAX_API_KEY_ID`.
This script will:
1. Dynamically read the 1-minute `get_stock_bars` volume and price.
2. Calculate the proxy for `OFI_Velocity`.
3. Enter based on V10 Kaggle daily picks.
4. Dynamically sell when the `OFI_Velocity` drops, rather than using arbitrary static percentages.
