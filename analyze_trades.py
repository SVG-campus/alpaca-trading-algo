import pandas as pd

df = pd.read_csv('optimal_trades_analysis.csv')

print("\n--- ANALYSIS OF LOSING TRADES (STOP_LOSS) ---")
bad_stops = df[df['reason'] == 'STOP_LOSS'].copy()

if not bad_stops.empty:
    print(f"Total Stop Losses Hit: {len(bad_stops)}")
    print(f"Avg Loss Taken: {bad_stops['profit_pct'].mean():.2f}%")
    print(f"Avg Max Loss Avoided: {bad_stops['max_loss_avoided'].mean():.2f}% (If we hadn't stopped out, it would have crashed this far)")
    
    # Calculate fakeouts (trades that stopped us out, then rallied massively)
    fakeouts = bad_stops[bad_stops['gain_after_loss'] > 1.0]
    print(f"Total Fakeouts (Stopped out but rallied >1% after): {len(fakeouts)}")
    if not fakeouts.empty:
        print(f"Avg Gain on Fakeouts: {fakeouts['gain_after_loss'].mean():.2f}%")
        print(f"Avg Time to Peak after Fakeout: {fakeouts['time_to_peak_after_loss_secs'].mean() / 60:.1f} minutes")
        print("\nTop 5 Worst Fakeouts:")
        print(fakeouts[['symbol', 'profit_pct', 'gain_after_loss', 'max_loss_avoided']].sort_values('gain_after_loss', ascending=False).head())
else:
    print("No stop losses hit!")

print("\n--- ANALYSIS OF WINNING TRADES (TRAILING_STOP / TARGET) ---")
winners = df[df['reason'].isin(['TRAILING_STOP', 'TARGET'])].copy()

if not winners.empty:
    print(f"Total Winners Hit: {len(winners)}")
    print(f"Avg Profit Taken: {winners['profit_pct'].mean():.2f}%")
    print(f"Avg Profit Left on Table: {winners['profit_left_on_table'].mean():.2f}%")
    
    early_exits = winners[winners['profit_left_on_table'] > 1.0]
    print(f"Total Early Exits (Left >1% on table): {len(early_exits)}")
    if not early_exits.empty:
        print("\nTop 5 Most Profit Left on Table:")
        print(early_exits[['symbol', 'profit_pct', 'profit_left_on_table', 'max_possible_profit']].sort_values('profit_left_on_table', ascending=False).head())
else:
    print("No winners hit!")
