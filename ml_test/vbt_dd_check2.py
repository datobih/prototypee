import pandas as pd, numpy as np, vectorbt as vbt

df = pd.read_csv('data/raw/XAUUSD15.csv', sep='\t', names=['Date','Time','Open','High','Low','Close','TickVol','Vol','Spread'])
df['Datetime'] = pd.to_datetime(df['Date']+' '+df['Time'], format='%Y.%m.%d %H:%M:%S')
price = df.set_index('Datetime').dropna(subset=['Close']).loc['2025-01-01':'2025-12-31']['Close']

sma_f = vbt.MA.run(price, window=40).ma
sma_s = vbt.MA.run(price, window=50).ma
rsi   = vbt.RSI.run(price, window=14).rsi

pf = vbt.Portfolio.from_signals(price,
    entries=(sma_f.shift(1)<=sma_s.shift(1))&(sma_f>sma_s)&(rsi>50),
    short_entries=(sma_f.shift(1)>=sma_s.shift(1))&(sma_f<sma_s)&(rsi<50),
    upon_opposite_entry='close', size=1.0, size_type='amount',
    init_cash=100000, fees=0.0001, sl_stop=0.002, tp_stop=0.005, freq='15min')

trades = pf.trades.records_readable
losing = trades[trades['PnL'] < 0].sort_values('PnL')

print("=== WORST 10 LOSING TRADES ===")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 160)
print(losing[['Entry Index', 'Exit Index', 'Direction', 'Entry Price', 'Exit Price', 'PnL', 'Exit Type']].head(10))

print("\n=== EXIT TYPE BREAKDOWN ===")
print(trades['Exit Type'].value_counts())

print("\n=== LOSSES BY EXIT TYPE ===")
for exit_type, group in trades[trades['PnL'] < 0].groupby('Exit Type'):
    print(f"  {exit_type}: count={len(group)}, avg loss=${group['PnL'].mean():.2f}, worst=${group['PnL'].min():.2f}")
