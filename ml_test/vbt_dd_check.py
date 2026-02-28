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
cum_pnl     = trades['PnL'].cumsum()
rolling_max = cum_pnl.cummax()
actual_dd   = (cum_pnl - rolling_max).min()

print("Total trades:            ", pf.trades.count())
print("Total PnL:               $", round(pf.total_profit(), 2))
print("Actual max PnL drawdown: $", round(actual_dd, 2))
print("Worst single trade loss: $", round(trades['PnL'].min(), 2))
print("Avg loss per trade:      $", round(trades[trades['PnL'] < 0]['PnL'].mean(), 2))
print("Max consecutive losses estimate:")
# Count max losing streak
streak, max_streak = 0, 0
for pnl in trades['PnL']:
    if pnl < 0:
        streak += 1
        max_streak = max(max_streak, streak)
    else:
        streak = 0
print("  Actual max losing streak:", max_streak, "trades in a row")
print("  = $", round(max_streak * trades[trades['PnL'] < 0]['PnL'].mean(), 2), "worst run")
