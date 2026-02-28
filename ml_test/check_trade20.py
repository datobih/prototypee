import pandas as pd

df = pd.read_csv('data/processed/ob_combos/top1_Fib_38-62pct__FVG.csv')
row = df.iloc[19]

print('='*70)
print('Trade #20 Details (BEAR OB - 2025-11-20 15:00)')
print('='*70)
print(f'Type:                {row["type"]}')
print(f'OB datetime:         {row["ob_dt"]}')
print(f'OB high:             ${row["ob_high"]:.2f}')
print(f'OB low:              ${row["ob_low"]:.2f}')
print(f'Zone size:           ${row["ob_high"] - row["ob_low"]:.2f}')
print(f'')
print(f'Reaction:            {row["reaction"]}')
print(f'Trade RR:            {row["trade_rr"]:.2f}')
print(f'Fib retrace:         {row["fib_retracement_pct"]:.1f}%')
print(f'Touch quality:       {row["touch_close_quality"]}')
print(f'')
print(f'Displacement:        {row["disp_atr"]:.2f}x ATR')
print(f'Impulse body:        {row["impulse_body_atr"]:.2f}x ATR')
print(f'Consecutive impulse: {row["consec_impulse"]}')
print(f'')
print(f'Has FVG:             {row["has_fvg"]}')
print(f'With structure:      {row["with_structure"]}')
print(f'Strong trend:        {row["strong_trend"]}')
print(f'')
print(f'Bars to mitigate:    {row["bars_to_mitigate"]}')
print(f'Max adverse:         {row["max_adverse_atr"]:.2f}x ATR')
print(f'Reaction (favor):    {row["reaction_atr"]:.2f}x ATR')
print(f'')
print(f'Session:             {row["session"]}')
print(f'OB vol relative:     {row["ob_vol_rel"]:.2f}x VMA')
print(f'Impulse vol rel:     {row["imp_vol_rel"]:.2f}x VMA')

print('\n' + '='*70)
print('VALIDITY CHECK')
print('='*70)

# Check if it meets the combo criteria
is_fib_valid = 38 <= row["fib_retracement_pct"] <= 62
has_fvg = row["has_fvg"] == True

print(f'Fib 38-62%:          {is_fib_valid} ({row["fib_retracement_pct"]:.1f}%)')
print(f'Has FVG:             {has_fvg}')
print(f'Combo valid:         {is_fib_valid and has_fvg}')

# Load price data to verify
print('\n' + '='*70)
print('PRICE ACTION VERIFICATION')
print('='*70)

df_price = pd.read_csv('data/raw/XAUUSD30.csv', sep='\t', header=0,
                       names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                              'TickVol', 'Vol', 'Spread'])
df_price['Datetime'] = pd.to_datetime(df_price['Date'] + ' ' + df_price['Time'])
df_price.set_index('Datetime', inplace=True)

ob_dt = pd.to_datetime(row['ob_dt'])
ob_pos = df_price.index.get_loc(ob_dt)

print(f'\nOB candle (bar {ob_pos}):')
ob_bar = df_price.iloc[ob_pos]
print(f'  {ob_dt}')
print(f'  O: ${ob_bar["Open"]:.2f}  H: ${ob_bar["High"]:.2f}  '
      f'L: ${ob_bar["Low"]:.2f}  C: ${ob_bar["Close"]:.2f}')
print(f'  Bearish: {ob_bar["Close"] < ob_bar["Open"]}')

# Find impulse candle (should be next bullish candle after OB)
print(f'\nNext 5 bars after OB:')
for i in range(1, 6):
    bar = df_price.iloc[ob_pos + i]
    bullish = bar['Close'] > bar['Open']
    body = abs(bar['Close'] - bar['Open'])
    print(f'  Bar +{i}: {"BULL" if bullish else "BEAR"}  '
          f'O: ${bar["Open"]:.2f}  H: ${bar["High"]:.2f}  '
          f'L: ${bar["Low"]:.2f}  C: ${bar["Close"]:.2f}  '
          f'Body: ${body:.2f}')

# Check for swing low formation
print(f'\nBars {int(row["bars_to_mitigate"])} bars later (touch):')
touch_pos = ob_pos + int(row["bars_to_mitigate"])
if touch_pos < len(df_price):
    touch_bar = df_price.iloc[touch_pos]
    print(f'  {df_price.index[touch_pos]}')
    print(f'  O: ${touch_bar["Open"]:.2f}  H: ${touch_bar["High"]:.2f}  '
          f'L: ${touch_bar["Low"]:.2f}  C: ${touch_bar["Close"]:.2f}')
    print(f'  Touch high: ${touch_bar["High"]:.2f} vs OB low: ${row["ob_low"]:.2f}')
    print(f'  Touched zone: {touch_bar["High"] >= row["ob_low"]}')

print('\n' + '='*70)
