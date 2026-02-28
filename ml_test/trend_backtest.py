import pandas as pd
import numpy as np
import os

# ============================================================================
# TREND FOLLOWING BACKTEST — XAUUSD Daily + 4H
#
# Logic:
#   1. Daily trend filter:
#      UP   = Close > 200 EMA  AND  50 EMA > 200 EMA  AND  ADX >= 25
#      DOWN = Close < 200 EMA  AND  50 EMA < 200 EMA  AND  ADX >= 25
#      NONE = everything else (no trades)
#
#   2. 4H pullback entry:
#      LONG  : prev 4H bar closed BELOW 21 EMA, current closes ABOVE 21 EMA
#              RSI(14) between 35-65 at entry
#      SHORT : prev 4H bar closed ABOVE 21 EMA, current closes BELOW 21 EMA
#              RSI(14) between 35-65 at entry
#
#   3. Stop loss  : 1.5x ATR(14) beyond entry bar low/high
#   4. Take profit: 2.5x risk (2.5:1 R:R)
#   5. Trend exit : daily trend flips while in trade -> close at next 4H bar
#   6. One trade at a time
#
# P&L in $ assuming $1 per point (scale to your lot size)
# ============================================================================

DATA_DAILY  = 'data/raw/XAUUSD_Daily.csv'
DATA_4H     = 'data/raw/XAUUSDH4.csv'
OUT_PATH    = 'data/processed/trend_backtest_results.csv'

# Daily trend parameters
EMA_FAST      = 50
EMA_SLOW      = 200
ADX_PERIOD    = 14
ADX_THRESHOLD = 25

# 4H entry parameters
EMA_ENTRY  = 21
RSI_PERIOD = 14
RSI_LOW    = 35
RSI_HIGH   = 65
ATR_PERIOD = 14

# Trade sizing
SL_ATR_MULT    = 1.5   # stop = entry +/- SL_ATR_MULT * ATR
TP_RR          = 2.5   # target = risk * TP_RR
PROFIT_PER_POINT = 1.0

BACKTEST_START = None
BACKTEST_END   = None


# ============================================================================
# DATA LOADERS
# ============================================================================
def load_daily(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip('<>') for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['DATE'])
    df = df.set_index('Datetime').sort_index()
    df = df.rename(columns={
        'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
        'CLOSE': 'Close', 'TICKVOL': 'TickVol'
    })
    return df[['Open', 'High', 'Low', 'Close', 'TickVol']]


def load_4h(path):
    df = pd.read_csv(path, sep='\t')
    df.columns = [c.strip('<>') for c in df.columns]
    df['Datetime'] = pd.to_datetime(df['DATE'] + ' ' + df['TIME'])
    df = df.set_index('Datetime').sort_index()
    df = df.rename(columns={
        'OPEN': 'Open', 'HIGH': 'High', 'LOW': 'Low',
        'CLOSE': 'Close', 'TICKVOL': 'TickVol'
    })
    return df[['Open', 'High', 'Low', 'Close', 'TickVol']]


# ============================================================================
# INDICATORS
# ============================================================================
def compute_adx(high, low, close, period=14):
    prev_high  = high.shift(1)
    prev_low   = low.shift(1)
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    dm_plus  = high - prev_high
    dm_minus = prev_low - low
    dm_plus  = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0.0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0.0)
    atr      = tr.ewm(span=period, adjust=False).mean()
    di_plus  = 100 * dm_plus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_minus = 100 * dm_minus.ewm(span=period, adjust=False).mean() / atr.replace(0, np.nan)
    di_sum   = (di_plus + di_minus).replace(0, np.nan)
    dx       = 100 * (di_plus - di_minus).abs() / di_sum
    return dx.ewm(span=period, adjust=False).mean()


def compute_rsi(close, period=14):
    delta    = close.diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.ewm(span=period, adjust=False).mean()
    avg_loss = loss.ewm(span=period, adjust=False).mean()
    rs       = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(high, low, close, period=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================
print('=' * 70)
print('TREND FOLLOWING BACKTEST — XAUUSD Daily + 4H')
print(f'Trend: {EMA_FAST}/{EMA_SLOW} EMA + ADX>{ADX_THRESHOLD}')
print(f'Entry: {EMA_ENTRY} EMA pullback  |  SL: {SL_ATR_MULT}x ATR  |  TP: {TP_RR}:1 R:R')
print('=' * 70)

df_d  = load_daily(DATA_DAILY)
df_4h = load_4h(DATA_4H)
print(f'\nDaily: {len(df_d):,} bars  ({df_d.index[0].date()} to {df_d.index[-1].date()})')
print(f'4H:    {len(df_4h):,} bars  ({df_4h.index[0]} to {df_4h.index[-1]})')

# -- Daily indicators ---------------------------------------------------------
df_d['ema_fast'] = df_d['Close'].ewm(span=EMA_FAST,  adjust=False).mean()
df_d['ema_slow'] = df_d['Close'].ewm(span=EMA_SLOW,  adjust=False).mean()
df_d['adx']      = compute_adx(df_d['High'], df_d['Low'], df_d['Close'], ADX_PERIOD)

df_d['trend'] = 'NONE'
df_d.loc[
    (df_d['Close']    > df_d['ema_slow']) &
    (df_d['ema_fast'] > df_d['ema_slow']) &
    (df_d['adx']      >= ADX_THRESHOLD),
    'trend'
] = 'UP'
df_d.loc[
    (df_d['Close']    < df_d['ema_slow']) &
    (df_d['ema_fast'] < df_d['ema_slow']) &
    (df_d['adx']      >= ADX_THRESHOLD),
    'trend'
] = 'DOWN'

# Shift(1): today's 4H bars use yesterday's completed daily close
daily_trend        = df_d['trend'].copy()
daily_trend.index  = daily_trend.index.date
daily_trend_lagged = daily_trend.shift(1)

trend_dist = df_d['trend'].value_counts()
print(f'\nDaily trend distribution:')
for t in ['UP', 'DOWN', 'NONE']:
    n = trend_dist.get(t, 0)
    print(f'  {t:<5} {n:>4} days  ({n/len(df_d)*100:.1f}%)')

# -- 4H indicators ------------------------------------------------------------
df_4h['ema21'] = df_4h['Close'].ewm(span=EMA_ENTRY, adjust=False).mean()
df_4h['rsi']   = compute_rsi(df_4h['Close'], RSI_PERIOD)
df_4h['atr']   = compute_atr(df_4h['High'], df_4h['Low'], df_4h['Close'], ATR_PERIOD)

# -- Align daily trend to 4H bars (use previous day's completed trend) --------
df_4h['date']  = df_4h.index.date
df_4h['trend'] = df_4h['date'].map(daily_trend_lagged)
df_4h['trend'] = df_4h['trend'].ffill()   # carry through weekends/gaps
df_4h = df_4h.drop(columns=['date'])
df_4h = df_4h.dropna(subset=['ema21', 'rsi', 'atr', 'trend'])

if BACKTEST_START:
    df_4h = df_4h[df_4h.index >= BACKTEST_START]
if BACKTEST_END:
    df_4h = df_4h[df_4h.index < BACKTEST_END]

print(f'\nBacktest window: {df_4h.index[0]} to {df_4h.index[-1]}  ({len(df_4h):,} bars)')


# ============================================================================
# SIMULATION
# ============================================================================
closed_trades = []
open_trade    = None
equity        = 0.0
equity_curve  = []

prev_close = None
prev_ema21 = None

for dt, bar in df_4h.iterrows():
    high  = bar['High']
    low   = bar['Low']
    close = bar['Close']
    ema21 = bar['ema21']
    rsi   = bar['rsi']
    atr   = bar['atr']
    trend = bar['trend']

    # ── Manage open trade ────────────────────────────────────────────────────
    if open_trade is not None:
        t = open_trade

        # Daily trend flipped — exit at this bar's close
        if trend != t['trend']:
            pnl = ((close - t['entry']) if t['side'] == 'LONG'
                   else (t['entry'] - close)) * PROFIT_PER_POINT
            closed_trades.append({**t,
                'close_time': dt, 'close_price': close,
                'pnl': round(pnl, 3), 'outcome': 'TREND_EXIT'})
            equity    += pnl
            open_trade = None

        elif t['side'] == 'LONG':
            if low <= t['sl']:
                pnl = (t['sl'] - t['entry']) * PROFIT_PER_POINT
                closed_trades.append({**t,
                    'close_time': dt, 'close_price': t['sl'],
                    'pnl': round(pnl, 3), 'outcome': 'SL'})
                equity    += pnl
                open_trade = None
            elif high >= t['tp']:
                pnl = (t['tp'] - t['entry']) * PROFIT_PER_POINT
                closed_trades.append({**t,
                    'close_time': dt, 'close_price': t['tp'],
                    'pnl': round(pnl, 3), 'outcome': 'TP'})
                equity    += pnl
                open_trade = None

        elif t['side'] == 'SHORT':
            if high >= t['sl']:
                pnl = (t['entry'] - t['sl']) * PROFIT_PER_POINT
                closed_trades.append({**t,
                    'close_time': dt, 'close_price': t['sl'],
                    'pnl': round(pnl, 3), 'outcome': 'SL'})
                equity    += pnl
                open_trade = None
            elif low <= t['tp']:
                pnl = (t['entry'] - t['tp']) * PROFIT_PER_POINT
                closed_trades.append({**t,
                    'close_time': dt, 'close_price': t['tp'],
                    'pnl': round(pnl, 3), 'outcome': 'TP'})
                equity    += pnl
                open_trade = None

    # ── Check for new entry (no open trade, need prev bar data) ──────────────
    if open_trade is None and prev_close is not None:

        if trend == 'UP':
            # Pullback: prev bar below 21 EMA, current bar back above it
            if (prev_close < prev_ema21 and close > ema21
                    and RSI_LOW <= rsi <= RSI_HIGH):
                entry = close
                sl    = round(low  - SL_ATR_MULT * atr, 3)
                risk  = entry - sl
                tp    = round(entry + TP_RR * risk, 3)
                open_trade = {
                    'side': 'LONG', 'trend': trend,
                    'open_time': dt, 'entry': entry,
                    'sl': sl, 'tp': tp, 'risk': round(risk, 3),
                }

        elif trend == 'DOWN':
            # Pullback: prev bar above 21 EMA, current bar back below it
            if (prev_close > prev_ema21 and close < ema21
                    and RSI_LOW <= rsi <= RSI_HIGH):
                entry = close
                sl    = round(high + SL_ATR_MULT * atr, 3)
                risk  = sl - entry
                tp    = round(entry - TP_RR * risk, 3)
                open_trade = {
                    'side': 'SHORT', 'trend': trend,
                    'open_time': dt, 'entry': entry,
                    'sl': sl, 'tp': tp, 'risk': round(risk, 3),
                }

    prev_close = close
    prev_ema21 = ema21
    equity_curve.append({'Datetime': dt, 'equity': equity})

# Force-close any trade still open at end of data
if open_trade is not None:
    last  = df_4h.iloc[-1]
    close = last['Close']
    pnl   = ((close - open_trade['entry']) if open_trade['side'] == 'LONG'
              else (open_trade['entry'] - close)) * PROFIT_PER_POINT
    closed_trades.append({**open_trade,
        'close_time': df_4h.index[-1], 'close_price': close,
        'pnl': round(pnl, 3), 'outcome': 'FORCE_CLOSE'})
    equity += pnl


# ============================================================================
# RESULTS
# ============================================================================
print(f'\n{"=" * 70}')
print('BACKTEST RESULTS')
print('=' * 70)

trades_df = pd.DataFrame(closed_trades)

if len(trades_df) == 0:
    print('No trades executed.')
else:
    total   = len(trades_df)
    tp_mask = trades_df['outcome'] == 'TP'
    sl_mask = trades_df['outcome'] == 'SL'
    te_mask = trades_df['outcome'] == 'TREND_EXIT'

    wins    = trades_df[trades_df['pnl'] > 0]
    losses  = trades_df[trades_df['pnl'] <= 0]
    pf      = wins['pnl'].sum() / abs(losses['pnl'].sum()) if len(losses) > 0 else float('inf')

    eq_s    = pd.Series([e['equity'] for e in equity_curve])
    max_dd  = (eq_s - eq_s.cummax()).min()

    print(f'\n--- OVERALL ---')
    print(f'  Total trades:      {total:>5}')
    print(f'  TP hits:           {tp_mask.sum():>5}  ({tp_mask.mean()*100:.1f}%)')
    print(f'  SL hits:           {sl_mask.sum():>5}  ({sl_mask.mean()*100:.1f}%)')
    print(f'  Trend exits:       {te_mask.sum():>5}  ({te_mask.mean()*100:.1f}%)')
    print(f'  Win rate:          {(trades_df["pnl"]>0).mean()*100:>5.1f}%')
    print(f'  Total P&L:         ${trades_df["pnl"].sum():>10.2f}')
    print(f'  Avg P&L/trade:     ${trades_df["pnl"].mean():>10.2f}')
    print(f'  Avg win:           ${wins["pnl"].mean():>10.2f}')
    print(f'  Avg loss:          ${losses["pnl"].mean():>10.2f}')
    print(f'  Profit factor:     {pf:>10.2f}')
    print(f'  Max drawdown:      ${max_dd:>10.2f}')
    print(f'  Avg risk/trade:    ${trades_df["risk"].mean():>10.2f}')

    print(f'\n--- BY DIRECTION ---')
    for side in ['LONG', 'SHORT']:
        sub = trades_df[trades_df['side'] == side]
        if len(sub) == 0:
            continue
        wr = (sub['pnl'] > 0).mean() * 100
        print(f'  {side:<5}  {len(sub):>3} trades  WR: {wr:>5.1f}%  P&L: ${sub["pnl"].sum():>8.2f}')

    print(f'\n--- OUTCOME BREAKDOWN ---')
    for outcome in ['TP', 'SL', 'TREND_EXIT', 'FORCE_CLOSE']:
        sub = trades_df[trades_df['outcome'] == outcome]
        if len(sub) == 0:
            continue
        print(f'  {outcome:<12}  {len(sub):>3} trades  Avg P&L: ${sub["pnl"].mean():>8.2f}  Total: ${sub["pnl"].sum():>9.2f}')

    trades_df['month'] = trades_df['open_time'].dt.to_period('M')
    monthly = trades_df.groupby('month').agg(
        trades=('pnl', 'count'),
        pnl=('pnl', 'sum'),
        wr=('pnl', lambda x: (x > 0).mean() * 100)
    )
    print(f'\n--- MONTHLY P&L ---')
    print(f'  {"Month":<10} {"Trades":>7} {"P&L":>10} {"WR%":>6}')
    print(f'  {"-"*38}')
    for month, row in monthly.iterrows():
        flag = '  <<' if row['pnl'] < -20 else ''
        print(f'  {str(month):<10} {int(row["trades"]):>7} ${row["pnl"]:>9.2f} {row["wr"]:>5.1f}%{flag}')
    print(f'  {"-"*38}')
    print(f'  {"TOTAL":<10} {len(trades_df):>7} ${trades_df["pnl"].sum():>9.2f}')

    print(f'\n--- SAMPLE TRADES ---')
    print(trades_df[['open_time','close_time','side','entry','sl','tp',
                      'close_price','pnl','outcome']].head(20).to_string(index=False))

# ── Save ──────────────────────────────────────────────────────────────────────
os.makedirs('data/processed', exist_ok=True)
if len(trades_df) > 0:
    trades_df.to_csv(OUT_PATH, index=False)
    print(f'\nSaved trades: {OUT_PATH}')

pd.DataFrame(equity_curve).to_csv('data/processed/trend_equity_curve.csv', index=False)
print(f'Saved equity curve: data/processed/trend_equity_curve.csv')

print('\n' + '=' * 70)
print('TREND BACKTEST COMPLETE')
print('=' * 70)
