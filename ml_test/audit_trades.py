"""
audit_trades.py
Verifies every trade in HEDGE_strategy.csv against raw 1-minute OHLC data.

Automatically reads TARGET, STOP, and HORIZON from hedge_strategy_strict.py
so you only ever need to change parameters in one place.

Classifications:
  GENUINE_WIN     - TP was hit cleanly before any SL in the same bar
  FAKE_WIN        - TP & SL both hit within the SAME 1-min candle
                    Backtester logged as WIN (TP checked first in code).
                    Live broker likely triggered the tighter SL first.
  MISLABELED_WIN  - SL hit cleanly before TP, but backtester still said WIN
  CONFIRMED_LOSS  - Already logged as LOSS in backtest (both stopped)
  TIMEOUT         - Neither TP nor SL hit within HORIZON bars
"""

import importlib.util
import sys
import os
import pandas as pd
import numpy as np

# --------------------------------------------------------------------------
# Auto-import parameters from hedge_strategy_strict.py
# --------------------------------------------------------------------------
STRATEGY_PATH = os.path.join(os.path.dirname(__file__), 'hedge_strategy_strict.py')

spec = importlib.util.spec_from_file_location("hedge_strategy_strict", STRATEGY_PATH)
strategy_mod = importlib.util.module_from_spec(spec)

# We only need the global constants — the module-level code will try to run
# training etc, so we monkey-patch heavy imports to avoid executing the whole
# script. Simpler: just parse the file for the three constants directly.
TARGET   = None
STOP     = None
HORIZON  = None

with open(STRATEGY_PATH) as f:
    for line in f:
        line = line.strip()
        if line.startswith('TARGET') and '=' in line and not line.startswith('#'):
            try:
                TARGET = float(line.split('=')[1].split('#')[0].strip())
            except:
                pass
        elif line.startswith('STOP') and '=' in line and not line.startswith('#'):
            try:
                STOP = float(line.split('=')[1].split('#')[0].strip())
            except:
                pass
        elif line.startswith('HORIZON') and '=' in line and not line.startswith('#'):
            try:
                HORIZON = int(line.split('=')[1].split('#')[0].strip())
            except:
                pass

if any(v is None for v in [TARGET, STOP, HORIZON]):
    print("ERROR: Could not parse TARGET / STOP / HORIZON from hedge_strategy_strict.py")
    sys.exit(1)

print(f"Parameters loaded from hedge_strategy_strict.py:")
print(f"  TARGET  = ${TARGET}")
print(f"  STOP    = ${STOP}")
print(f"  HORIZON = {HORIZON} bars")
print(f"  Min candle span to trigger fake win: ${TARGET + STOP:.2f}")
print()

# --------------------------------------------------------------------------
# Load raw 1-minute price data
# --------------------------------------------------------------------------
print("Loading raw 1-minute price data...")
df = pd.read_csv(
    'data/raw/XAUUSD1.csv',
    sep='\t',
    names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
)
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M:%S')
df = df.dropna(subset=['High', 'Low'])

# O(1) lookup dict: {datetime -> {'Open': o, 'High': h, 'Low': l, 'Close': c}}
price_dict = df.set_index('Datetime')[['Open', 'High', 'Low', 'Close']].to_dict('index')
print(f"Loaded {len(price_dict):,} price bars.")

# --------------------------------------------------------------------------
# Load the simulated trade log
# --------------------------------------------------------------------------
print("Loading HEDGE_strategy.csv...")
trades = pd.read_csv('data/processed/HEDGE_strategy.csv')
trades['Datetime'] = pd.to_datetime(trades['Datetime'])
print(f"Loaded {len(trades):,} trades.\n")

# --------------------------------------------------------------------------
# Audit every trade
# --------------------------------------------------------------------------
records = []

for _, row in trades.iterrows():
    trade_time = row['Datetime']
    entry      = row['entry']
    side       = row['surviving_side']
    result     = row['result']

    tp_level = round(entry + TARGET, 2) if side == 'LONG' else round(entry - TARGET, 2)
    sl_level = round(entry - STOP,   2) if side == 'LONG' else round(entry + STOP,   2)

    base = {
        'trade_entry_time':   trade_time,
        'session':            row['session'],
        'rf_prob':            row['rf_prob'],
        'entry_price':        entry,
        'side':               side,
        'tp_level':           tp_level,
        'sl_level':           sl_level,
        'backtest_result':    result,
        'offending_bar_time': None,
        'bar_offset_mins':    None,
        'bar_high':           None,
        'bar_low':            None,
        'bar_range':          None,
        'audit_result':       None,
        'classification':     None,
        'live_pnl_impact':    None,
    }

    # Already a loss — no need to scan bars
    if result == 'LOSS':
        base.update(
            audit_result='LOSS',
            classification='CONFIRMED_LOSS',
            live_pnl_impact=round(-STOP * 2, 2),
        )
        records.append(base)
        continue

    classification  = 'TIMEOUT'
    audit_result    = 'UNKNOWN'
    live_pnl_impact = 0.0

    for i in range(1, HORIZON + 1):
        ft = trade_time + pd.Timedelta(minutes=i)
        if ft not in price_dict:
            continue

        h = price_dict[ft]['High']
        l = price_dict[ft]['Low']
        o = price_dict[ft]['Open']
        c = price_dict[ft]['Close']

        if side == 'LONG':
            hit_tp = h >= entry + TARGET
            hit_sl = l <= entry - STOP
        elif side == 'SHORT':
            hit_tp = l <= entry - TARGET
            hit_sl = h >= entry + STOP
        else:
            classification = 'SKIP'
            break

        if hit_tp and hit_sl:
            # Double-touch: apply Candle Structure Wick Analysis
            tp_first = False
            if side == 'LONG':
                if c >= o:  # Bullish candle
                    if (o - l) >= STOP:
                        tp_first = False # SL hit on bottom wick first
                    else:
                        tp_first = True  # Went up first
                else:
                    tp_first = False # Bearish, went down first
            else: # SHORT
                if c <= o:  # Bearish candle
                    if (h - o) >= STOP:
                        tp_first = False # SL hit on top wick first
                    else:
                        tp_first = True  # Went down first
                else:
                    tp_first = False # Bullish, went up first

            if tp_first:
                classification  = 'WICK_WIN'       # heuristic says TP fired first
                audit_result    = 'WIN'
                live_pnl_impact = round(TARGET - STOP, 2)
            else:
                classification  = 'WICK_LOSS'      # heuristic says SL fired first
                audit_result    = 'PROBABLE_LOSS'
                live_pnl_impact = round(-STOP * 2, 2)
            base.update(offending_bar_time=ft, bar_offset_mins=i,
                        bar_high=h, bar_low=l, bar_range=round(h - l, 2))
            break
        elif hit_tp:
            classification  = 'GENUINE_WIN'
            audit_result    = 'WIN'
            live_pnl_impact = round(TARGET - STOP, 2)
            base.update(offending_bar_time=ft, bar_offset_mins=i,
                        bar_high=h, bar_low=l, bar_range=round(h - l, 2))
            break
        elif hit_sl:
            classification  = 'MISLABELED_WIN'
            audit_result    = 'LOSS'
            live_pnl_impact = round(-STOP, 2)
            base.update(offending_bar_time=ft, bar_offset_mins=i,
                        bar_high=h, bar_low=l, bar_range=round(h - l, 2))
            break

    base.update(audit_result=audit_result, classification=classification,
                live_pnl_impact=live_pnl_impact)
    records.append(base)

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
audit_df = pd.DataFrame(records)

total          = len(audit_df)
bt_wins        = (audit_df['backtest_result'] == 'WIN').sum()
bt_losses      = (audit_df['backtest_result'] == 'LOSS').sum()
genuine_wins   = (audit_df['classification'] == 'GENUINE_WIN').sum()
wick_wins      = (audit_df['classification'] == 'WICK_WIN').sum()
wick_loss      = (audit_df['classification'] == 'WICK_LOSS').sum()
total_wins_real= genuine_wins + wick_wins
mislabeled     = (audit_df['classification'] == 'MISLABELED_WIN').sum()
confirmed_loss = (audit_df['classification'] == 'CONFIRMED_LOSS').sum()
timeouts       = (audit_df['classification'] == 'TIMEOUT').sum()

print('='*70)
print('TRADE AUDIT SUMMARY (Candle Wick Structure Heuristic)')
print('='*70)
print(f'\nTotal trades:              {total}')
print(f'Backtest WINs:             {bt_wins}  ({bt_wins/total*100:.1f}%)')
print(f'Backtest LOSSes:           {bt_losses}  ({bt_losses/total*100:.1f}%)')
print(f'\n--- BREAKDOWN OF BACKTEST WINs ({bt_wins}) ---')
print(f'  Genuine Wins             {genuine_wins:>6}  ({genuine_wins/bt_wins*100 if bt_wins else 0:.1f}%)  [TP hit cleanly, no SL in same bar]')
print(f'  Heuristic WICK WIN       {wick_wins:>6}  ({wick_wins/bt_wins*100 if bt_wins else 0:.1f}%)  [Double-touch: Wick analysis -> TP first]')
print(f'  Heuristic WICK LOSS      {wick_loss:>6}  ({wick_loss/bt_wins*100 if bt_wins else 0:.1f}%)  [Double-touch: Wick analysis -> SL first]')
print(f'  Mislabeled Wins          {mislabeled:>6}  ({mislabeled/bt_wins*100 if bt_wins else 0:.1f}%)  [SL hit first, TP check was first in code]')
print(f'  Timeouts                 {timeouts:>6}  ({timeouts/bt_wins*100 if bt_wins else 0:.1f}%)  [Neither TP nor SL hit in {HORIZON} bars]')

# ---- Realistic P&L ----
genuine_pnl    = genuine_wins  * (TARGET - STOP)
wick_w_pnl     = wick_wins * (TARGET - STOP)
wick_l_pnl     = wick_loss * (-STOP * 2)
loss_pnl       = confirmed_loss* (-STOP * 2)
mislabel_pnl   = mislabeled    * (-STOP)
total_real_pnl = genuine_pnl + wick_w_pnl + wick_l_pnl + loss_pnl + mislabel_pnl
bt_pnl         = bt_wins * (TARGET - STOP) + bt_losses * (-STOP * 2)
real_wr        = total_wins_real / (total_wins_real + wick_loss + confirmed_loss + mislabeled) * 100 if (total_wins_real + wick_loss + confirmed_loss + mislabeled) > 0 else 0

print(f'\n--- P&L COMPARISON ---')
print(f'  Genuine wins   {genuine_wins:>6}  x +${TARGET - STOP:.2f}  = +${genuine_pnl:.2f}')
print(f'  Wick WINs      {wick_wins:>6}  x +${TARGET - STOP:.2f}  = +${wick_w_pnl:.2f}')
print(f'  Wick LOSSs     {wick_loss:>6}  x -${STOP*2:.2f}  = ${wick_l_pnl:.2f}')
print(f'  Conf losses    {confirmed_loss:>6}  x -${STOP*2:.2f}  = ${loss_pnl:.2f}')
print(f'\n  Backtest PnL (inflated):     ${bt_pnl:.2f}')
print(f'  Realistic (Wick) PnL:        ${total_real_pnl:.2f}')
print(f'  PnL Overstatement:           ${bt_pnl - total_real_pnl:.2f}')
print(f'\n  Realistic Win Rate:          {real_wr:.1f}%')
print(f'  Profitable?                  {"YES" if total_real_pnl > 0 else "NO"}')

print(f'\n--- SESSION BREAKDOWN (Wick Win Rate) ---')
fake_df    = audit_df[audit_df['classification'].isin(['WICK_LOSS', 'WICK_WIN'])]
genuine_df = audit_df[audit_df['classification'].isin(['GENUINE_WIN', 'WICK_WIN'])]
for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
    s = audit_df[audit_df['session'] == sess]
    g  = s['classification'].isin(['GENUINE_WIN', 'WICK_WIN']).sum()
    bl = (s['classification'] == 'WICK_LOSS').sum()
    l  = (s['classification'] == 'CONFIRMED_LOSS').sum()
    pnl = g*(TARGET-STOP) + bl*(-STOP*2) + l*(-STOP*2)
    wr = g/(g+bl+l)*100 if (g+bl+l) > 0 else 0
    print(f'  {sess:<12} RealWins:{g:<5} WickLoss:{bl:<5} Loss:{l:<5} WR:{wr:>5.1f}%  PnL:${pnl:.2f}')

print(f'\n--- CANDLE RANGE ANALYSIS ---')
brown_df = audit_df[audit_df['classification'].isin(['WICK_WIN', 'WICK_LOSS'])]
genuine_only = audit_df[audit_df['classification'] == 'GENUINE_WIN']
print(f'  Min candle range to trigger double-touch: ${TARGET + STOP:.2f}')
print(f'  Avg candle range on double-touch bars:    ${brown_df["bar_range"].mean():.2f}')
print(f'  Avg candle range on GENUINE WINs:         ${genuine_only["bar_range"].mean():.2f}')
print(f'  Double-touch bars resolved as WIN:        {(audit_df["classification"]=="WICK_WIN").sum()}')
print(f'  Double-touch bars resolved as LOSS:       {(audit_df["classification"]=="WICK_LOSS").sum()}')

# ---- Save outputs ----
audit_df.to_csv('data/processed/HEDGE_audit.csv', index=False)
print(f'\nFull audit saved:          data/processed/HEDGE_audit.csv ({len(audit_df)} rows)')

fake_cols = [
    'trade_entry_time', 'session', 'rf_prob', 'entry_price', 'side',
    'tp_level', 'sl_level',
    'offending_bar_time', 'bar_offset_mins',
    'bar_high', 'bar_low', 'bar_range',
    'live_pnl_impact',
]
double_touch_df = audit_df[audit_df['classification'].isin(['WICK_WIN', 'WICK_LOSS'])]
double_touch_df[fake_cols].to_csv('data/processed/HEDGE_fake_wins.csv', index=False)
print(f'Double-touch trades saved: data/processed/HEDGE_fake_wins.csv ({len(double_touch_df)} rows)')
print('='*70)
