import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'XAUUSD30.csv')
# Default: visualize the top combo (Fib 38-62% + FVG)
COMBO_CSV = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos',
                         'top1_Fib_38-62pct__FVG.csv')
COMBO_LABEL = 'Fib 38-62% + FVG'
OUT_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_charts')
os.makedirs(OUT_DIR, exist_ok=True)

CONTEXT_BEFORE = 30   # bars before OB candle
CONTEXT_AFTER = 60    # bars after mitigation (or after impulse if no mitigation)
MAX_CHARTS = None     # None = render all

# =============================================================================
# LOAD DATA
# =============================================================================
print('Loading OHLC data...')
df = pd.read_csv(DATA_PATH, sep='\t', header=0,
                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close',
                        'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close']].copy()
print(f'  {len(df):,} bars loaded')

print(f'Loading combo trades: {COMBO_LABEL}...')
filtered = pd.read_csv(COMBO_CSV)
filtered['ob_dt'] = pd.to_datetime(filtered['ob_dt'])
filtered = filtered.reset_index(drop=True)
print(f'  {len(filtered)} trades loaded')

bounces = (filtered['reaction'] == 'BOUNCE').sum()
breaks = (filtered['reaction'] == 'BREAK').sum()
weak = (filtered['reaction'] == 'WEAK').sum()
print(f'  BOUNCE: {bounces}  |  BREAK: {breaks}  |  WEAK: {weak}')
print(f'  Bounce rate: {bounces/len(filtered)*100:.1f}%')

# =============================================================================
# PLOT INDIVIDUAL OB CHARTS
# =============================================================================
n_charts = min(MAX_CHARTS, len(filtered)) if MAX_CHARTS else len(filtered)
print(f'\nRendering {n_charts} individual OB charts...')

FIB_LEVELS = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
FIB_COLORS = {
    0.0:   '#2196F3',  # blue — swing extreme
    0.236: '#9C27B0',  # purple
    0.382: '#FF9800',  # orange — golden zone start
    0.5:   '#F44336',  # red — midpoint
    0.618: '#FF9800',  # orange — golden zone end
    0.786: '#9C27B0',  # purple
    1.0:   '#2196F3',  # blue — origin
}

for i in range(n_charts):
    row = filtered.iloc[i]
    ob_dt = row['ob_dt']
    ob_type = row['type']
    reaction = row['reaction']
    rr = row['trade_rr']
    disp = row['disp_atr']
    bars_to_touch = row.get('bars_to_mitigate', None)

    # Find bar positions
    if ob_dt not in df.index:
        continue
    ob_pos = df.index.get_loc(ob_dt)

    start = max(ob_pos - CONTEXT_BEFORE, 0)
    end = min(ob_pos + CONTEXT_AFTER + CONTEXT_BEFORE, len(df))
    chunk = df.iloc[start:end].copy()

    if len(chunk) < 10:
        continue

    # OB zone
    zone_hi = row['ob_high']
    zone_lo = row['ob_low']

    # Relative position of OB candle in the chunk
    ob_rel = ob_pos - start

    # ─── FIND SWING EXTREME & TOUCH POINT ───
    # Scan forward from OB to find the impulse move's peak, then the touch
    swing_peak_pos = None
    swing_peak_price = None
    touch_pos = None
    fib_origin = None

    if ob_type == 'bull':
        fib_origin = zone_lo  # 100% fib = bottom of OB
        # Find where price first re-enters OB zone from above
        # First, price must move UP past the zone, then come BACK
        peak_search_end = min(ob_rel + 80, len(chunk))
        after_ob = chunk.iloc[ob_rel + 1:peak_search_end]

        if len(after_ob) > 2:
            # Find touch: where Low drops back to zone_hi (entry level)
            for k_rel in range(len(after_ob)):
                k_abs = ob_rel + 1 + k_rel
                bar_low = after_ob.iloc[k_rel]['Low']
                bar_high = after_ob.iloc[k_rel]['High']
                # Must first go above zone, then come back
                if k_rel > 1 and bar_low <= zone_hi:
                    touch_pos = k_abs
                    break

            if touch_pos is not None:
                # Swing high = max High between OB and touch
                swing_slice = chunk.iloc[ob_rel + 1:touch_pos]
                if len(swing_slice) > 0:
                    best = swing_slice['High'].values.argmax()
                    swing_peak_pos = ob_rel + 1 + best
                    swing_peak_price = swing_slice['High'].iloc[best]
            else:
                # No touch found in window — use max high after OB
                swing_peak_pos = ob_rel + 1 + after_ob['High'].values.argmax()
                swing_peak_price = after_ob['High'].max()
    else:
        fib_origin = zone_hi  # 100% fib = top of OB
        peak_search_end = min(ob_rel + 80, len(chunk))
        after_ob = chunk.iloc[ob_rel + 1:peak_search_end]

        if len(after_ob) > 2:
            for k_rel in range(len(after_ob)):
                k_abs = ob_rel + 1 + k_rel
                bar_high = after_ob.iloc[k_rel]['High']
                if k_rel > 1 and bar_high >= zone_lo:
                    touch_pos = k_abs
                    break

            if touch_pos is not None:
                swing_slice = chunk.iloc[ob_rel + 1:touch_pos]
                if len(swing_slice) > 0:
                    best = swing_slice['Low'].values.argmin()
                    swing_peak_pos = ob_rel + 1 + best
                    swing_peak_price = swing_slice['Low'].iloc[best]
            else:
                swing_peak_pos = ob_rel + 1 + after_ob['Low'].values.argmin()
                swing_peak_price = after_ob['Low'].min()

    # ─── COMPUTE FIB LEVELS ───
    fib_prices = {}
    full_move = None
    if swing_peak_price is not None and fib_origin is not None:
        full_move = abs(swing_peak_price - fib_origin)

        if full_move > 0.01:
            for level in FIB_LEVELS:
                if ob_type == 'bull':
                    # 0% = swing high, 100% = OB low
                    fib_prices[level] = swing_peak_price - level * full_move
                else:
                    # 0% = swing low, 100% = OB high
                    fib_prices[level] = swing_peak_price + level * full_move

    # ─── BUILD CHART ───
    fib_pct = row.get('fib_retracement_pct', 0)
    fig, axes = mpf.plot(
        chunk,
        type='candle', style='charles',
        figsize=(18, 9),
        title=f'{"BULL" if ob_type=="bull" else "BEAR"} OB  |  {reaction}  |  '
              f'RR={rr:.1f}  |  Disp={disp:.1f}x ATR  |  Fib={fib_pct:.0f}%  |  '
              f'{ob_dt.strftime("%Y-%m-%d %H:%M")}',
        returnfig=True,
    )
    ax = axes[0]

    # ─── DRAW FIB LEVELS ───
    if fib_prices and swing_peak_pos is not None:
        fib_x_start = swing_peak_pos
        fib_x_end = len(chunk) - 1

        # Shade golden zone (38.2% - 61.8%)
        if 0.382 in fib_prices and 0.618 in fib_prices:
            golden_top = max(fib_prices[0.382], fib_prices[0.618])
            golden_bot = min(fib_prices[0.382], fib_prices[0.618])
            golden_rect = Rectangle(
                (fib_x_start, golden_bot), fib_x_end - fib_x_start,
                golden_top - golden_bot,
                linewidth=0, facecolor='#FFD700', alpha=0.12
            )
            ax.add_patch(golden_rect)

        # Draw each fib line
        for level, price in fib_prices.items():
            lw = 1.2 if level in [0.382, 0.5, 0.618] else 0.7
            ls = '-' if level in [0.0, 1.0] else '--'
            clr = FIB_COLORS.get(level, 'gray')

            ax.plot([fib_x_start, fib_x_end], [price, price],
                    color=clr, linewidth=lw, linestyle=ls, alpha=0.7)

            # Label on right side: "38.2% — $2856.50"
            label = f'{level*100:.1f}%  ${price:.2f}'
            ax.text(fib_x_end + 0.3, price, label,
                    fontsize=7, color=clr, va='center', fontweight='bold')

        # Mark swing peak with a diamond
        if ob_type == 'bull':
            ax.plot(swing_peak_pos, swing_peak_price, marker='D',
                    color='#2196F3', markersize=8, zorder=5)
            ax.text(swing_peak_pos, swing_peak_price, '  Swing High',
                    fontsize=7, color='#2196F3', va='bottom', fontweight='bold')
        else:
            ax.plot(swing_peak_pos, swing_peak_price, marker='D',
                    color='#2196F3', markersize=8, zorder=5)
            ax.text(swing_peak_pos, swing_peak_price, '  Swing Low',
                    fontsize=7, color='#2196F3', va='top', fontweight='bold')

    # ─── MARK TOUCH POINT ───
    if touch_pos is not None and touch_pos < len(chunk):
        touch_bar = chunk.iloc[touch_pos]
        if ob_type == 'bull':
            touch_y = touch_bar['Low']
        else:
            touch_y = touch_bar['High']
        ax.plot(touch_pos, touch_y, marker='v' if ob_type == 'bull' else '^',
                color='magenta', markersize=10, zorder=5)
        ax.text(touch_pos, touch_y,
                f'  Touch\n  (Fib {fib_pct:.0f}%)',
                fontsize=7, color='magenta', va='top' if ob_type == 'bull' else 'bottom',
                fontweight='bold')

    # ─── OB ZONE RECTANGLE ───
    color = 'green' if ob_type == 'bull' else 'red'
    zone_width = len(chunk) - ob_rel
    rect = Rectangle(
        (ob_rel - 0.4, zone_lo), zone_width, zone_hi - zone_lo,
        linewidth=1.0, edgecolor=color, facecolor=color, alpha=0.15
    )
    ax.add_patch(rect)

    # OB candle highlight
    ob_rect = Rectangle(
        (ob_rel - 0.4, zone_lo), 1, zone_hi - zone_lo,
        linewidth=2.0, edgecolor=color, facecolor=color, alpha=0.4
    )
    ax.add_patch(ob_rect)
    ax.text(ob_rel, zone_hi if ob_type == 'bull' else zone_lo, 'OB',
            fontsize=9, color=color, fontweight='bold',
            va='bottom' if ob_type == 'bull' else 'top', ha='center')

    # ─── ENTRY / SL / TP ───
    if ob_type == 'bull':
        entry_price = zone_hi
        sl_price = zone_lo
        risk = entry_price - sl_price
    else:
        entry_price = zone_lo
        sl_price = zone_hi
        risk = sl_price - entry_price

    ax.axhline(y=entry_price, color='green', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.axhline(y=sl_price, color='red', linestyle='--', alpha=0.4, linewidth=0.8)

    if rr > 0 and risk > 0:
        if ob_type == 'bull':
            tp_price = entry_price + rr * risk
        else:
            tp_price = entry_price - rr * risk
        ax.axhline(y=tp_price, color='blue', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.text(0, tp_price, f'TP ({rr:.1f}R)  ', fontsize=7,
                color='blue', va='center', ha='right')

    # ─── REACTION BADGE ───
    react_color = {'BOUNCE': 'green', 'BREAK': 'red', 'WEAK': 'orange'}[reaction]
    ax.text(0.98, 0.02, reaction, transform=ax.transAxes,
            fontsize=14, fontweight='bold', color=react_color,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=react_color, alpha=0.9))

    # ─── INFO BOX (top-left) ───
    info_lines = [
        f'Type: {"BULL" if ob_type == "bull" else "BEAR"}',
        f'Displacement: {disp:.1f}x ATR',
        f'Fib retrace: {fib_pct:.1f}%',
        f'Touch quality: {row.get("touch_close_quality", "N/A")}',
        f'R:R achieved: {rr:.1f}',
    ]
    info_text = '\n'.join(info_lines)
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            fontsize=8, va='top', ha='left', family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='gray', alpha=0.85))

    fname = f'ob_{i+1:03d}_{ob_type}_{reaction}_{ob_dt.strftime("%Y%m%d_%H%M")}.png'
    fpath = os.path.join(OUT_DIR, fname)
    plt.savefig(fpath, dpi=140, bbox_inches='tight')
    plt.close(fig)

print(f'Saved {n_charts} charts to {OUT_DIR}/')

# =============================================================================
# SUMMARY GRID — 4x4 sample of bounces vs breaks
# =============================================================================
print('\nRendering summary grid...')

bounce_rows = filtered[filtered['reaction'] == 'BOUNCE'].head(8)
break_rows = filtered[filtered['reaction'] == 'BREAK'].head(8)
grid_rows = pd.concat([bounce_rows, break_rows]).reset_index(drop=True)

n_grid = min(16, len(grid_rows))
cols = 4
rows = (n_grid + cols - 1) // cols

fig_grid, axes_grid = plt.subplots(rows, cols, figsize=(28, 5 * rows))
axes_flat = axes_grid.flatten() if n_grid > 1 else [axes_grid]

for idx in range(len(axes_flat)):
    ax = axes_flat[idx]
    if idx >= n_grid:
        ax.set_visible(False)
        continue

    row = grid_rows.iloc[idx]
    ob_dt = row['ob_dt']
    ob_type = row['type']
    reaction = row['reaction']
    rr = row['trade_rr']

    if ob_dt not in df.index:
        ax.set_visible(False)
        continue

    ob_pos = df.index.get_loc(ob_dt)
    start = max(ob_pos - 15, 0)
    end = min(ob_pos + 45, len(df))
    chunk = df.iloc[start:end].copy()
    ob_rel = ob_pos - start
    zone_hi = row['ob_high']
    zone_lo = row['ob_low']

    # Plot candles manually (mini chart)
    for j in range(len(chunk)):
        c = chunk.iloc[j]
        color = '#26a69a' if c['Close'] >= c['Open'] else '#ef5350'
        ax.plot([j, j], [c['Low'], c['High']], color=color, linewidth=0.5)
        ax.plot([j, j], [min(c['Open'], c['Close']), max(c['Open'], c['Close'])],
                color=color, linewidth=2.5)

    # OB zone
    ob_color = 'green' if ob_type == 'bull' else 'red'
    rect = Rectangle(
        (ob_rel - 0.4, zone_lo), len(chunk) - ob_rel, zone_hi - zone_lo,
        linewidth=0.8, edgecolor=ob_color, facecolor=ob_color, alpha=0.15
    )
    ax.add_patch(rect)

    react_color = {'BOUNCE': 'green', 'BREAK': 'red', 'WEAK': 'orange'}[reaction]
    ax.set_title(f'{ob_type.upper()} | {reaction} | RR={rr:.1f}\n{ob_dt.strftime("%Y-%m-%d %H:%M")}',
                 fontsize=9, color=react_color, fontweight='bold')
    ax.set_xticks([])
    ax.tick_params(labelsize=7)

fig_grid.suptitle(f'{COMBO_LABEL} OBs — BOUNCE (top) vs BREAK (bottom)',
                   fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
grid_path = os.path.join(OUT_DIR, '_summary_grid.png')
fig_grid.savefig(grid_path, dpi=150, bbox_inches='tight')
plt.close(fig_grid)
print(f'Saved summary grid: {grid_path}')

# =============================================================================
# EQUITY CURVE — cumulative R:R if trading all 161 setups
# =============================================================================
print('\nRendering equity curve...')

pnl = []
for _, row in filtered.iterrows():
    if row['reaction'] == 'BOUNCE':
        pnl.append(row['trade_rr'])
    elif row['reaction'] == 'BREAK':
        pnl.append(-1.0)  # SL hit = -1R
    else:
        pnl.append(-0.2)  # WEAK = small loss / scratch

cumulative = np.cumsum(pnl)

fig_eq, ax_eq = plt.subplots(figsize=(14, 6))
ax_eq.plot(cumulative, color='dodgerblue', linewidth=1.5)
ax_eq.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax_eq.fill_between(range(len(cumulative)), cumulative, 0,
                    where=np.array(cumulative) >= 0, color='green', alpha=0.1)
ax_eq.fill_between(range(len(cumulative)), cumulative, 0,
                    where=np.array(cumulative) < 0, color='red', alpha=0.1)
ax_eq.set_xlabel('Trade #')
ax_eq.set_ylabel('Cumulative R')
ax_eq.set_title(f'{COMBO_LABEL} OBs — Equity Curve ({len(filtered)} trades)\n'
                f'Bounce rate: {bounces/len(filtered)*100:.1f}%  |  '
                f'Final: {cumulative[-1]:.1f}R  |  '
                f'Max DD: {min(cumulative - np.maximum.accumulate(cumulative)):.1f}R')
ax_eq.grid(True, alpha=0.3)

eq_path = os.path.join(OUT_DIR, '_equity_curve.png')
fig_eq.savefig(eq_path, dpi=150, bbox_inches='tight')
plt.close(fig_eq)
print(f'Saved equity curve: {eq_path}')

print(f'\nAll outputs in: {OUT_DIR}/')
print('Done.')
