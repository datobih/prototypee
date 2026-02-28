"""
visualize_ob_combo.py
=====================
Plots filtered Order Blocks (FVG + Inducement + High OB Vol) on the H3 price chart.

For each matching OB it draws:
  - The OB zone (shaded green/red rectangle)
  - Reaction outcome label (BOUNCE / BREAK)
  - First touch bar (dashed vertical)
  - Displacement candle region (lighter shade)

Requires:  matplotlib, pandas, mplfinance
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH    = os.path.join(PROJECT_ROOT, 'data', 'raw', 'XAUUSD30.csv')
OB_CSV       = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos', 'top1_Fib_38-62pct__FVG.csv')
OUT_DIR      = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_charts_fib_golden')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load price data ─────────────────────────────────────────────────────────
col_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread']
with open(DATA_PATH) as f:
    first = f.readline().strip()
has_header = first.startswith('<') or first.upper().startswith('DATE')

price_df = pd.read_csv(DATA_PATH, sep='\t', names=col_names,
                       skiprows=1 if has_header else 0,
                       dtype={'Date': str, 'Time': str})
price_df['Datetime'] = pd.to_datetime(price_df['Date'] + ' ' + price_df['Time'],
                                      format='%Y.%m.%d %H:%M:%S')
price_df = price_df.set_index('Datetime')[['Open','High','Low','Close']].apply(pd.to_numeric, errors='coerce')
price_df = price_df.dropna()

# Load the specific combo CSV
combo = pd.read_csv(OB_CSV, parse_dates=['ob_dt'])

print(f"Loaded trades from {os.path.basename(OB_CSV)}: {len(combo)}")
print(f"  BOUNCE: {(combo['reaction']=='BOUNCE').sum()}")
print(f"  BREAK : {(combo['reaction']=='BREAK').sum()}")

CONTEXT_BEFORE_TOUCH = 30   # bars of context shown before the touch bar
CONTEXT_AFTER_TOUCH  = 50   # bars shown after the touch bar (to see the reaction)

def plot_ob(idx, row, price_df, out_dir):
    ob_time   = row['ob_dt']
    ob_high   = row['ob_high']
    ob_low    = row['ob_low']
    atr       = row['atr']
    direction = row['type']   # 'bull' or 'bear'
    reaction  = row['reaction']
    bars_to   = int(row['bars_to_mitigate']) if not pd.isna(row['bars_to_mitigate']) else None
    trade_rr  = row['trade_rr']  if not pd.isna(row['trade_rr'])  else None

    # Locate OB bar in price data
    if ob_time not in price_df.index:
        candidates = price_df.index[price_df.index >= ob_time]
        if len(candidates) == 0:
            return
        ob_time = candidates[0]

    ob_pos = price_df.index.get_loc(ob_time)

    # Center the chart on the TOUCH bar (not the OB creation bar)
    if bars_to is not None:
        touch_pos = ob_pos + bars_to
    else:
        touch_pos = ob_pos  # fallback: show around OB if never touched

    start = max(0, touch_pos - CONTEXT_BEFORE_TOUCH)
    end   = min(len(price_df) - 1, touch_pos + CONTEXT_AFTER_TOUCH)
    chunk = price_df.iloc[start:end+1].copy()

    # Local coordinate of key bars
    ob_x    = ob_pos - start
    touch_x = touch_pos - start

    fig, ax = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # ── Draw candles ────────────────────────────────────────────────────────
    for j, (dt, c) in enumerate(chunk.iterrows()):
        x = j
        bull = c['Close'] >= c['Open']
        col  = '#26a69a' if bull else '#ef5350'
        ax.plot([x, x], [c['Low'], c['High']], color=col, linewidth=0.8, alpha=0.7)
        body_bot = min(c['Open'], c['Close'])
        body_top = max(c['Open'], c['Close'])
        rect = mpatches.FancyBboxPatch(
            (x - 0.3, body_bot), 0.6, max(body_top - body_bot, 0.01),
            boxstyle="square,pad=0", linewidth=0,
            facecolor=col, alpha=0.9
        )
        ax.add_patch(rect)

    # ── OB zone (horizontal band extending through chart) ───────────────────
    zone_color = '#26a69a' if direction == 'bull' else '#ef5350'

    # OB creation region — only show if it's within chart window
    if 0 <= ob_x < len(chunk):
        ax.axvline(ob_x, color=zone_color, linewidth=1, linestyle=':', alpha=0.5)
        ax.text(ob_x + 0.3, ob_high + atr * 0.1,
                f"OB @ {price_df.index[ob_pos].strftime('%b %d')}",
                color=zone_color, fontsize=7.5, alpha=0.8)

    # OB zone rect — from start to end of chart
    zone_rect = mpatches.Rectangle(
        (-1, ob_low), len(chunk) + 2, ob_high - ob_low,
        linewidth=1.5, edgecolor=zone_color, facecolor=zone_color,
        alpha=0.2, linestyle='--'
    )
    ax.add_patch(zone_rect)

    # Zone boundary lines
    ax.axhline(ob_high, color=zone_color, linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(ob_low,  color=zone_color, linewidth=0.8, linestyle='--', alpha=0.6)

    # ── Touch bar ────────────────────────────────────────────────────────────
    if 0 <= touch_x < len(chunk):
        ax.axvline(touch_x, color='yellow', linewidth=1.2, linestyle='--', alpha=0.8)
        ax.text(touch_x + 0.3, chunk['Low'].min() + atr * 0.3,
                'TOUCH', color='yellow', fontsize=8, fontweight='bold', alpha=0.9)

    # ── TP / SL reference lines ─────────────────────────────────────────────
    if direction == 'bull':
        entry = ob_high
        sl    = ob_low
        tp    = entry + (entry - sl) * max(trade_rr or 2, 1)
    else:
        entry = ob_low
        sl    = ob_high
        tp    = entry - (sl - entry) * max(trade_rr or 2, 1)

    ax.axhline(entry, color='white',  linewidth=0.7, linestyle='-',  alpha=0.4)
    ax.axhline(sl,    color='#ef5350', linewidth=0.7, linestyle=':',  alpha=0.5)
    ax.text(len(chunk) - 1, sl + atr * 0.05, 'SL', color='#ef5350', fontsize=7, ha='right')
    ax.axhline(tp,    color='#26a69a', linewidth=0.7, linestyle=':',  alpha=0.5)
    ax.text(len(chunk) - 1, tp + atr * 0.05, f'TP ({trade_rr:.1f}R)' if trade_rr else 'TP',
            color='#26a69a', fontsize=7, ha='right')

    # ── Reaction annotation ──────────────────────────────────────────────────
    react_color = '#26a69a' if reaction == 'BOUNCE' else '#ef5350'
    rr_str = f"  {trade_rr:.1f}R" if trade_rr is not None and trade_rr > 0 else ""
    ax.text(0.02, 0.97,
            f"{'BOUNCE' if reaction=='BOUNCE' else 'BREAK'}{rr_str}",
            transform=ax.transAxes, color=react_color,
            fontsize=13, fontweight='bold', va='top')

    # ── Filter badge ─────────────────────────────────────────────────────────
    ax.text(0.02, 0.90, "Top 1 Combo: Fib 38-62% + FVG",
            transform=ax.transAxes, color='#aaaaaa', fontsize=8, va='top')
    ax.text(0.02, 0.85,
            f"Fib Retrace: {row.get('fib_retracement_pct', 0):.1f}% | {row['session']} | "
            f"Touch in {bars_to} bars" if bars_to else "Not touched",
            transform=ax.transAxes, color='#888888', fontsize=7.5, va='top')

    # ── Styling ──────────────────────────────────────────────────────────────
    ax.set_xlim(-1, len(chunk))
    price_range = chunk['High'].max() - chunk['Low'].min()
    ymin = min(chunk['Low'].min(), sl) - price_range * 0.05
    ymax = max(chunk['High'].max(), tp) + price_range * 0.10
    ax.set_ylim(ymin, ymax)

    step = max(1, len(chunk) // 10)
    xticks = range(0, len(chunk), step)
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([chunk.index[t].strftime('%b %d %H:%M') for t in xticks],
                       rotation=30, ha='right', color='#888', fontsize=7)
    ax.tick_params(colors='#666', which='both')
    ax.yaxis.tick_right()
    ax.yaxis.set_tick_params(labelcolor='#888', labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor('#333')

    title = (f"XAUUSD H3  |  OB: {ob_time.strftime('%Y-%m-%d %H:%M')}  |  "
             f"{'Bull' if direction=='bull' else 'Bear'} OB  |  Touch: "
             f"{price_df.index[touch_pos].strftime('%Y-%m-%d %H:%M') if bars_to else 'N/A'}")
    ax.set_title(title, color='#cccccc', fontsize=9, pad=10)

    fname = os.path.join(out_dir,
                         f"ob_{idx:03d}_{direction}_{reaction}_{ob_time.strftime('%Y%m%d_%H%M')}.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {os.path.basename(fname)}")


# ── Generate all charts ──────────────────────────────────────────────────────
print(f"\nGenerating charts → {OUT_DIR}\n")
for i, (_, row) in enumerate(combo.iterrows()):
    plot_ob(i, row, price_df, OUT_DIR)

print(f"\nDone. {len(combo)} charts saved to: {OUT_DIR}")

# ── Summary bar chart ────────────────────────────────────────────────────────
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
fig2.patch.set_facecolor('#0d1117')

# Bounce vs Break pie
reactions = combo['reaction'].value_counts()
colors    = ['#26a69a', '#ef5350', '#f5a623']
axes[0].pie(reactions.values, labels=reactions.index, autopct='%1.1f%%',
            colors=colors[:len(reactions)], textprops={'color': 'white', 'fontsize': 11})
axes[0].set_facecolor('#0d1117')
axes[0].set_title('Reaction (Fib 38-62% + FVG)', color='white', fontsize=11)

# RR distribution for bounces
bounces = combo[combo['reaction'] == 'BOUNCE']['trade_rr'].dropna()
axes[1].set_facecolor('#161b22')
if len(bounces) > 0:
    axes[1].hist(bounces, bins=10, color='#26a69a', alpha=0.8, edgecolor='#0d1117')
    axes[1].axvline(bounces.median(), color='yellow', linewidth=1.5,
                    linestyle='--', label=f'Median RR: {bounces.median():.1f}x')
    axes[1].legend(facecolor='#222', labelcolor='white', fontsize=9)
axes[1].set_title('R:R Distribution (Bounces only)', color='white', fontsize=11)
axes[1].set_xlabel('Trade R:R', color='#888')
axes[1].set_ylabel('Count', color='#888')
axes[1].tick_params(colors='#666')
for spine in axes[1].spines.values():
    spine.set_edgecolor('#333')

plt.tight_layout()
summary_path = os.path.join(OUT_DIR, '_summary_fib_golden_fvg.png')
plt.savefig(summary_path, dpi=130, bbox_inches='tight', facecolor=fig2.get_facecolor())
plt.close(fig2)
print(f"Summary chart: {summary_path}")
