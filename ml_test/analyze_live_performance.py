"""
Analyze actual live trading performance from MT5 history
Accounts for spread, slippage, and commissions
Auto-detects hedge and directional strategies by magic number.
"""

import MetaTrader5 as mt5
from datetime import datetime, timedelta, timezone
import pandas as pd
import numpy as np

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialize failed")
    exit()

# Get deals history for last 30 days
from_date = datetime.now(timezone.utc) - timedelta(days=30)
to_date = datetime.now(timezone.utc)

# Known strategy magic numbers
STRATEGIES = {
    'HEDGE_XAUUSD':       {'long': 123456, 'short': 654321, 'mode': 'hedge'},
    'HEDGE_USDJPY':       {'long': 223456, 'short': 754321, 'mode': 'hedge'},
    'DIRECTIONAL_XAUUSD': {'long': 235001, 'short': 235002, 'mode': 'directional'},
}

deals = mt5.history_deals_get(from_date, to_date)
if deals is None or len(deals) == 0:
    print("No deals found")
    mt5.shutdown()
    exit()

df = pd.DataFrame(list(deals), columns=deals[0]._asdict().keys())

# Auto-detect active strategies
all_magics = set(df['magic'].unique())
active = {n: s for n, s in STRATEGIES.items()
          if s['long'] in all_magics or s['short'] in all_magics}

if not active:
    print("No trades found for any known strategy.")
    print(f"Magic numbers in history: {sorted(all_magics)}")
    mt5.shutdown()
    exit()

print(f"Found {len(active)} active strategy(ies): {', '.join(active.keys())}")

# ═══════════════════════════════════════════════════════════════════════
for strat_name, strat_info in active.items():
    MAGIC_LONG  = strat_info['long']
    MAGIC_SHORT = strat_info['short']
    MODE        = strat_info['mode']

    sdeals = df[(df['magic'] == MAGIC_LONG) | (df['magic'] == MAGIC_SHORT)].copy()
    if len(sdeals) == 0:
        continue

    print("\n" + "=" * 80)
    print(f"  LIVE {strat_name} PERFORMANCE  ({MODE} mode, magic {MAGIC_LONG}/{MAGIC_SHORT})")
    print("=" * 80)

    sdeals['time'] = pd.to_datetime(sdeals['time'], unit='s')
    print(f"\n  Total deals: {len(sdeals)}")
    print(f"  Date range: {sdeals['time'].min()} to {sdeals['time'].max()}")

    entries = sdeals[sdeals['entry'] == 0]
    exits   = sdeals[sdeals['entry'] == 1]
    print(f"  Entries: {len(entries)}  |  Exits: {len(exits)}")

    # ── SPREAD ANALYSIS (hedge mode only) ──
    if MODE == 'hedge':
        print(f"\n--- SPREAD ANALYSIS ---")
        long_entries  = entries[entries['magic'] == MAGIC_LONG].copy()
        short_entries = entries[entries['magic'] == MAGIC_SHORT].copy()
        spreads = []
        for _, lr in long_entries.iterrows():
            td = abs((short_entries['time'] - lr['time']).dt.total_seconds())
            m = short_entries[td <= 2]
            if len(m) > 0:
                spreads.append({'time': lr['time'],
                                'long_price': lr['price'],
                                'short_price': m.iloc[0]['price'],
                                'spread': lr['price'] - m.iloc[0]['price']})
        if spreads:
            sdf = pd.DataFrame(spreads)
            print(f"  Matched pairs: {len(sdf)}")
            print(f"  Avg spread: ${sdf['spread'].mean():.2f}  |  Med: ${sdf['spread'].median():.2f}")
            print(f"  Min: ${sdf['spread'].min():.2f}  |  Max: ${sdf['spread'].max():.2f}")

    # ── P&L BREAKDOWN ──
    total_profit     = sdeals['profit'].sum()
    total_commission = sdeals['commission'].sum()
    total_swap       = sdeals['swap'].sum()
    net_pnl = total_profit + total_commission + total_swap

    print(f"\n--- P&L BREAKDOWN ---")
    print(f"  Gross Profit: ${total_profit:.2f}")
    print(f"  Commission:   ${total_commission:.2f}")
    print(f"  Swap:         ${total_swap:.2f}")
    print(f"  NET P&L:      ${net_pnl:.2f}")

    # ── BY SIDE ──
    long_deals  = sdeals[sdeals['magic'] == MAGIC_LONG]
    short_deals = sdeals[sdeals['magic'] == MAGIC_SHORT]
    long_pnl  = long_deals['profit'].sum() + long_deals['commission'].sum()
    short_pnl = short_deals['profit'].sum() + short_deals['commission'].sum()
    print(f"\n--- BY SIDE ---")
    print(f"  LONG  P&L: ${long_pnl:.2f}  ({len(long_deals[long_deals['entry']==1])} exits)")
    print(f"  SHORT P&L: ${short_pnl:.2f}  ({len(short_deals[short_deals['entry']==1])} exits)")

    # ── WIN / LOSS ──
    winning = exits[exits['profit'] > 0]
    losing  = exits[exits['profit'] < 0]
    be      = exits[exits['profit'] == 0]
    win_rate = len(winning) / len(exits) * 100 if len(exits) > 0 else 0

    print(f"\n--- WIN / LOSS ---")
    print(f"  Wins: {len(winning)}  |  Losses: {len(losing)}  |  Breakeven: {len(be)}")
    print(f"  Win Rate: {win_rate:.1f}%")

    if len(winning) > 0:
        avg_win = winning['profit'].mean()
        print(f"  Avg Win:  ${avg_win:.2f}")
    if len(losing) > 0:
        avg_loss = losing['profit'].mean()
        print(f"  Avg Loss: ${avg_loss:.2f}")

    # Profit factor
    gross_w = winning['profit'].sum() if len(winning) > 0 else 0
    gross_l = abs(losing['profit'].sum()) if len(losing) > 0 else 1e-10
    pf = gross_w / gross_l
    print(f"  Profit Factor: {pf:.2f}")

    # EV per trade
    if len(exits) > 0:
        ev = net_pnl / len(exits)
        print(f"  EV per trade: ${ev:.2f}")

    # ── HEDGE PAIR WIN RATE (hedge only) ──
    if MODE == 'hedge':
        print(f"\n--- HEDGE PAIR WIN RATE ---")
        long_exits  = exits[exits['magic'] == MAGIC_LONG].copy()
        short_exits = exits[exits['magic'] == MAGIC_SHORT].copy()
        pairs = []
        used = set()
        for _, lr in long_exits.iterrows():
            for idx, sr in short_exits.iterrows():
                if idx in used:
                    continue
                if abs((sr['time'] - lr['time']).total_seconds()) <= 60:
                    pp = lr['profit'] + sr['profit']
                    pairs.append({'time': lr['time'],
                                  'long_profit': lr['profit'],
                                  'short_profit': sr['profit'],
                                  'pair_profit': pp,
                                  'result': 'WIN' if pp > 0 else ('LOSS' if pp < 0 else 'BE')})
                    used.add(idx)
                    break
        if pairs:
            pdf = pd.DataFrame(pairs)
            pw = len(pdf[pdf['result'] == 'WIN'])
            pl = len(pdf[pdf['result'] == 'LOSS'])
            print(f"  Pairs matched: {len(pdf)}  |  Wins: {pw}  |  Losses: {pl}")
            print(f"  HEDGE WIN RATE: {pw/len(pdf)*100:.1f}%")
            print(f"  Avg pair profit: ${pdf['pair_profit'].mean():.2f}")
            print(f"\n  Last 10 pairs:")
            print(pdf.tail(10).to_string(index=False))

    # ── SAMPLE TRADES ──
    print(f"\n--- LAST 20 EXITS ---")
    cols = ['time', 'type', 'price', 'volume', 'profit', 'commission', 'magic']
    rex = exits.tail(20)[cols].copy()
    rex['side'] = rex['magic'].apply(lambda x: 'LONG' if x == MAGIC_LONG else 'SHORT')
    print(rex.to_string(index=False))

    # ── DAILY BREAKDOWN ──
    sdeals['date'] = sdeals['time'].dt.date
    daily = sdeals.groupby('date').agg(
        profit=('profit', 'sum'),
        commission=('commission', 'sum'),
        swap=('swap', 'sum'),
    ).reset_index()
    daily['net'] = daily['profit'] + daily['commission'] + daily['swap']

    print(f"\n--- DETAILED DAILY ANALYSIS ---")
    for date in sorted(sdeals['date'].unique()):
        dd = sdeals[sdeals['date'] == date]
        de = dd[dd['entry'] == 1]
        dw = de[de['profit'] > 0]
        dl = de[de['profit'] < 0]
        dp = dd['profit'].sum()
        dwr = len(dw) / len(de) * 100 if len(de) > 0 else 0
        dev = dp / len(de) if len(de) > 0 else 0
        print(f"  {date} | Exits: {len(de):>3} | W: {len(dw):>3} | L: {len(dl):>3} "
              f"| WR: {dwr:>5.1f}% | P&L: ${dp:>8.2f} | EV: ${dev:>5.2f}")

    # ── BEST / WORST DAYS ──
    if len(daily) >= 2:
        print(f"\n--- WORST 3 DAYS ---")
        for _, row in daily.nsmallest(3, 'net').iterrows():
            print(f"  {row['date']}: ${row['net']:>8.2f}")
        print(f"\n--- BEST 3 DAYS ---")
        for _, row in daily.nlargest(3, 'net').iterrows():
            print(f"  {row['date']}: ${row['net']:>8.2f}")

    # ── SUMMARY ──
    print(f"\n{'='*80}")
    print(f"  SUMMARY — {strat_name}")
    print(f"{'='*80}")
    print(f"  Total Trades: {len(exits)}")
    print(f"  Net P&L:      ${net_pnl:.2f}")
    print(f"  Win Rate:     {win_rate:.1f}%")
    print(f"  Profit Factor:{pf:.2f}")
    print(f"  Commission:   ${total_commission:.2f}")

    if len(exits) > 0 and len(losing) > 0 and len(winning) > 0:
        avg_loss_size = abs(losing['profit'].mean())
        avg_win_size  = winning['profit'].mean()
        if avg_win_size > 0 and avg_loss_size > 0:
            be_wr = avg_loss_size / (avg_win_size + avg_loss_size) * 100
            print(f"\n  Breakeven WR needed: {be_wr:.1f}%")
            print(f"  Your actual WR:      {win_rate:.1f}%")
            if win_rate > be_wr:
                print("  >> PROFITABLE with current spread/slippage")
            else:
                print("  >> NOT profitable — need higher WR or better fills")

mt5.shutdown()
