"""Quick analysis of confluence_wr_stability to find most stable filter."""
import pandas as pd
import numpy as np

res = pd.read_csv('ml_test/output/confluence_results.csv')

print('=' * 80)
print('  STABILITY ANALYSIS — BEST FILTER RECOMMENDATION')
print('=' * 80)

improved = res[res['wr_delta'] >= 0].copy()
print(f'\n  Filters with WR >= baseline: {len(improved)}\n')


def parse_ratio(s):
    parts = str(s).split('/')
    if len(parts) == 2 and int(parts[1]) > 0:
        return int(parts[0]) / int(parts[1]) * 100
    return 0


scores = []
for _, r in improved.iterrows():
    mo_pct = parse_ratio(r['prof_months'])
    wk_pct = parse_ratio(r['prof_weeks'])
    dy_pct = parse_ratio(r['prof_days'])

    # Composite: high WR, low std, high profitable ratio, decent trade count
    score = (
        r['wr_mean_m'] * 0.3 - r['wr_std_m'] * 0.5 + mo_pct * 0.15
        + r['wr_mean_w'] * 0.2 - r['wr_std_w'] * 0.3 + wk_pct * 0.10
        + dy_pct * 0.05
        + min(r['trades'] / 1000, 3) * 2
    )
    scores.append(score)

improved = improved.copy()
improved['score'] = scores
improved = improved.sort_values('score', ascending=False)

hdr = (f"  {'Filter':<30s}  {'WR%':>5s}  {'Trd':>5s}  "
       f"{'MoWR':>5s} {'±':>4s}  {'WkWR':>5s} {'±':>4s}  "
       f"{'PrfM':>5s}  {'PrfW':>6s}  {'PrfD':>6s}  {'Score':>6s}")
print(hdr)
print(f"  {'-'*105}")
for _, r in improved.iterrows():
    print(f"  {r['name']:<30s}  {r['wr']:>5.1f}  {int(r['trades']):>5d}  "
          f"{r['wr_mean_m']:>5.1f} {r['wr_std_m']:>4.1f}  "
          f"{r['wr_mean_w']:>5.1f} {r['wr_std_w']:>4.1f}  "
          f"{r['prof_months']:>5s}  {r['prof_weeks']:>6s}  "
          f"{r['prof_days']:>6s}  {r['score']:>6.1f}")

print()
print('=' * 80)
print('  TOP 3 RECOMMENDATIONS')
print('=' * 80)
for rank, (_, r) in enumerate(improved.head(3).iterrows(), 1):
    print(f"\n  #{rank}: {r['name']}")
    print(f"      Desc: {r['desc']}")
    print(f"      Net WR: {r['wr']}%  |  Trades: {int(r['trades']):,}  "
          f"|  P&L: ${r['pnl']:+,.2f}  |  t: {r['t_stat']}")
    print(f"      Monthly: avg {r['wr_mean_m']}% +/- {r['wr_std_m']}%  "
          f"|  Profitable: {r['prof_months']}")
    print(f"      Weekly:  avg {r['wr_mean_w']}% +/- {r['wr_std_w']}%  "
          f"|  Profitable: {r['prof_weeks']}")
    print(f"      Daily:   avg {r['wr_mean_d']}% +/- {r['wr_std_d']}%  "
          f"|  Profitable: {r['prof_days']}")
    print(f"      Stability Score: {r['score']:.1f}")

print()
print('=' * 80)
