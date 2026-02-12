import pandas as pd

COMMISSION = 0.36
TARGET = 3.0
STOP = 1.5
WIN_NET = (TARGET - STOP) - COMMISSION
LOSS_NET = (-2 * STOP) - COMMISSION

df = pd.read_csv('data/processed/HEDGE_strategy.csv')
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['pnl'] = df['result'].apply(lambda x: WIN_NET if x == 'WIN' else LOSS_NET)
df['month'] = df['Datetime'].dt.to_period('M')
df['week'] = df['Datetime'].dt.to_period('W')

print('='*80)
print('MONTHLY P&L BREAKDOWN (after $0.36 commission)')
print('='*80)
monthly = df.groupby('month').agg(
    trades=('pnl','count'),
    wins=('result', lambda x: (x=='WIN').sum()),
    pnl=('pnl','sum')
).reset_index()
monthly['losses'] = monthly['trades'] - monthly['wins']
monthly['wr'] = (monthly['wins']/monthly['trades']*100).round(1)
monthly['pnl'] = monthly['pnl'].round(2)
monthly['per_trade'] = (monthly['pnl']/monthly['trades']).round(2)
monthly['profitable'] = monthly['pnl'].apply(lambda x: 'YES' if x > 0 else 'NO')

print(f'\n{"Month":<10} {"Trades":<8} {"Wins":<6} {"Losses":<8} {"WR%":<8} {"P&L $":<10} {"$/trade":<10} {"Profit?"}')
print('-'*70)
for _, r in monthly.iterrows():
    print(f'{str(r["month"]):<10} {r["trades"]:<8} {r["wins"]:<6} {r["losses"]:<8} {r["wr"]:<8} {r["pnl"]:>8} {r["per_trade"]:>8}   {r["profitable"]}')
print(f'\nProfitable months: {(monthly["pnl"]>0).sum()}/{len(monthly)} ({(monthly["pnl"]>0).sum()/len(monthly)*100:.0f}%)')
print(f'Total monthly P&L: ${monthly["pnl"].sum():.2f}')

print('\n' + '='*80)
print('WEEKLY P&L BREAKDOWN (after $0.36 commission)')
print('='*80)
weekly = df.groupby('week').agg(
    trades=('pnl','count'),
    wins=('result', lambda x: (x=='WIN').sum()),
    pnl=('pnl','sum')
).reset_index()
weekly['losses'] = weekly['trades'] - weekly['wins']
weekly['wr'] = (weekly['wins']/weekly['trades']*100).round(1)
weekly['pnl'] = weekly['pnl'].round(2)
weekly['per_trade'] = (weekly['pnl']/weekly['trades']).round(2)
weekly['profitable'] = weekly['pnl'].apply(lambda x: 'YES' if x > 0 else 'NO')

print(f'\n{"Week":<16} {"Trades":<8} {"Wins":<6} {"Losses":<8} {"WR%":<8} {"P&L $":<10} {"$/trade":<10} {"Profit?"}')
print('-'*75)
for _, r in weekly.iterrows():
    print(f'{str(r["week"]):<16} {r["trades"]:<8} {r["wins"]:<6} {r["losses"]:<8} {r["wr"]:<8} {r["pnl"]:>8} {r["per_trade"]:>8}   {r["profitable"]}')

print(f'\nProfitable weeks: {(weekly["pnl"]>0).sum()}/{len(weekly)} ({(weekly["pnl"]>0).sum()/len(weekly)*100:.0f}%)')
print(f'Losing weeks: {(weekly["pnl"]<0).sum()}/{len(weekly)}')
print(f'Worst week: ${weekly["pnl"].min():.2f}')
print(f'Best week: ${weekly["pnl"].max():.2f}')
print(f'Avg week: ${weekly["pnl"].mean():.2f}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)
print(f'Total trades: {len(df)}')
print(f'Overall WR: {(df["result"]=="WIN").sum()/len(df)*100:.1f}%')
print(f'Total P&L (after comm): ${df["pnl"].sum():.2f}')
print(f'Profitable months: {(monthly["pnl"]>0).sum()}/{len(monthly)}')
print(f'Profitable weeks: {(weekly["pnl"]>0).sum()}/{len(weekly)}')
print(f'Max drawdown week: ${weekly["pnl"].min():.2f}')
