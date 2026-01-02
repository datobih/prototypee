import pandas as pd
print('Loading data...')
df = pd.read_csv('XAU_1m_data.csv', sep=';')
print(f'Original rows: {len(df):,}')

# Filter to 2025 only
df_2025 = df[df['Date'] >= '2025.01.01']
print(f'2025 only: {len(df_2025):,} rows')
print('First:', df_2025['Date'].iloc[0])
print('Last:', df_2025['Date'].iloc[-1])

# Convert date format from 2025.01.01 to 2025-01-01
df_2025 = df_2025.copy()
df_2025['Date'] = df_2025['Date'].str.replace('.', '-', n=2, regex=False)

# Save to XAUUSD1.csv in correct format (tab-separated, no header)
df_2025.to_csv('data/raw/XAUUSD1.csv', sep='\t', header=False, index=False)
print('Saved to data/raw/XAUUSD1.csv')
