import pandas as pd
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..')

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'raw', 'GBPUSDH1.csv')
TAIL_BARS = None          # None = use ALL data for maximum sample size
ATR_PERIOD = 14
IMPULSE_ATR_MULT = 1.5    # body > X * ATR = impulsive candle
OB_LOOKBACK = 20          # Increased to capture deeper OBs
DISPLACEMENT_BARS = 10    # bars to measure displacement after impulse
MITIGATION_HORIZON = 200  # max bars to wait for mitigation (avoids right-censoring)
MIN_SWING_BARS = 5        # minimum bars between impulse and touch for valid fib measurement
SWING_LEFT = 3            # FIX #7: bars to the left for swing confirmation
SWING_RIGHT = 2           # FIX #7: bars to the right for swing confirmation

# FIX #5: Fixed R:R targets for realistic profitability assessment
FIXED_RR_TARGETS = [1.0, 1.5, 2.0, 3.0]
REACTION_BARS = 30

# =============================================================================
# LOAD DATA
# =============================================================================
print('='*90)
print('ORDER BLOCK MITIGATION FILTER ANALYSIS (FIXED)')
print('='*90)

df = pd.read_csv(DATA_PATH, sep='\t', header=0,
                 names=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'TickVol', 'Vol', 'Spread'])
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.set_index('Datetime', inplace=True)
df = df[['Open', 'High', 'Low', 'Close', 'TickVol']].copy()

if TAIL_BARS:
    df = df.tail(TAIL_BARS).copy()

print(f'Loaded {len(df):,} bars  |  {df.index[0]} → {df.index[-1]}')

# =============================================================================
# PRECOMPUTE BAR PROPERTIES
# =============================================================================
df['Range'] = df['High'] - df['Low']
df['Body'] = (df['Close'] - df['Open']).abs()
df['Bullish'] = df['Close'] > df['Open']

tr = pd.DataFrame({
    'hl': df['Range'],
    'hc': (df['High'] - df['Close'].shift(1)).abs(),
    'lc': (df['Low'] - df['Close'].shift(1)).abs()
}).max(axis=1)
df['ATR'] = tr.rolling(ATR_PERIOD).mean()

# EMAs for trend context
df['EMA_50'] = df['Close'].ewm(span=50).mean()
df['EMA_200'] = df['Close'].ewm(span=200).mean()

# Volume moving average
df['VMA_50'] = df['TickVol'].rolling(50).mean()

df.dropna(subset=['ATR', 'EMA_200', 'VMA_50'], inplace=True)

# Rolling 200-bar range for premium/discount classification
ROLLING_RANGE_BARS = 200
df['roll_high'] = df['High'].rolling(ROLLING_RANGE_BARS, min_periods=1).max()
df['roll_low']  = df['Low'].rolling(ROLLING_RANGE_BARS, min_periods=1).min()
df['roll_mid']  = (df['roll_high'] + df['roll_low']) / 2

# =============================================================================
# FIX #7: CENTERED SWING DETECTION WITH RIGHT-SIDE CONFIRMATION
# =============================================================================
# Original: left-sided only (no right confirmation → phantom pivots)
# Fixed: requires the bar to be higher/lower than both left AND right neighbours

highs_arr = df['High'].values
lows_arr = df['Low'].values
n_total = len(df)

swing_high_flags = np.zeros(n_total, dtype=bool)
swing_low_flags = np.zeros(n_total, dtype=bool)

for i in range(SWING_LEFT, n_total - SWING_RIGHT):
    left_highs = highs_arr[i - SWING_LEFT:i]
    right_highs = highs_arr[i + 1:i + SWING_RIGHT + 1]
    if highs_arr[i] > left_highs.max() and highs_arr[i] > right_highs.max():
        swing_high_flags[i] = True

    left_lows = lows_arr[i - SWING_LEFT:i]
    right_lows = lows_arr[i + 1:i + SWING_RIGHT + 1]
    if lows_arr[i] < left_lows.min() and lows_arr[i] < right_lows.min():
        swing_low_flags[i] = True

df['is_swing_high'] = swing_high_flags
df['is_swing_low'] = swing_low_flags

df['Impulsive'] = df['Body'] > (IMPULSE_ATR_MULT * df['ATR'])

# =============================================================================
# FIX #6: PROPER MARKET STRUCTURE STATE MACHINE
# =============================================================================
# Original: updated last_high/last_low from EVERY swing pivot regardless of trend.
#   A minor pullback swing high in a bear trend would reset last_high to a low value,
#   making the next rally trivially "break structure."
# Fixed: In BULLISH, only swing lows update the structural low (for CHoCH detection),
#   and only new higher highs update structural high. Vice versa for BEARISH.

ms_trend = ['Neutral'] * n_total
structural_high = highs_arr[0]
structural_low = lows_arr[0]
current_state = 'Neutral'

for i in range(n_total):
    close_price = df['Close'].iloc[i]
    is_sh = swing_high_flags[i]
    is_sl = swing_low_flags[i]

    if current_state == 'Neutral':
        # Bootstrap: first directional break sets the trend
        if close_price > structural_high:
            current_state = 'Bullish'
        elif close_price < structural_low:
            current_state = 'Bearish'
        # In neutral, update both pivots from any swing
        if is_sh:
            structural_high = highs_arr[i]
        if is_sl:
            structural_low = lows_arr[i]

    elif current_state == 'Bullish':
        # CHoCH: price closes below the structural low → bearish
        if close_price < structural_low:
            current_state = 'Bearish'
            # Reset structural high to most recent swing high for new bearish context
            if is_sh:
                structural_high = highs_arr[i]
        else:
            # Update structural low from swing lows (pullback lows in bull trend)
            if is_sl:
                structural_low = lows_arr[i]
            # Update structural high only if it's a HIGHER high (BOS confirmation)
            if is_sh and highs_arr[i] > structural_high:
                structural_high = highs_arr[i]

    elif current_state == 'Bearish':
        # CHoCH: price closes above the structural high → bullish
        if close_price > structural_high:
            current_state = 'Bullish'
            # Reset structural low for new bullish context
            if is_sl:
                structural_low = lows_arr[i]
        else:
            # Update structural high from swing highs (pullback highs in bear trend)
            if is_sh:
                structural_high = highs_arr[i]
            # Update structural low only if it's a LOWER low (BOS confirmation)
            if is_sl and lows_arr[i] < structural_low:
                structural_low = lows_arr[i]

    ms_trend[i] = current_state

df['ms_trend'] = ms_trend

impulsive_bull = df.index[df['Impulsive'] & df['Bullish']]
impulsive_bear = df.index[df['Impulsive'] & ~df['Bullish']]
print(f'Impulsive candles: {len(impulsive_bull)} bullish, {len(impulsive_bear)} bearish')

# Precompute FVG sets
fvg_bull_set = set()
fvg_bear_set = set()
for i in range(2, n_total):
    if highs_arr[i - 2] < lows_arr[i]:
        fvg_bull_set.update([i - 2, i - 1, i])
    if lows_arr[i - 2] > highs_arr[i]:
        fvg_bear_set.update([i - 2, i - 1, i])

# =============================================================================
# DETECT ORDER BLOCKS WITH ALL FIXES
# =============================================================================
idx_list = df.index.tolist()
idx_pos = {v: i for i, v in enumerate(idx_list)}
n_bars = len(df)

# FIX #1: OB invalidation check — verify price didn't trade through the zone
# between OB formation and impulse candle
def find_opposing_validated(impulse_dt, want_bearish):
    """Find the nearest opposing candle before the impulse, but ONLY if the zone
    wasn't already invalidated (price closed beyond it) between formation and impulse."""
    pos = idx_pos[impulse_dt]
    for step in range(1, OB_LOOKBACK + 1):
        j = pos - step
        if j < 0:
            return None
        row = df.iloc[j]
        if want_bearish and not row['Bullish']:
            # Candidate bearish OB candle for a bullish impulse
            ob_low = row['Low']
            # Check: did any bar between OB and impulse close below ob_low?
            # If so, the zone was already invalidated
            invalidated = False
            for k in range(j + 1, pos):
                if df['Close'].iloc[k] < ob_low:
                    invalidated = True
                    break
            if not invalidated:
                return j
            # If invalidated, keep searching further back
        if not want_bearish and row['Bullish']:
            # Candidate bullish OB candle for a bearish impulse
            ob_high = row['High']
            # Check: did any bar between OB and impulse close above ob_high?
            invalidated = False
            for k in range(j + 1, pos):
                if df['Close'].iloc[k] > ob_high:
                    invalidated = True
                    break
            if not invalidated:
                return j
    return None

# FIX #2: Track which OB candle indices have been used to prevent duplicate claims
used_ob_indices = set()

all_obs = []

for is_bull, impulse_dates in [(True, impulsive_bull), (False, impulsive_bear)]:
    for dt in impulse_dates:
        # FIX #1: Use validated OB finder
        ob_idx = find_opposing_validated(dt, want_bearish=is_bull)
        if ob_idx is None:
            continue

        # FIX #2: Skip if this OB candle was already claimed
        ob_key = (ob_idx, 'bull' if is_bull else 'bear')
        if ob_key in used_ob_indices:
            continue
        used_ob_indices.add(ob_key)

        imp_idx = idx_pos[dt]

        # Skip if not enough future bars for mitigation check
        if imp_idx + MITIGATION_HORIZON >= n_bars:
            continue

        ob_row = df.iloc[ob_idx]
        imp_row = df.iloc[imp_idx]
        atr = ob_row['ATR']
        ob_high = ob_row['High']
        ob_low = ob_row['Low']
        zone_size = ob_high - ob_low

        # --- FIX #3: DISPLACEMENT anchored to IMPULSE, not OB ---
        # Original: measured from ob_low/ob_high which could be 20 bars away
        # Fixed: measure the move LEAVING the impulse candle
        end_disp = min(imp_idx + DISPLACEMENT_BARS + 1, n_bars)
        future_disp = df.iloc[imp_idx + 1:end_disp]
        if is_bull:
            impulse_exit = imp_row['Close']  # bull impulse closes high
            displacement = future_disp['High'].max() - impulse_exit if not future_disp.empty else 0
        else:
            impulse_exit = imp_row['Close']  # bear impulse closes low
            displacement = impulse_exit - future_disp['Low'].min() if not future_disp.empty else 0
        displacement = max(displacement, 0)  # clamp negatives
        disp_atr = displacement / (atr + 1e-10)

        # --- IMPULSE CANDLE PROPERTIES ---
        impulse_body_atr = imp_row['Body'] / (atr + 1e-10)
        impulse_range_atr = (imp_row['High'] - imp_row['Low']) / (atr + 1e-10)

        # --- CONSECUTIVE IMPULSE CANDLES after impulse ---
        consec = 0
        for step in range(imp_idx + 1, min(imp_idx + 6, n_bars)):
            r = df.iloc[step]
            if is_bull and r['Bullish'] and r['Body'] > IMPULSE_ATR_MULT * r['ATR']:
                consec += 1
            elif not is_bull and not r['Bullish'] and r['Body'] > IMPULSE_ATR_MULT * r['ATR']:
                consec += 1
            else:
                break

        # --- FVG CONFLUENCE ---
        if is_bull:
            has_fvg = any(j in fvg_bull_set for j in range(max(imp_idx-2, 0), imp_idx + 1))
        else:
            has_fvg = any(j in fvg_bear_set for j in range(max(imp_idx-2, 0), imp_idx + 1))

        # --- TREND CONTEXT ---
        close_at_ob = ob_row['Close']
        ema50 = ob_row['EMA_50']
        ema200 = ob_row['EMA_200']

        strong_trend = (ema50 > ema200) if is_bull else (ema50 < ema200)

        ob_ms_trend = ob_row['ms_trend']
        if is_bull:
            with_structure = (ob_ms_trend == 'Bullish')
        else:
            with_structure = (ob_ms_trend == 'Bearish')

        # --- OB CANDLE PROPERTIES ---
        ob_body_atr = ob_row['Body'] / (atr + 1e-10)
        ob_zone_atr = zone_size / (atr + 1e-10)

        # --- VOLUME ---
        ob_vma = ob_row['VMA_50']
        ob_tick_vol = ob_row['TickVol']
        imp_tick_vol = imp_row['TickVol']
        ob_vol_rel = ob_tick_vol / (ob_vma + 1e-10)
        imp_vol_rel = imp_tick_vol / (ob_vma + 1e-10)

        # --- SESSION ---
        hour = idx_list[ob_idx].hour
        if 0 <= hour < 8:
            session = 'ASIAN'
        elif 8 <= hour < 13:
            session = 'LONDON'
        elif 13 <= hour < 17:
            session = 'NY_OVERLAP'
        elif 17 <= hour < 22:
            session = 'NY'
        else:
            session = 'LATE'

        # --- DISTANCE FROM EMA ---
        dist_ema50_pct = (close_at_ob - ema50) / close_at_ob * 100

        # --- PREMIUM / DISCOUNT ZONE ---
        roll_mid_at_ob = df.iloc[ob_idx].get('roll_mid', np.nan)
        if not np.isnan(roll_mid_at_ob):
            if is_bull:
                in_discount_zone = close_at_ob < roll_mid_at_ob
            else:
                in_discount_zone = close_at_ob > roll_mid_at_ob
        else:
            in_discount_zone = None

        # --- TOUCH / MITIGATION CHECK ---
        touched = False
        bars_to_touch = None
        touch_bar = None

        end_mit = min(imp_idx + MITIGATION_HORIZON, n_bars)
        for k in range(imp_idx + 1, end_mit):
            if is_bull and df.iloc[k]['Low'] <= ob_high:
                touched = True
                bars_to_touch = k - imp_idx
                touch_bar = k
                break
            if not is_bull and df.iloc[k]['High'] >= ob_low:
                touched = True
                bars_to_touch = k - imp_idx
                touch_bar = k
                break

        mitigated = False
        strong_mitigated = False
        reaction = None
        reaction_atr = None
        max_adverse_atr = None
        trade_rr = None       # MFE-based (kept for reference)
        has_inducement = False
        fib_retracement_pct = None
        touch_close_quality = None
        atr_expansion_ratio = None

        # FIX #5: Fixed-target R:R tracking
        fixed_rr_hits = {f'hit_{rr}R': False for rr in FIXED_RR_TARGETS}
        fixed_rr_exit = None   # The R:R achieved at REACTION_BARS timeout

        if touched and touch_bar is not None:
            # --- FIX #8: RELAXED INDUCEMENT DETECTION ---
            # Original: required ALL swing lows to be above ob_high and sweep of the LOWEST.
            # Fixed: check if ANY swing low/high between impulse and touch was swept on the way in.
            if touch_bar > imp_idx + 1:
                path_to_mitigation = df.iloc[imp_idx + 1:touch_bar]
                if is_bull:
                    swing_lows_before_touch = path_to_mitigation[path_to_mitigation['is_swing_low']]
                    if not swing_lows_before_touch.empty:
                        # Check if ANY swing low was swept (touch bar went below it)
                        touch_low = df.iloc[touch_bar]['Low']
                        for _, sl_row in swing_lows_before_touch.iterrows():
                            if touch_low < sl_row['Low']:
                                has_inducement = True
                                break
                else:
                    swing_highs_before_touch = path_to_mitigation[path_to_mitigation['is_swing_high']]
                    if not swing_highs_before_touch.empty:
                        touch_high = df.iloc[touch_bar]['High']
                        for _, sh_row in swing_highs_before_touch.iterrows():
                            if touch_high > sh_row['High']:
                                has_inducement = True
                                break

            # --- FIB RETRACEMENT DEPTH at touch ---
            swing_bars = touch_bar - imp_idx
            if swing_bars >= MIN_SWING_BARS:
                if is_bull:
                    swing_high = df.iloc[imp_idx:touch_bar]['High'].max()
                    full_move = abs(swing_high - ob_low)
                    retrace_dist = swing_high - df.iloc[touch_bar]['Low']
                    fib_retracement_pct = round(retrace_dist / (full_move + 1e-10) * 100, 1)
                else:
                    swing_low = df.iloc[imp_idx:touch_bar]['Low'].min()
                    full_move = abs(ob_high - swing_low)
                    retrace_dist = df.iloc[touch_bar]['High'] - swing_low
                    fib_retracement_pct = round(retrace_dist / (full_move + 1e-10) * 100, 1)
            else:
                fib_retracement_pct = None

            # --- TOUCH CANDLE CLOSE QUALITY ---
            touch_close = df.iloc[touch_bar]['Close']
            if is_bull:
                if touch_close > ob_high:
                    touch_close_quality = 'rejected'
                elif touch_close >= ob_low:
                    touch_close_quality = 'inside'
                else:
                    touch_close_quality = 'broke'
            else:
                if touch_close < ob_low:
                    touch_close_quality = 'rejected'
                elif touch_close <= ob_high:
                    touch_close_quality = 'inside'
                else:
                    touch_close_quality = 'broke'

            # --- ATR EXPANSION RATIO at touch ---
            atr_at_touch = df.iloc[touch_bar]['ATR']
            atr_expansion_ratio = round(atr_at_touch / (atr + 1e-10), 2)

            # =================================================================
            # FIX #4: REACTION ASSESSMENT STARTS AFTER THE TOUCH BAR
            # =================================================================
            # Original: react_slice started at touch_bar, so the touch candle's
            # wick through the zone could immediately trigger BREAK even if the
            # close rejected. This contradicted touch_close_quality.
            # Fixed: Assess reaction from bar AFTER touch. The touch bar itself
            # is only used for close-quality classification above.
            react_start = touch_bar + 1
            react_end = min(touch_bar + REACTION_BARS + 1, n_bars)
            react_slice = df.iloc[react_start:react_end]

            if is_bull:
                entry = ob_high
                sl = ob_low
                risk = max(entry - sl, 1e-5)

                if len(react_slice) > 0:
                    # Check where price CLOSES below zone low (proper break, not just wick)
                    break_mask = react_slice['Close'] < ob_low
                    has_break = break_mask.any()
                    if has_break:
                        first_break_pos = break_mask.values.argmax()
                        valid_slice = react_slice.iloc[:first_break_pos]
                    else:
                        valid_slice = react_slice

                    if len(valid_slice) > 0:
                        max_adv = max(0, entry - valid_slice['Low'].min())
                        max_fav = max(0, valid_slice['High'].max() - entry)
                    else:
                        max_fav = 0
                        max_adv = max(0, entry - react_slice.iloc[0]['Low'])
                else:
                    max_fav = 0
                    max_adv = 0
                    has_break = False

                # FIX #5: Fixed-target R:R — simulate bar-by-bar
                for k_bar in range(react_start, react_end):
                    bar_data = df.iloc[k_bar]
                    # Check SL hit first (conservative: assume SL checked before TP on same bar)
                    if bar_data['Low'] <= sl:
                        fixed_rr_exit = -1.0
                        break
                    # Check each TP level
                    for rr_target in FIXED_RR_TARGETS:
                        tp = entry + rr_target * risk
                        if bar_data['High'] >= tp and not fixed_rr_hits[f'hit_{rr_target}R']:
                            fixed_rr_hits[f'hit_{rr_target}R'] = True
                else:
                    # Timed out — compute exit R:R at last bar
                    if react_end <= n_bars and len(react_slice) > 0:
                        last_close = react_slice.iloc[-1]['Close']
                        fixed_rr_exit = round((last_close - entry) / risk, 2)
                    else:
                        fixed_rr_exit = 0.0

            else:  # bear OB
                entry = ob_low
                sl = ob_high
                risk = max(sl - entry, 1e-5)

                if len(react_slice) > 0:
                    break_mask = react_slice['Close'] > ob_high
                    has_break = break_mask.any()
                    if has_break:
                        first_break_pos = break_mask.values.argmax()
                        valid_slice = react_slice.iloc[:first_break_pos]
                    else:
                        valid_slice = react_slice

                    if len(valid_slice) > 0:
                        max_adv = max(0, valid_slice['High'].max() - entry)
                        max_fav = max(0, entry - valid_slice['Low'].min())
                    else:
                        max_fav = 0
                        max_adv = max(0, react_slice.iloc[0]['High'] - entry)
                else:
                    max_fav = 0
                    max_adv = 0
                    has_break = False

                # FIX #5: Fixed-target R:R — simulate bar-by-bar
                for k_bar in range(react_start, react_end):
                    bar_data = df.iloc[k_bar]
                    if bar_data['High'] >= sl:
                        fixed_rr_exit = -1.0
                        break
                    for rr_target in FIXED_RR_TARGETS:
                        tp = entry - rr_target * risk
                        if bar_data['Low'] <= tp and not fixed_rr_hits[f'hit_{rr_target}R']:
                            fixed_rr_hits[f'hit_{rr_target}R'] = True
                else:
                    if react_end <= n_bars and len(react_slice) > 0:
                        last_close = react_slice.iloc[-1]['Close']
                        fixed_rr_exit = round((entry - last_close) / risk, 2)
                    else:
                        fixed_rr_exit = 0.0

            reaction_atr = round(max_fav / (atr + 1e-10), 2)
            max_adverse_atr = round(max_adv / (atr + 1e-10), 2)
            trade_rr = round(max_fav / risk, 2)  # MFE R:R (theoretical ceiling)

            if has_break:
                reaction = 'BREAK'
                mitigated = False
            elif max_fav >= 1.0 * atr:
                reaction = 'BOUNCE'
                mitigated = True
                strong_mitigated = True
            else:
                reaction = 'WEAK'
                mitigated = True

        record = {
            'type': 'bull' if is_bull else 'bear',
            'ob_idx': ob_idx,
            'ob_dt': idx_list[ob_idx],
            'imp_idx': imp_idx,
            'ob_high': ob_high,
            'ob_low': ob_low,
            'atr': round(atr, 3),
            'disp_atr': round(disp_atr, 2),
            'impulse_body_atr': round(impulse_body_atr, 2),
            'impulse_range_atr': round(impulse_range_atr, 2),
            'consec_impulse': consec,
            'has_fvg': has_fvg,
            'with_structure': with_structure,
            'strong_trend': strong_trend,
            'ob_body_atr': round(ob_body_atr, 2),
            'ob_zone_atr': round(ob_zone_atr, 2),
            'ob_vol_rel': round(ob_vol_rel, 2),
            'imp_vol_rel': round(imp_vol_rel, 2),
            'session': session,
            'dist_ema50_pct': round(dist_ema50_pct, 3),
            'has_inducement': has_inducement,
            'in_discount_zone': in_discount_zone,
            'touched': touched,
            'mitigated': mitigated,
            'strong_mitigated': strong_mitigated,
            'bars_to_mitigate': bars_to_touch,
            'reaction': reaction,
            'reaction_atr': reaction_atr,
            'max_adverse_atr': max_adverse_atr,
            'trade_rr_mfe': trade_rr,
            'fib_retracement_pct': fib_retracement_pct,
            'touch_close_quality': touch_close_quality,
            'atr_expansion_ratio': atr_expansion_ratio,
        }
        # FIX #5: Add fixed R:R columns
        for rr_target in FIXED_RR_TARGETS:
            record[f'hit_{rr_target}R'] = fixed_rr_hits.get(f'hit_{rr_target}R', False)
        record['fixed_rr_exit'] = fixed_rr_exit

        all_obs.append(record)

# =============================================================================
# FIX #9: DEDUPLICATE OVERLAPPING ZONES
# =============================================================================
# If two OBs of the same type have overlapping price zones, keep only the one
# closer to the impulse (more recent formation = more relevant).

ob_df = pd.DataFrame(all_obs)

def deduplicate_overlapping_zones(df_in):
    """Remove OBs whose zones overlap with a previously-kept OB of the same type."""
    if len(df_in) == 0:
        return df_in
    
    kept = []
    for ob_type in ['bull', 'bear']:
        type_obs = df_in[df_in['type'] == ob_type].sort_values('imp_idx').copy()
        kept_zones = []  # list of (ob_low, ob_high, imp_idx)
        
        for _, row in type_obs.iterrows():
            overlap = False
            for (kl, kh, _) in kept_zones:
                # Check overlap: two intervals [a,b] and [c,d] overlap if a <= d and c <= b
                if row['ob_low'] <= kh and kl <= row['ob_high']:
                    overlap = True
                    break
            if not overlap:
                kept.append(row)
                kept_zones.append((row['ob_low'], row['ob_high'], row['imp_idx']))
    
    return pd.DataFrame(kept)

pre_dedup = len(ob_df)
ob_df = deduplicate_overlapping_zones(ob_df)
ob_df = ob_df.dropna(subset=['in_discount_zone', 'disp_atr']).reset_index(drop=True)
post_dedup = len(ob_df)

print(f'\nTotal OBs before dedup: {pre_dedup}')
print(f'Total OBs after dedup:  {post_dedup}  (removed {pre_dedup - post_dedup} overlapping)')
print(f'  Bullish: {(ob_df["type"]=="bull").sum()}  |  Bearish: {(ob_df["type"]=="bear").sum()}')
print(f'  Touched zone: {ob_df["touched"].sum()} ({ob_df["touched"].mean()*100:.1f}%)')
print(f'  Mitigated (Respected): {ob_df["mitigated"].sum()} ({ob_df["mitigated"].mean()*100:.1f}%)')
print(f'  Strong Mitigated (Bounce): {ob_df["strong_mitigated"].sum()} ({ob_df["strong_mitigated"].mean()*100:.1f}%)')
print(f'  Not touched (within {MITIGATION_HORIZON} bars): {(~ob_df["touched"]).sum()}')

# =============================================================================
# MITIGATION RATE BY SINGLE FILTERS
# =============================================================================
print('\n' + '='*90)
print('MITIGATION RATE BY FILTER (on all OBs)')
print('='*90)

MIN_SAMPLE = 30  # FIX: Raised from 5 to 30 for statistical significance

def print_filter(label, mask, use_strong=False):
    subset = ob_df[mask]
    if len(subset) < MIN_SAMPLE:
        return
    if use_strong:
        rate = subset['strong_mitigated'].mean() * 100
    else:
        rate = subset['mitigated'].mean() * 100
    count = len(subset)
    col = 'strong_mitigated' if use_strong else 'mitigated'
    avg_bars = subset[subset[col]]['bars_to_mitigate'].mean()
    avg_bars_str = f'{avg_bars:>6.0f}' if not np.isnan(avg_bars) else '   N/A'
    print(f'{label:<45} {rate:>6.1f}%  ({count:>5} OBs)  avg bars: {avg_bars_str}')

base_rate = ob_df['mitigated'].mean() * 100
base_strong_rate = ob_df['strong_mitigated'].mean() * 100
print(f'\n{"BASELINE (all OBs)":<45} {base_rate:>6.1f}% mitigated / {base_strong_rate:>6.1f}% strong ({len(ob_df):>5} OBs)')
print('-'*90)

# --- TYPE ---
print('\n--- BY TYPE ---')
print_filter('Bull OBs', ob_df['type'] == 'bull')
print_filter('Bear OBs', ob_df['type'] == 'bear')

# --- DISPLACEMENT STRENGTH ---
print('\n--- BY DISPLACEMENT (ATR multiples) ---')
for thresh in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
    print_filter(f'Displacement >= {thresh}x ATR', ob_df['disp_atr'] >= thresh)
for thresh in [1.0, 1.5, 2.0, 3.0]:
    print_filter(f'Displacement < {thresh}x ATR', ob_df['disp_atr'] < thresh)

# --- IMPULSE CANDLE SIZE ---
print('\n--- BY IMPULSE CANDLE BODY SIZE ---')
for thresh in [1.5, 2.0, 2.5, 3.0]:
    print_filter(f'Impulse body >= {thresh}x ATR', ob_df['impulse_body_atr'] >= thresh)

# --- CONSECUTIVE IMPULSE CANDLES ---
print('\n--- BY CONSECUTIVE IMPULSE CANDLES ---')
for n in [1, 2, 3]:
    print_filter(f'Consecutive impulse >= {n}', ob_df['consec_impulse'] >= n)

# --- FVG ---
print('\n--- BY FVG CONFLUENCE ---')
print_filter('Has FVG', ob_df['has_fvg'] == True)
print_filter('No FVG', ob_df['has_fvg'] == False)

# --- TREND ---
print('\n--- BY TRUE MARKET STRUCTURE (CHoCH/BOS) ---')
print_filter('With True Structure', ob_df['with_structure'])
print_filter('Counter Structure', ~ob_df['with_structure'])
print_filter('Strong trend (EMA50>200)', ob_df['strong_trend'] == True)
print_filter('Weak/no trend', ob_df['strong_trend'] == False)

# --- OB ZONE SIZE ---
print('\n--- BY OB ZONE SIZE ---')
median_zone = ob_df['ob_zone_atr'].median()
print_filter(f'Tight OB zone (< {median_zone:.1f}x ATR)', ob_df['ob_zone_atr'] < median_zone)
print_filter(f'Wide OB zone (>= {median_zone:.1f}x ATR)', ob_df['ob_zone_atr'] >= median_zone)
for thresh in [0.5, 0.75, 1.0, 1.5]:
    print_filter(f'OB zone <= {thresh}x ATR', ob_df['ob_zone_atr'] <= thresh)

# --- VOLUME ---
print('\n--- BY RELATIVE VOLUME ---')
print_filter('High OB Vol (> 1.2x VMA)', ob_df['ob_vol_rel'] > 1.2)
print_filter('Low OB Vol (<= 1.2x VMA)', ob_df['ob_vol_rel'] <= 1.2)
print_filter('High Impulse Vol (> 1.5x VMA)', ob_df['imp_vol_rel'] > 1.5)
print_filter('Very High Impulse Vol (> 2.0x VMA)', ob_df['imp_vol_rel'] > 2.0)

# --- SESSION ---
print('\n--- BY SESSION ---')
for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
    print_filter(f'Session: {sess}', ob_df['session'] == sess)

# --- DISTANCE FROM EMA ---
print('\n--- BY DISTANCE FROM EMA50 ---')
ob_df['abs_dist_ema50'] = ob_df['dist_ema50_pct'].abs()
for thresh in [0.1, 0.2, 0.5, 1.0]:
    print_filter(f'Close within {thresh}% of EMA50', ob_df['abs_dist_ema50'] <= thresh)
    print_filter(f'Close > {thresh}% from EMA50', ob_df['abs_dist_ema50'] > thresh)

# =============================================================================
# BEST FILTER COMBINATIONS
# =============================================================================
print('\n' + '='*90)
print('BEST FILTER COMBINATIONS')
print('='*90)
print(f'\n{"Combo Filter":<55} {"Mit %":>6}  {"Count":>7}  {"Avg Bars":>9}')
print('-'*90)

combos = [
    ('FVG + with_structure', ob_df['has_fvg'] & ob_df['with_structure']),
    ('FVG + strong_trend', ob_df['has_fvg'] & ob_df['strong_trend']),
    ('FVG + disp >= 3x', ob_df['has_fvg'] & (ob_df['disp_atr'] >= 3.0)),
    ('FVG + disp >= 2x + with_structure', ob_df['has_fvg'] & (ob_df['disp_atr'] >= 2.0) & ob_df['with_structure']),
    ('FVG + consec >= 2', ob_df['has_fvg'] & (ob_df['consec_impulse'] >= 2)),
    ('FVG + consec >= 2 + with_structure', ob_df['has_fvg'] & (ob_df['consec_impulse'] >= 2) & ob_df['with_structure']),
    ('FVG + tight zone (<0.75 ATR)', ob_df['has_fvg'] & (ob_df['ob_zone_atr'] <= 0.75)),
    ('Strong trend + disp >= 3x', ob_df['strong_trend'] & (ob_df['disp_atr'] >= 3.0)),
    ('Strong trend + FVG + disp >= 2x', ob_df['strong_trend'] & ob_df['has_fvg'] & (ob_df['disp_atr'] >= 2.0)),
    ('LONDON + FVG + with_structure', (ob_df['session']=='LONDON') & ob_df['has_fvg'] & ob_df['with_structure']),
    ('NY_OVERLAP + FVG + with_structure', (ob_df['session']=='NY_OVERLAP') & ob_df['has_fvg'] & ob_df['with_structure']),
    ('Impulse body >= 2x + FVG', (ob_df['impulse_body_atr'] >= 2.0) & ob_df['has_fvg']),
    ('Impulse Vol > 1.5x + FVG', (ob_df['imp_vol_rel'] > 1.5) & ob_df['has_fvg']),
    ('Impulse Vol > 2.0x + FVG + disp >= 3x', (ob_df['imp_vol_rel'] > 2.0) & ob_df['has_fvg'] & (ob_df['disp_atr'] >= 3.0)),
    ('High OB Vol (> 1.2x) + FVG', (ob_df['ob_vol_rel'] > 1.2) & ob_df['has_fvg']),
    ('FVG + Inducement', ob_df['has_fvg'] & ob_df['has_inducement']),
    ('Inducement + True Structure', ob_df['has_inducement'] & ob_df['with_structure']),
    ('FVG + Inducement + High Vol (>1.2)', ob_df['has_fvg'] & ob_df['has_inducement'] & (ob_df['ob_vol_rel'] > 1.2)),
    ('Impulse body >= 2x + FVG + structure', (ob_df['impulse_body_atr'] >= 2.0) & ob_df['has_fvg'] & ob_df['with_structure']),
    ('Counter structure + FVG + disp >= 3x', ~ob_df['with_structure'] & ob_df['has_fvg'] & (ob_df['disp_atr'] >= 3.0)),
    ('All: FVG+struct+disp3+consec2', ob_df['has_fvg'] & ob_df['with_structure'] & (ob_df['disp_atr'] >= 3.0) & (ob_df['consec_impulse'] >= 2)),
]

combo_results = []
for label, mask in combos:
    subset = ob_df[mask]
    if len(subset) < MIN_SAMPLE:
        continue
    rate = subset['mitigated'].mean() * 100
    avg_bars = subset[subset['mitigated']]['bars_to_mitigate'].mean()
    avg_bars_str = f'{avg_bars:>8.0f}' if not np.isnan(avg_bars) else '     N/A'
    combo_results.append((label, rate, len(subset), avg_bars_str))

combo_results.sort(key=lambda x: x[1], reverse=True)
for label, rate, count, avg_bars in combo_results:
    marker = ' <<<' if rate > base_rate + 5 else ''
    print(f'{label:<55} {rate:>6.1f}%  ({count:>5})  {avg_bars}{marker}')

print(f'\nBaseline mitigation rate: {base_rate:.1f}%')

# =============================================================================
# BARS TO TOUCH DISTRIBUTION
# =============================================================================
mit_df = ob_df[ob_df['touched']].copy()
if len(mit_df) > 0:
    print('\n' + '='*90)
    print('BARS TO TOUCH (touched OBs only)')
    print('='*90)
    print(f'  Median: {mit_df["bars_to_mitigate"].median():.0f} bars')
    print(f'  Mean:   {mit_df["bars_to_mitigate"].mean():.0f} bars')
    print(f'  25th:   {mit_df["bars_to_mitigate"].quantile(0.25):.0f} bars')
    print(f'  75th:   {mit_df["bars_to_mitigate"].quantile(0.75):.0f} bars')
    print(f'\n  Mitigated within:')
    for b in [5, 10, 20, 50, 100]:
        pct = (mit_df['bars_to_mitigate'] <= b).mean() * 100
        print(f'    <= {b:>3} bars: {pct:>5.1f}%')

# =============================================================================
# REACTION QUALITY — BOUNCE vs BREAK
# =============================================================================
print('\n' + '='*90)
print('REACTION QUALITY — BOUNCE vs BREAK (touched OBs only)')
print('='*90)

total_mit = len(mit_df)
bounces = (mit_df['reaction'] == 'BOUNCE').sum()
breaks = (mit_df['reaction'] == 'BREAK').sum()
weak = (mit_df['reaction'] == 'WEAK').sum()

print(f'\nTotal touched OBs: {total_mit}')
print(f'  BOUNCE (held, >= 1 ATR favour): {bounces} ({bounces/total_mit*100:.1f}%)')
print(f'  BREAK  (closed beyond zone):    {breaks} ({breaks/total_mit*100:.1f}%)')
print(f'  WEAK   (held but barely moved): {weak} ({weak/total_mit*100:.1f}%)')

bounce_df = mit_df[mit_df['reaction'] == 'BOUNCE']
if len(bounce_df) > 0:
    print(f'\nBOUNCE stats (MFE-based, theoretical ceiling):')
    print(f'  Avg MFE reaction:    {bounce_df["reaction_atr"].mean():.1f}x ATR')
    print(f'  Avg max adverse:     {bounce_df["max_adverse_atr"].mean():.1f}x ATR')
    print(f'  Avg MFE R:R:         {bounce_df["trade_rr_mfe"].mean():.1f}')
    print(f'  Median MFE R:R:      {bounce_df["trade_rr_mfe"].median():.1f}')

# FIX #5: REALISTIC FIXED-TARGET R:R STATS
print(f'\n--- FIXED-TARGET R:R PERFORMANCE (realistic) ---')
for rr_target in FIXED_RR_TARGETS:
    col = f'hit_{rr_target}R'
    if col in mit_df.columns:
        hit_rate = mit_df[col].mean() * 100
        # Expected value: (hit_rate * rr_target) - ((1-hit_rate) * 1.0)
        ev = (hit_rate/100 * rr_target) - ((1 - hit_rate/100) * 1.0)
        print(f'  {rr_target}R target: {hit_rate:>5.1f}% hit rate  |  EV per trade: {ev:>+.2f}R')

base_bounce = bounces / total_mit * 100 if total_mit > 0 else 0

# =============================================================================
# BOUNCE RATE BY FILTER
# =============================================================================
print('\n' + '='*90)
print('BOUNCE RATE BY FILTER (what predicts a GOOD reaction)')
print('='*90)
print(f'\n{"BASELINE (all touched)":<50} {base_bounce:>6.1f}% bounce  ({total_mit} OBs)')
print('-'*90)

def print_bf(label, mask):
    s = mit_df[mask]
    if len(s) < MIN_SAMPLE:
        return
    b = (s['reaction'] == 'BOUNCE').sum()
    r = b / len(s) * 100
    rr = s[s['reaction']=='BOUNCE']['trade_rr_mfe'].mean() if b > 0 else 0
    # FIX #5: Add 2R hit rate as a realistic metric
    hit_2r = s['hit_2.0R'].mean() * 100 if 'hit_2.0R' in s.columns else 0
    print(f'{label:<50} {r:>6.1f}% bounce  ({len(s):>5} OBs)  MFE RR: {rr:>4.1f}  2R hit: {hit_2r:>4.1f}%')

print('\n--- BY TYPE ---')
print_bf('Bull OBs', mit_df['type'] == 'bull')
print_bf('Bear OBs', mit_df['type'] == 'bear')

print('\n--- BY DISPLACEMENT ---')
for t in [2.0, 3.0, 4.0, 5.0]:
    print_bf(f'Displacement >= {t}x ATR', mit_df['disp_atr'] >= t)

print('\n--- BY IMPULSE BODY SIZE ---')
for t in [1.5, 2.0, 2.5, 3.0]:
    print_bf(f'Impulse body >= {t}x ATR', mit_df['impulse_body_atr'] >= t)

print('\n--- BY CONSECUTIVE IMPULSE ---')
for n in [1, 2, 3]:
    print_bf(f'Consecutive impulse >= {n}', mit_df['consec_impulse'] >= n)

print('\n--- BY FVG ---')
print_bf('Has FVG', mit_df['has_fvg'] == True)
print_bf('No FVG', mit_df['has_fvg'] == False)

print('\n--- BY INDUCEMENT ---')
print_bf('Has Inducement (swept liquidity)', mit_df['has_inducement'] == True)
print_bf('No Inducement', mit_df['has_inducement'] == False)

print('\n--- BY TRUE MARKET STRUCTURE (CHoCH/BOS) ---')
print_bf('With True Structure', mit_df['with_structure'] == True)
print_bf('Counter Structure', mit_df['with_structure'] == False)
print_bf('Strong trend (EMA50>200)', mit_df['strong_trend'] == True)

print('\n--- BY ZONE SIZE ---')
for t in [0.5, 0.75, 1.0]:
    print_bf(f'Zone <= {t}x ATR', mit_df['ob_zone_atr'] <= t)
    print_bf(f'Zone > {t}x ATR', mit_df['ob_zone_atr'] > t)

print('\n--- BY VOLUME ---')
print_bf('High OB Vol (> 1.2x VMA)', mit_df['ob_vol_rel'] > 1.2)
print_bf('Low OB Vol (<= 1.2x VMA)', mit_df['ob_vol_rel'] <= 1.2)
print_bf('High Impulse Vol (> 1.5x VMA)', mit_df['imp_vol_rel'] > 1.5)
print_bf('Very High Impulse Vol (> 2.0x VMA)', mit_df['imp_vol_rel'] > 2.0)

print('\n--- BY SESSION ---')
for sess in ['ASIAN', 'LONDON', 'NY_OVERLAP', 'NY', 'LATE']:
    print_bf(f'Session: {sess}', mit_df['session'] == sess)

print('\n--- BY BARS TO MITIGATION ---')
for b in [5, 10, 20, 50]:
    print_bf(f'Mitigated within {b} bars', mit_df['bars_to_mitigate'] <= b)
    print_bf(f'Mitigated after {b} bars', mit_df['bars_to_mitigate'] > b)

print('\n--- BY PREMIUM / DISCOUNT ALIGNMENT ---')
disc_mask = mit_df['in_discount_zone'] == True
print_bf('In discount/premium zone (aligned)', disc_mask)
print_bf('Counter zone (misaligned)', mit_df['in_discount_zone'] == False)

print('\n--- BY TOUCH CANDLE CLOSE QUALITY ---')
print_bf('Touch closed REJECTED (back outside zone)', mit_df['touch_close_quality'] == 'rejected')
print_bf('Touch closed INSIDE zone', mit_df['touch_close_quality'] == 'inside')
print_bf('Touch closed BROKE zone (thru SL side)', mit_df['touch_close_quality'] == 'broke')

print('\n--- BY FIB RETRACEMENT DEPTH ---')
fib = mit_df['fib_retracement_pct']
print_bf('Fib: 38-62% (golden zone)',    (fib >= 38) & (fib <= 62))
print_bf('Fib: 50-78.6% (deep retrace)', (fib >= 50) & (fib <= 79))
print_bf('Fib: <38% (shallow)',          fib < 38)
print_bf('Fib: >79% (over-extended)',    fib > 79)
print_bf('Fib: >100% (broke structure)', fib > 100)

print('\n--- BY ATR EXPANSION AT TOUCH ---')
exp = mit_df['atr_expansion_ratio']
print_bf('ATR expanding at touch (>1.1x formation ATR)', exp > 1.1)
print_bf('ATR contracting at touch (<0.9x formation ATR)', exp < 0.9)
print_bf('ATR spike at touch (>1.3x)', exp > 1.3)

# =============================================================================
# ACTIONABLE ENTRY-TIME COMBOS
# =============================================================================
print('\n' + '='*90)
print('ACTIONABLE ENTRY-TIME COMBOS (known at OB formation)')
print('='*90)
print(f'\n{"Combo":<58} {"Bnc%":>5} {"Cnt":>5} {"MFE":>5} {"2Rhit":>6}')
print('-'*90)

_disc  = mit_df['in_discount_zone'] == True
_hvol  = mit_df['ob_vol_rel'] > 1.2
_hivol = mit_df['imp_vol_rel'] > 1.5
_vhivol = mit_df['imp_vol_rel'] > 2.0

tc = [
    # Single filters
    ('FVG only', mit_df['has_fvg']),
    ('High OB Vol (>1.2) only', _hvol),
    ('Discount zone only', _disc),
    ('True Structure only', mit_df['with_structure']),
    ('Strong trend only', mit_df['strong_trend']),
    ('Inducement only', mit_df['has_inducement']),

    # FVG combos
    ('FVG + with_structure', mit_df['has_fvg'] & mit_df['with_structure']),
    ('FVG + strong_trend', mit_df['has_fvg'] & mit_df['strong_trend']),
    ('FVG + disp >= 3x', mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0)),
    ('FVG + disp >= 4x', mit_df['has_fvg'] & (mit_df['disp_atr'] >= 4.0)),
    ('FVG + disp >= 5x', mit_df['has_fvg'] & (mit_df['disp_atr'] >= 5.0)),
    ('FVG + disp >= 3x + structure', mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0) & mit_df['with_structure']),
    ('FVG + consec >= 2', mit_df['has_fvg'] & (mit_df['consec_impulse'] >= 2)),
    ('FVG + consec >= 2 + structure', mit_df['has_fvg'] & (mit_df['consec_impulse'] >= 2) & mit_df['with_structure']),
    ('FVG + impulse >= 2x', mit_df['has_fvg'] & (mit_df['impulse_body_atr'] >= 2.0)),
    ('FVG + impulse >= 2x + structure', mit_df['has_fvg'] & (mit_df['impulse_body_atr'] >= 2.0) & mit_df['with_structure']),
    ('FVG + High OB Vol', mit_df['has_fvg'] & _hvol),
    ('FVG + High Impulse Vol (>1.5)', mit_df['has_fvg'] & _hivol),
    ('FVG + V.High Imp Vol (>2)', mit_df['has_fvg'] & _vhivol),
    ('FVG + V.High Imp Vol + disp>=3', mit_df['has_fvg'] & _vhivol & (mit_df['disp_atr'] >= 3.0)),
    ('FVG + tight zone (<0.75 ATR)', mit_df['has_fvg'] & (mit_df['ob_zone_atr'] <= 0.75)),
    ('FVG + tight zone + structure', mit_df['has_fvg'] & (mit_df['ob_zone_atr'] <= 0.75) & mit_df['with_structure']),
    ('FVG + disp>=3 + tight zone', mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0) & (mit_df['ob_zone_atr'] <= 0.75)),

    # Discount zone combos
    ('Discount + FVG', _disc & mit_df['has_fvg']),
    ('Discount + High OB Vol', _disc & _hvol),
    ('Discount + FVG + disp>=3', _disc & mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0)),
    ('Discount + FVG + structure', _disc & mit_df['has_fvg'] & mit_df['with_structure']),
    ('Discount + FVG + High Vol', _disc & mit_df['has_fvg'] & _hvol),

    # Inducement combos
    ('Inducement + FVG', mit_df['has_inducement'] & mit_df['has_fvg']),
    ('Inducement + structure', mit_df['has_inducement'] & mit_df['with_structure']),
    ('Inducement + FVG + High Vol', mit_df['has_inducement'] & mit_df['has_fvg'] & _hvol),

    # Session combos
    ('LONDON + FVG', (mit_df['session']=='LONDON') & mit_df['has_fvg']),
    ('LONDON + FVG + structure', (mit_df['session']=='LONDON') & mit_df['has_fvg'] & mit_df['with_structure']),
    ('NY_OVERLAP + FVG', (mit_df['session']=='NY_OVERLAP') & mit_df['has_fvg']),
    ('NY_OVERLAP + FVG + structure', (mit_df['session']=='NY_OVERLAP') & mit_df['has_fvg'] & mit_df['with_structure']),
    ('ASIAN + FVG', (mit_df['session']=='ASIAN') & mit_df['has_fvg']),

    # Volume combos
    ('High OB Vol + disp>=3', _hvol & (mit_df['disp_atr'] >= 3.0)),
    ('High OB Vol + structure', _hvol & mit_df['with_structure']),
    ('High OB Vol + FVG + disp>=3', _hvol & mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0)),

    # Multi-filter combos
    ('FVG + structure + disp>=3', mit_df['has_fvg'] & mit_df['with_structure'] & (mit_df['disp_atr'] >= 3.0)),
    ('FVG + structure + High Vol', mit_df['has_fvg'] & mit_df['with_structure'] & _hvol),
    ('Disc + FVG + struct + disp>=3', _disc & mit_df['has_fvg'] & mit_df['with_structure'] & (mit_df['disp_atr'] >= 3.0)),
    ('Strong trend + FVG + disp>=3', mit_df['strong_trend'] & mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0)),
    ('Strong + FVG + disp>=3 + consec>=2', mit_df['strong_trend'] & mit_df['has_fvg'] & (mit_df['disp_atr'] >= 3.0) & (mit_df['consec_impulse'] >= 2)),
]

tr = []
for label, mask in tc:
    s = mit_df[mask]
    if len(s) < MIN_SAMPLE:
        continue
    b = (s['reaction'] == 'BOUNCE').sum()
    r = b / len(s) * 100
    rr = s[s['reaction']=='BOUNCE']['trade_rr_mfe'].mean() if b > 0 else 0
    hit_2r = s['hit_2.0R'].mean() * 100 if 'hit_2.0R' in s.columns else 0
    tr.append((label, r, len(s), rr, hit_2r, mask))

tr.sort(key=lambda x: x[1], reverse=True)
for label, rate, count, avg_rr, hit_2r, _ in tr:
    mk = ' <<<' if rate > base_bounce + 5 else ''
    print(f'{label:<58} {rate:>5.1f} {count:>5}  {avg_rr:>4.1f}  {hit_2r:>5.1f}%{mk}')

print(f'\nBaseline bounce rate: {base_bounce:.1f}%')

# =============================================================================
# FIX #5: EXPECTED VALUE TABLE FOR TOP COMBOS AT EACH R:R TARGET
# =============================================================================
print('\n' + '='*90)
print('EXPECTED VALUE PER TRADE — TOP COMBOS AT FIXED R:R TARGETS')
print('='*90)
print(f'\n{"Combo":<45}', end='')
for rr_target in FIXED_RR_TARGETS:
    print(f'  {rr_target}R EV', end='')
print(f'  {"Cnt":>5}')
print('-'*100)

for label, rate, count, avg_rr, hit_2r, mask in tr[:15]:
    s = mit_df[mask]
    print(f'{label:<45}', end='')
    for rr_target in FIXED_RR_TARGETS:
        col = f'hit_{rr_target}R'
        if col in s.columns:
            hr = s[col].mean()
            ev = (hr * rr_target) - ((1 - hr) * 1.0)
            print(f'  {ev:>+5.2f}', end='')
        else:
            print(f'    N/A', end='')
    print(f'  {count:>5}')

# =============================================================================
# SAVE
# =============================================================================
save_path = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_mitigation_study.csv')
os.makedirs(os.path.dirname(save_path), exist_ok=True)
ob_df.to_csv(save_path, index=False)
print(f'\nSaved {len(ob_df)} OBs to {save_path}')

# =============================================================================
# EXPORT TOP 3 COMBOS
# =============================================================================
print('\n' + '='*90)
print('EXPORTING TOP 3 COMBO TRADE LISTS')
print('='*90)

combo_dir = os.path.join(PROJECT_ROOT, 'data', 'processed', 'ob_combos')
os.makedirs(combo_dir, exist_ok=True)

for rank, (label, rate, count, avg_rr, hit_2r, mask) in enumerate(tr[:3], 1):
    combo_trades = mit_df[mask].copy()
    safe_name = label.replace(' ', '_').replace('>', 'gt').replace('<', 'lt')
    safe_name = safe_name.replace('+', '_').replace('(', '').replace(')', '')
    safe_name = safe_name.replace(':', '').replace('%', 'pct').replace('=', 'eq')
    safe_name = safe_name.replace('__', '_').replace('___', '_').strip('_')
    fname = f"top{rank}_{safe_name}.csv"
    fpath = os.path.join(combo_dir, fname)
    combo_trades.to_csv(fpath, index=False)
    bounce_n = (combo_trades['reaction'] == 'BOUNCE').sum()
    print(f"\n  #{rank}: {label}")
    print(f"       Bounce: {rate:.1f}% ({bounce_n}/{count})  MFE RR: {avg_rr:.1f}  2R hit: {hit_2r:.1f}%")
    print(f"       → {fpath}")

print('\n' + '='*90)
print('FIXES APPLIED:')
print('  #1: OB invalidation — zones broken before impulse are skipped')
print('  #2: Deduplication — each OB candle claimed only once per direction')
print('  #3: Displacement anchored to impulse close, not OB low/high')
print('  #4: Reaction assessed from bar AFTER touch; BREAK = close-based')
print('  #5: Fixed-target R:R (1R/1.5R/2R/3R) with EV alongside MFE')
print('  #6: Market structure pivots scoped to trend context')
print('  #7: Swing detection with right-side confirmation')
print('  #8: Inducement detects ANY swept liquidity, not just perfect staircase')
print('  #9: Overlapping same-direction zones deduplicated')
print(f'  Min sample size raised to {MIN_SAMPLE} (was 5)')
print('='*90)