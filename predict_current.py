"""
Live Trading Prediction Script for XAUUSD 5-Minute Timeframe
==============================================================
This script calculates the probability of success for the current candle
using pre-trained Random Forest and Logistic Regression models.

Usage:
    python predict_current.py                    # Fetch live data from yfinance
    python predict_current.py --live             # Same as above
    python predict_current.py <csv_file>         # Use historical CSV file
    
    Example: python predict_current.py
    Example: python predict_current.py data/raw/XAUUSD5.csv

For live data: Automatically fetches last 50 bars of 5-minute XAUUSD data
For CSV: File should have columns: Datetime, Open, High, Low, Close, Volume (tab-separated)
"""

import pandas as pd
import numpy as np
import pickle
import sys
import os
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

def create_microstructure_features(df):
    """Create all 29 features used by the models"""
    df = df.copy()
    
    # Price structure
    df['range'] = df['High'] - df['Low']
    df['body'] = df['Close'] - df['Open']
    df['abs_body'] = abs(df['body'])
    df['upper_wick'] = df['High'] - df[['Open','Close']].max(axis=1)
    df['lower_wick'] = df[['Open','Close']].min(axis=1) - df['Low']
    df['body_pct'] = df['abs_body'] / (df['range'] + 1e-10)
    
    # Order flow
    df['close_position'] = (df['Close'] - df['Low']) / (df['range'] + 1e-10)
    df['directional_flow'] = df['body'] / df['Close']
    df['flow_3'] = df['directional_flow'].rolling(3).sum()
    df['flow_5'] = df['directional_flow'].rolling(5).sum()
    df['flow_10'] = df['directional_flow'].rolling(10).sum()
    df['flow_momentum'] = df['flow_3'] - df['flow_5'].shift(2)
    
    # Imbalance (strong directional pressure)
    df['buy_imbalance'] = ((df['body'] > 0) & (df['body_pct'] > 0.6) & (df['close_position'] > 0.7)).astype(float)
    df['sell_imbalance'] = ((df['body'] < 0) & (df['body_pct'] > 0.6) & (df['close_position'] < 0.3)).astype(float)
    df['imbalance_3'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(3).sum()
    df['imbalance_5'] = (df['buy_imbalance'] - df['sell_imbalance']).rolling(5).sum()
    
    # Momentum consistency - is_up = bullish candle (close > open)
    df['is_up'] = (df['Close'] > df['Open']).astype(int)
    df['up_count_3'] = df['is_up'].rolling(3).sum()
    df['up_count_5'] = df['is_up'].rolling(5).sum()
    df['consistency_3'] = df['up_count_3'].apply(lambda x: max(x, 3-x))
    df['consistency_5'] = df['up_count_5'].apply(lambda x: max(x, 5-x))
    
    # Volatility
    df['atr_3'] = df['range'].rolling(3).mean()
    df['atr_10'] = df['range'].rolling(10).mean()
    df['atr_20'] = df['range'].rolling(20).mean()
    df['vol_ratio'] = df['atr_3'] / (df['atr_10'] + 1e-10)
    df['vol_expansion'] = (df['range'] > df['atr_10'] * 1.2).astype(int)
    df['vol_contraction'] = (df['range'] < df['atr_10'] * 0.7).astype(int)
    
    # Trend structure
    df['ema_8'] = df['Close'].ewm(span=8).mean()
    df['ema_21'] = df['Close'].ewm(span=21).mean()
    df['trend_align'] = ((df['Close'] > df['ema_8']) & (df['ema_8'] > df['ema_21'])).astype(int) - ((df['Close'] < df['ema_8']) & (df['ema_8'] < df['ema_21'])).astype(int)
    df['dist_ema8'] = (df['Close'] - df['ema_8']) / df['Close']
    
    # Support/Resistance
    df['high_10'] = df['High'].rolling(10).max()
    df['low_10'] = df['Low'].rolling(10).min()
    df['at_high'] = (df['Close'] >= df['high_10'].shift(1) * 0.9999).astype(int)
    df['at_low'] = (df['Close'] <= df['low_10'].shift(1) * 1.0001).astype(int)
    
    # Rejection patterns
    df['upper_reject'] = (df['upper_wick'] > df['abs_body'] * 2).astype(int)
    df['lower_reject'] = (df['lower_wick'] > df['abs_body'] * 2).astype(int)
    
    # Size patterns
    df['big_body'] = (df['abs_body'] > df['abs_body'].rolling(10).mean() * 1.5).astype(int)
    df['small_body'] = (df['abs_body'] < df['abs_body'].rolling(10).mean() * 0.5).astype(int)
    
    return df

def create_combo_features(df):
    """Create combination features"""
    df['combo_flow_trend'] = df['flow_momentum'] * df['trend_align']
    df['combo_vol_imbalance'] = df['vol_ratio'] * df['imbalance_3']
    df['combo_consistency_position'] = df['consistency_5'] * df['close_position']
    df['combo_body_reject'] = df['big_body'] * (df['lower_reject'] - df['upper_reject'])
    df['combo_trend_volatility'] = df['trend_align'] * df['vol_expansion']
    df['combo_imbalance_momentum'] = df['imbalance_5'] * df['flow_5']
    df['combo_position_consistency'] = df['close_position'] * df['consistency_3']
    df['combo_vol_flow'] = df['vol_ratio'] * df['flow_3']
    return df

def load_models():
    """Load pre-trained models and scaler"""
    try:
        with open('models/random_forest.pkl', 'rb') as f:
            rf = pickle.load(f)
        with open('models/logistic_regression.pkl', 'rb') as f:
            lr = pickle.load(f)
        with open('models/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('models/feature_names.txt', 'r') as f:
            feature_names = [line.strip() for line in f.readlines()]
        return rf, lr, scaler, feature_names
    except FileNotFoundError as e:
        print(f"ERROR: Model files not found. Please run 'analyze_features.py' first to train models.")
        print(f"Missing file: {e.filename}")
        sys.exit(1)

def fetch_live_data(symbol='GC=F', period='1d', interval='5m'):
    """
    Fetch live XAUUSD data from Yahoo Finance
    
    Args:
        symbol: Ticker symbol (GC=F for gold futures, XAUUSD=X also works)
        period: Time period to fetch (1d, 5d, 1mo)
        interval: Candle interval (1m, 5m, 15m, 30m, 1h)
    
    Returns:
        DataFrame with OHLCV data
    """
    if not YFINANCE_AVAILABLE:
        print("ERROR: yfinance not installed. Install it with:")
        print("  pip install yfinance")
        sys.exit(1)
    
    print(f"Fetching live {interval} data for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"ERROR: No data returned for {symbol}")
            print("Trying alternative ticker: XAUUSD=X")
            ticker = yf.Ticker('XAUUSD=X')
            df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print("ERROR: Could not fetch data from either GC=F or XAUUSD=X")
                print("Please provide a CSV file instead.")
                sys.exit(1)
        
        # Rename columns to match our format
        df = df.rename(columns={
            'Open': 'Open',
            'High': 'High', 
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume'
        })
        
        # Keep only OHLCV columns
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        print(f"✓ Fetched {len(df)} bars")
        print(f"  Latest candle: {df.index[-1]}")
        print(f"  Price: {df['Close'].iloc[-1]:.2f}")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to fetch live data: {e}")
        print("Please provide a CSV file instead.")
        sys.exit(1)

def predict_current_bar(df=None, csv_file=None):
    """
    Predict probability for the most recent bar
    
    Args:
        df: DataFrame with OHLCV data (from live fetch)
        csv_file: Path to CSV file with OHLCV data (alternative to df)
    
    Returns:
        Dictionary with prediction results
    """
    # Load models
    print("Loading models...")
    rf, lr, scaler, feature_names = load_models()
    
    # Load data
    if df is None:
        if csv_file is None:
            print("ERROR: Must provide either df or csv_file")
            sys.exit(1)
            
        print(f"\nLoading data from: {csv_file}")
        df = pd.read_csv(csv_file, sep='\t', 
                         names=['Datetime','Open','High','Low','Close','Volume'], 
                         parse_dates=['Datetime'])
        df.set_index('Datetime', inplace=True)
    
    if len(df) < 30:
        print(f"ERROR: Need at least 30 bars of history. Found only {len(df)} bars.")
        sys.exit(1)
    
    # Take last 50 bars to ensure we have enough for feature calculation
    df = df.tail(50)
    
    print(f"\nProcessing {len(df)} bars (using last bar for prediction)")
    print(f"Current bar: {df.index[-1]}")
    print(f"  Open:  {df['Open'].iloc[-1]:.2f}")
    print(f"  High:  {df['High'].iloc[-1]:.2f}")
    print(f"  Low:   {df['Low'].iloc[-1]:.2f}")
    print(f"  Close: {df['Close'].iloc[-1]:.2f}")
    
    # Engineer features
    print("\nCalculating features...")
    df = create_microstructure_features(df)
    df = create_combo_features(df)
    
    # Get the last valid row
    current = df.iloc[-1]
    
    # Prepare features for prediction
    X = current[feature_names].fillna(0).values.reshape(1, -1)
    X_scaled = scaler.transform(X)
    
    # Make predictions
    rf_prob = rf.predict_proba(X)[:, 1][0]
    lr_prob = lr.predict_proba(X_scaled)[:, 1][0]
    
    # Determine signal quality
    signal = "WAIT"
    if rf_prob >= 0.85:
        signal = "STRONG TRADE ⭐⭐⭐"
    elif rf_prob >= 0.80:
        signal = "GOOD TRADE ⭐⭐"
    elif rf_prob >= 0.75:
        signal = "MODERATE TRADE ⭐"
    elif rf_prob >= 0.70:
        signal = "WEAK TRADE"
    
    # Display results
    print("\n" + "="*70)
    print("PREDICTION RESULTS")
    print("="*70)
    print(f"\nRandom Forest Probability:      {rf_prob*100:.1f}%")
    print(f"Logistic Regression Probability: {lr_prob*100:.1f}%")
    print(f"\nSIGNAL: {signal}")
    
    # Show key features
    print("\n" + "-"*70)
    print("KEY FEATURES:")
    print("-"*70)
    key_features = ['vol_ratio', 'imbalance_3', 'consistency_5', 'flow_momentum', 
                    'trend_align', 'vol_expansion', 'big_body', 'close_position']
    for feat in key_features:
        if feat in current.index:
            print(f"  {feat:<25} {current[feat]:>10.4f}")
    
    # Trading guidance
    print("\n" + "="*70)
    print("TRADING GUIDANCE:")
    print("="*70)
    
    if rf_prob >= 0.80:
        print("✅ HIGH PROBABILITY SETUP")
        print(f"   Historical win rate at this level: ~87%")
        print(f"   Entry: Current price {current['Close']:.2f}")
        print(f"   Target: +0.2% = {current['Close'] * 1.002:.2f}")
        print(f"   Stop:   -0.1% = {current['Close'] * 0.999:.2f}")
        print(f"   Risk:Reward = 1:2")
        
        # Suggest direction based on features
        if current['imbalance_3'] > 0 or current['trend_align'] > 0:
            print(f"\n   Suggested Direction: LONG (positive flow/trend)")
        elif current['imbalance_3'] < 0 or current['trend_align'] < 0:
            print(f"\n   Suggested Direction: SHORT (negative flow/trend)")
        else:
            print(f"\n   Direction: UNCLEAR - wait for confirmation")
            
    elif rf_prob >= 0.70:
        print("⚠️  MODERATE PROBABILITY")
        print(f"   Historical win rate: ~65-70%")
        print(f"   Consider smaller position size or wait for stronger signal")
    else:
        print("❌ LOW PROBABILITY - DO NOT TRADE")
        print(f"   Baseline win rate: 48.9%")
        print(f"   Wait for RF probability ≥ 0.70")
    
    print("\n" + "="*70)
    
    return {
        'datetime': df.index[-1],
        'close': current['Close'],
        'rf_prob': rf_prob,
        'lr_prob': lr_prob,
        'signal': signal,
        'features': {feat: current[feat] for feat in key_features if feat in current.index}
    }

if __name__ == "__main__":
    print("="*70)
    print("XAUUSD LIVE TRADING PREDICTOR")
    print("="*70)
    
    # Determine if using live data or CSV file
    use_live = True
    csv_file = None
    
    if len(sys.argv) >= 2:
        arg = sys.argv[1]
        if arg in ['--live', '-l']:
            use_live = True
        elif arg in ['--help', '-h']:
            print("\nUsage:")
            print("  python predict_current.py                    # Fetch live data")
            print("  python predict_current.py --live             # Fetch live data")
            print("  python predict_current.py <csv_file>         # Use CSV file")
            print("\nExamples:")
            print("  python predict_current.py")
            print("  python predict_current.py data/raw/XAUUSD5.csv")
            sys.exit(0)
        else:
            csv_file = arg
            use_live = False
    
    # Fetch or load data
    if use_live:
        if not YFINANCE_AVAILABLE:
            print("\n⚠️  yfinance not installed!")
            print("Install it with: pip install yfinance")
            print("\nOr provide a CSV file: python predict_current.py <csv_file>")
            sys.exit(1)
        
        df = fetch_live_data(symbol='GC=F', period='1d', interval='5m')
        result = predict_current_bar(df=df)
    else:
        if not os.path.exists(csv_file):
            print(f"ERROR: File not found: {csv_file}")
            sys.exit(1)
        result = predict_current_bar(csv_file=csv_file)
    
    print("\n✓ Prediction complete!")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
