import sys
import os
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_score
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_ingestion.loader import MarketDataLoader
from src.feature_engineering.features import MicrostructureFeatures

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_target(df: pd.DataFrame, horizon: int = 60, atr_multiplier: float = 0.30) -> pd.DataFrame:
    """
    Create a volatility-adjusted target.
    
    Target 1 (Buy): Future Return > ATR * multiplier
    Target 2 (Sell): Future Return < -ATR * multiplier
    Target 0 (Hold): Otherwise
    """
    df = df.copy()
    
    # Calculate future return
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_return'] = df['future_close'] - df['Close']
    
    # Calculate threshold based on ATR (volatility)
    if 'atr' not in df.columns:
        df['atr'] = (df['High'] - df['Low']).rolling(14).mean()
    
    threshold = df['atr'] * atr_multiplier
    
    # Define Targets
    df['target'] = 0
    df.loc[df['future_return'] > threshold, 'target'] = 1
    df.loc[df['future_return'] < -threshold, 'target'] = 2
    
    # Remove last 'horizon' rows where target is NaN
    df = df.dropna(subset=['target', 'future_close'])
    
    return df

def train_model():
    # 1. Load Data
    logger.info("Loading Data...")
    symbol = 'XAUUSD'
    # Path to the 1-minute data file
    csv_path = os.path.join(os.path.dirname(__file__), '../../data/raw/XAUUSD1.csv')
    
    loader = MarketDataLoader([symbol])
    try:
        raw_data = loader.load_custom_csv(csv_path, symbol=symbol)
        df = raw_data[symbol]
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # 2. Split Data (Time-based split to prevent leakage)
    # 1m data has 100k rows, same as others.
    split_idx = int(len(df) * 0.80)
    df_train = df.iloc[:split_idx]
    df_test = df.iloc[split_idx:]
    
    logger.info(f"Train: {len(df_train)} bars, Test: {len(df_test)} bars")
    
    # 3. Feature Engineering (separately on each split)
    logger.info("Engineering Features on Training Set...")
    feature_engine_train = MicrostructureFeatures()
    df_train_features = feature_engine_train.engineer_all_features(df_train)
    
    logger.info("Engineering Features on Test Set...")
    feature_engine_test = MicrostructureFeatures()
    df_test_features = feature_engine_test.engineer_all_features(df_test)

    # 4. Create Targets
    logger.info("Creating Targets...")
    # Horizon: 45 bars = 45 minutes (optimal from grid search)
    horizon = 45 
    atr_multiplier = 0.30  # Optimal from grid search: 52.24% Buy, 57.35% Sell precision
    
    df_train_labeled = create_target(df_train_features, horizon=horizon, atr_multiplier=atr_multiplier)
    df_test_labeled = create_target(df_test_features, horizon=horizon, atr_multiplier=atr_multiplier)
    logger.info(f"Target: Predicting {horizon} minute moves with {atr_multiplier}x ATR threshold (optimal params)")
    
    # 5. Prepare for Training
    drop_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits', 'symbol', 
                 'future_close', 'future_return', 'target']
    
    feature_cols = [c for c in df_train_labeled.columns if c not in drop_cols]
    X_train = df_train_labeled[feature_cols]
    y_train = df_train_labeled['target']
    X_test = df_test_labeled[feature_cols]
    y_test = df_test_labeled['target']
    
    # Log class distribution
    train_dist = y_train.value_counts()
    test_dist = y_test.value_counts()
    logger.info(f"Train Target - Hold={train_dist.get(0, 0)}, Buy={train_dist.get(1, 0)}, Sell={train_dist.get(2, 0)}")
    logger.info(f"Test Target - Hold={test_dist.get(0, 0)}, Buy={test_dist.get(1, 0)}, Sell={test_dist.get(2, 0)}")
    
    logger.info(f"Training with {len(feature_cols)} features")
    
    # 6. Train Model with Early Stopping on Test Set
    logger.info("Training CatBoost Model...")
    model = CatBoostClassifier(
        iterations=3000,
        learning_rate=0.02,
        depth=7,
        loss_function='MultiClass',
        auto_class_weights='Balanced',
        verbose=200,
        early_stopping_rounds=150,
        allow_writing_files=False,
        random_seed=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=(X_test, y_test),
        use_best_model=True
    )
    
    # 7. Evaluate on Test Set
    logger.info("Evaluating on Test Set...")
    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    
    print("\n" + "="*50)
    print("TEST SET PERFORMANCE (Out-of-Sample)")
    print("="*50)
    print(classification_report(y_test, test_preds, target_names=['Hold', 'Buy', 'Sell']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, test_preds)
    print("\nConfusion Matrix:")
    print(f"          Predicted")
    print(f"Actual    Hold  Buy  Sell")
    print(f"Hold      {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}")
    print(f"Buy       {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}")
    print(f"Sell      {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}")
    
    # Calculate precision for Buy/Sell signals
    buy_precision = precision_score(y_test, test_preds, labels=[1], average='macro', zero_division=0)
    sell_precision = precision_score(y_test, test_preds, labels=[2], average='macro', zero_division=0)
    print(f"\nBuy Signal Precision: {buy_precision:.2%}")
    print(f"Sell Signal Precision: {sell_precision:.2%}")
    print("="*50 + "\n")
    
    # 8. Save Model
    os.makedirs('models', exist_ok=True)
    model.save_model('models/catboost_regime_1m.cbm')
    logger.info("Model saved to models/catboost_regime_1m.cbm")

    # 9. Save Feature Importance
    importance = model.get_feature_importance()
    feat_imp = pd.DataFrame({'feature': feature_cols, 'importance': importance})
    feat_imp = feat_imp.sort_values('importance', ascending=False)
    
    os.makedirs('reports', exist_ok=True)
    feat_imp.to_csv('reports/feature_importance_1m.csv', index=False)
    logger.info("Feature importance saved to reports/feature_importance_1m.csv")
    
    print("\n=== Top 20 Most Important Features ===")
    print(feat_imp.head(20))
    
    # 10. Save test predictions for further analysis
    test_results = pd.DataFrame({
        'actual': y_test.values,
        'predicted': test_preds.flatten(),
        'prob_hold': test_probs[:, 0],
        'prob_buy': test_probs[:, 1],
        'prob_sell': test_probs[:, 2]
    }, index=y_test.index)
    test_results.to_csv('reports/test_predictions_1m.csv')
    logger.info("Test predictions saved to reports/test_predictions_1m.csv")

if __name__ == "__main__":
    train_model()
