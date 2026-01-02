"""
DIRECTION PREDICTION MODEL TRAINING
====================================
Trains ML models to predict direction (LONG vs SHORT) for XAUUSD 1-minute bars.
Focuses on high-confidence predictions only using heavy regularization and
proper walk-forward validation.

Key strategies:
1. Use gradient boosting with DART (dropout) to prevent overfitting
2. Heavy regularization (small trees, L1/L2)
3. Walk-forward validation (TimeSeriesSplit)
4. Only trade high-confidence predictions (probability thresholding)
5. Focus on LONG-only strategy (shown to work better in analysis)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

import joblib
import os

# Try to import gradient boosting libraries
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("LightGBM not installed. Run: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Run: pip install xgboost")

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("CatBoost not installed. Run: pip install catboost")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression


def load_data():
    """Load the feature analysis CSV with all engineered features."""
    print("Loading data...")
    df = pd.read_csv('data/processed/XAUUSD1_feature_analysis.csv', parse_dates=['Datetime'])
    df.set_index('Datetime', inplace=True)
    print(f"Loaded {len(df)} bars")
    return df


def prepare_direction_data(df):
    """
    Prepare data for direction prediction.
    Only use bars where movement was detected (outcome 1 or 2).
    Target: 1 = LONG won, 0 = SHORT won
    """
    # Filter to only movement bars
    movement_df = df[df['outcome'] != 0].copy()
    print(f"\nMovement bars: {len(movement_df)} ({len(movement_df)/len(df)*100:.1f}% of total)")
    
    # Create direction target: 1 = LONG, 0 = SHORT
    movement_df['direction'] = (movement_df['outcome'] == 1).astype(int)
    
    long_count = movement_df['direction'].sum()
    short_count = len(movement_df) - long_count
    print(f"LONG wins: {long_count} ({long_count/len(movement_df)*100:.1f}%)")
    print(f"SHORT wins: {short_count} ({short_count/len(movement_df)*100:.1f}%)")
    
    return movement_df


def get_feature_columns(df):
    """Get the feature columns to use for training."""
    # Core features that showed some predictive value
    base_features = [
        # Order flow features
        'flow_3', 'flow_5', 'flow_10', 'flow_momentum',
        'imbalance_3', 'imbalance_5',
        
        # Momentum consistency
        'consistency_3', 'consistency_5', 'up_count_3', 'up_count_5',
        
        # Volatility
        'vol_ratio', 'vol_expansion', 'vol_contraction',
        'atr_3', 'atr_10',
        
        # Trend
        'trend_align', 'dist_ema8',
        
        # Position/Range
        'close_position', 'body_pct',
        'at_high', 'at_low',
        
        # Rejection patterns
        'upper_reject', 'lower_reject',
        
        # Size patterns
        'big_body', 'small_body',
        
        # Combination features
        'combo_flow_trend', 'combo_vol_imbalance',
        'combo_consistency_position', 'combo_body_reject',
        'combo_imbalance_momentum', 'combo_position_consistency',
    ]
    
    # Filter to features that exist in dataframe
    features = [f for f in base_features if f in df.columns]
    print(f"\nUsing {len(features)} features")
    return features


def create_lightgbm_model():
    """Create LightGBM with DART boosting and heavy regularization."""
    if not HAS_LIGHTGBM:
        return None
    
    return lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=3,  # Very shallow trees
        num_leaves=8,  # Small number of leaves
        learning_rate=0.02,
        boosting_type='dart',  # Dropout to prevent overfitting
        drop_rate=0.15,  # 15% dropout
        min_data_in_leaf=50,  # Require many samples per leaf
        feature_fraction=0.7,  # Use 70% of features per tree
        bagging_fraction=0.7,  # Use 70% of data per iteration
        bagging_freq=5,
        reg_alpha=0.5,  # L1 regularization
        reg_lambda=0.5,  # L2 regularization
        is_unbalance=True,  # Handle class imbalance
        verbose=-1,
        random_state=42
    )


def create_xgboost_model(scale_pos_weight=1.0):
    """Create XGBoost with heavy regularization."""
    if not HAS_XGBOOST:
        return None
    
    return xgb.XGBClassifier(
        n_estimators=150,
        max_depth=3,  # Very shallow
        learning_rate=0.03,
        min_child_weight=10,  # Require many samples
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=0.5,  # L1
        reg_lambda=0.5,  # L2
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )


def create_catboost_model():
    """Create CatBoost with regularization."""
    if not HAS_CATBOOST:
        return None
    
    return CatBoostClassifier(
        iterations=150,
        depth=3,
        learning_rate=0.03,
        l2_leaf_reg=5,  # Strong L2
        min_data_in_leaf=50,
        auto_class_weights='Balanced',
        verbose=False,
        random_state=42
    )


def create_rf_model():
    """Create Random Forest with constraints."""
    return RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        min_samples_leaf=50,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )


def create_logistic_model():
    """Create regularized logistic regression."""
    return LogisticRegression(
        C=0.1,  # Strong regularization (smaller C = more regularization)
        penalty='l2',
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    )


def walk_forward_validation(X, y, model_fn, n_splits=5, threshold=0.55):
    """
    Perform walk-forward validation for time series.
    Returns accuracy at different probability thresholds.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    all_probs = []
    all_true = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model = model_fn()
        if model is None:
            return None
        
        model.fit(X_train_scaled, y_train)
        
        # Get probabilities
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        all_probs.extend(probs)
        all_true.extend(y_test.values)
        
        # Evaluate at different thresholds
        for thresh in [0.50, 0.55, 0.60, 0.65]:
            # LONG-only: only predict LONG when prob > threshold
            preds = (probs > thresh).astype(int)
            n_signals = preds.sum()
            if n_signals > 0:
                accuracy = (preds == y_test).sum() / n_signals if n_signals > 0 else 0
                correct = ((preds == 1) & (y_test == 1)).sum()
                results.append({
                    'fold': fold,
                    'threshold': thresh,
                    'signals': n_signals,
                    'correct': correct,
                    'accuracy': accuracy
                })
    
    return pd.DataFrame(results), np.array(all_probs), np.array(all_true)


def evaluate_all_models(X, y):
    """Evaluate all available models using walk-forward validation."""
    
    models = {
        'LogisticRegression': create_logistic_model,
        'RandomForest': create_rf_model,
    }
    
    if HAS_LIGHTGBM:
        models['LightGBM_DART'] = create_lightgbm_model
    
    if HAS_XGBOOST:
        # Calculate scale_pos_weight for class imbalance
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        models['XGBoost'] = lambda: create_xgboost_model(scale_weight)
    
    if HAS_CATBOOST:
        models['CatBoost'] = create_catboost_model
    
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION RESULTS")
    print("="*80)
    
    best_model = None
    best_accuracy = 0
    best_threshold = 0.55
    
    for name, model_fn in models.items():
        print(f"\n--- {name} ---")
        
        result = walk_forward_validation(X, y, model_fn, n_splits=5)
        if result is None:
            print("Model not available")
            continue
        
        results_df, all_probs, all_true = result
        
        # Aggregate by threshold
        summary = results_df.groupby('threshold').agg({
            'signals': 'sum',
            'correct': 'sum'
        }).reset_index()
        summary['accuracy'] = summary['correct'] / summary['signals']
        
        print(f"\n{'Threshold':<12} {'Signals':<10} {'Accuracy':<10}")
        print("-"*35)
        for _, row in summary.iterrows():
            print(f"{row['threshold']:<12.2f} {int(row['signals']):<10} {row['accuracy']*100:<10.1f}%")
        
        # Track best model
        for _, row in summary.iterrows():
            if row['signals'] >= 50 and row['accuracy'] > best_accuracy:
                best_accuracy = row['accuracy']
                best_model = name
                best_threshold = row['threshold']
    
    print(f"\n{'='*80}")
    print(f"BEST MODEL: {best_model} at threshold {best_threshold:.2f}")
    print(f"Accuracy: {best_accuracy*100:.1f}%")
    print(f"{'='*80}")
    
    return best_model, best_threshold


def train_final_model(X, y, model_type='LightGBM_DART'):
    """Train final model on all data and save."""
    
    print(f"\n{'='*80}")
    print(f"TRAINING FINAL {model_type} MODEL")
    print(f"{'='*80}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Create model
    if model_type == 'LightGBM_DART' and HAS_LIGHTGBM:
        model = create_lightgbm_model()
    elif model_type == 'XGBoost' and HAS_XGBOOST:
        neg_count = (y == 0).sum()
        pos_count = (y == 1).sum()
        scale_weight = neg_count / pos_count if pos_count > 0 else 1.0
        model = create_xgboost_model(scale_weight)
    elif model_type == 'CatBoost' and HAS_CATBOOST:
        model = create_catboost_model()
    elif model_type == 'RandomForest':
        model = create_rf_model()
    else:
        model = create_logistic_model()
    
    # Train
    model.fit(X_scaled, y)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Important Features:")
        print(importance.head(10).to_string(index=False))
    
    # Save model and scaler
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/direction_model.pkl')
    joblib.dump(scaler, 'models/direction_scaler.pkl')
    
    # Save feature list
    with open('models/direction_features.txt', 'w') as f:
        for feat in X.columns:
            f.write(feat + '\n')
    
    print(f"\nModel saved to models/direction_model.pkl")
    
    return model, scaler


def test_long_only_strategy(X, y, model, scaler, thresholds=[0.50, 0.55, 0.60, 0.65, 0.70]):
    """
    Test LONG-only strategy at different thresholds.
    Only predict LONG when model probability exceeds threshold.
    """
    print("\n" + "="*80)
    print("LONG-ONLY STRATEGY TEST")
    print("="*80)
    
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    print(f"\n{'Threshold':<12} {'Signals':<10} {'Correct':<10} {'Accuracy':<10} {'% of Data'}")
    print("-"*60)
    
    for thresh in thresholds:
        signals = probs > thresh
        n_signals = signals.sum()
        if n_signals > 0:
            correct = (y[signals] == 1).sum()
            accuracy = correct / n_signals
            pct_data = n_signals / len(y) * 100
            print(f"{thresh:<12.2f} {n_signals:<10} {correct:<10} {accuracy*100:<10.1f}% {pct_data:.1f}%")
    
    return probs


def test_combined_strategy(df, features, model, scaler):
    """
    Test combined strategy: Movement model (rf_prob) + Direction model.
    This simulates real usage where we first filter by movement probability.
    """
    print("\n" + "="*80)
    print("COMBINED STRATEGY TEST (Movement + Direction)")
    print("="*80)
    
    # Only look at movement bars
    movement_df = df[df['outcome'] != 0].copy()
    direction = (movement_df['outcome'] == 1).astype(int)
    
    # Get direction predictions
    X_scaled = scaler.transform(movement_df[features])
    dir_probs = model.predict_proba(X_scaled)[:, 1]
    movement_df['dir_prob'] = dir_probs
    movement_df['direction'] = direction.values
    
    print("\nFiltering by RF Movement Probability first, then Direction probability:")
    print(f"\n{'RF_Thresh':<12} {'Dir_Thresh':<12} {'Signals':<10} {'Correct':<10} {'Accuracy':<10}")
    print("-"*70)
    
    best_accuracy = 0
    best_config = {}
    
    for rf_thresh in [0.70, 0.75, 0.80, 0.85, 0.90]:
        # First filter by movement probability
        high_rf = movement_df[movement_df['rf_prob'] >= rf_thresh].copy()
        if len(high_rf) < 10:
            continue
        
        for dir_thresh in [0.50, 0.52, 0.54, 0.55, 0.56, 0.58, 0.60]:
            # Then filter by direction probability (LONG only)
            signals = high_rf[high_rf['dir_prob'] > dir_thresh]
            n_signals = len(signals)
            
            if n_signals >= 10:
                correct = (signals['direction'] == 1).sum()
                accuracy = correct / n_signals
                print(f"{rf_thresh:<12.2f} {dir_thresh:<12.2f} {n_signals:<10} {correct:<10} {accuracy*100:<10.1f}%")
                
                if accuracy > best_accuracy and n_signals >= 20:
                    best_accuracy = accuracy
                    best_config = {
                        'rf_thresh': rf_thresh,
                        'dir_thresh': dir_thresh,
                        'signals': n_signals,
                        'accuracy': accuracy
                    }
    
    if best_config:
        print(f"\n{'='*70}")
        print(f"BEST COMBINED STRATEGY:")
        print(f"  RF Movement >= {best_config['rf_thresh']:.2f}")
        print(f"  Direction > {best_config['dir_thresh']:.2f}")
        print(f"  Signals: {best_config['signals']}")
        print(f"  Accuracy: {best_config['accuracy']*100:.1f}%")
        print(f"{'='*70}")
    
    return best_config


def walk_forward_combined_test(df, features, model_fn, n_splits=5):
    """
    Walk-forward test of the combined strategy.
    Train direction model on each fold separately.
    """
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION OF COMBINED STRATEGY")
    print("="*80)
    
    # Only movement bars
    movement_df = df[df['outcome'] != 0].copy()
    movement_df['direction'] = (movement_df['outcome'] == 1).astype(int)
    
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(movement_df)):
        train_df = movement_df.iloc[train_idx]
        test_df = movement_df.iloc[test_idx].copy()
        
        X_train = train_df[features]
        y_train = train_df['direction']
        X_test = test_df[features]
        y_test = test_df['direction']
        
        # Train direction model
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = model_fn()
        model.fit(X_train_scaled, y_train)
        
        # Get direction probabilities
        dir_probs = model.predict_proba(X_test_scaled)[:, 1]
        test_df['dir_prob'] = dir_probs
        
        # Test combined strategy
        for rf_thresh in [0.70, 0.75, 0.80, 0.85]:
            for dir_thresh in [0.50, 0.52, 0.55, 0.58]:
                # Filter by RF prob then Direction prob
                signals = test_df[(test_df['rf_prob'] >= rf_thresh) & (test_df['dir_prob'] > dir_thresh)]
                n_signals = len(signals)
                
                if n_signals > 0:
                    correct = (signals['direction'] == 1).sum()
                    accuracy = correct / n_signals
                    results.append({
                        'fold': fold,
                        'rf_thresh': rf_thresh,
                        'dir_thresh': dir_thresh,
                        'signals': n_signals,
                        'correct': correct,
                        'accuracy': accuracy
                    })
    
    results_df = pd.DataFrame(results)
    
    # Aggregate by thresholds
    summary = results_df.groupby(['rf_thresh', 'dir_thresh']).agg({
        'signals': 'sum',
        'correct': 'sum'
    }).reset_index()
    summary['accuracy'] = summary['correct'] / summary['signals']
    summary = summary[summary['signals'] >= 20]  # Minimum signals
    summary = summary.sort_values('accuracy', ascending=False)
    
    print(f"\n{'RF_Thresh':<12} {'Dir_Thresh':<12} {'Total_Signals':<15} {'Accuracy':<10}")
    print("-"*60)
    for _, row in summary.head(10).iterrows():
        print(f"{row['rf_thresh']:<12.2f} {row['dir_thresh']:<12.2f} {int(row['signals']):<15} {row['accuracy']*100:<10.1f}%")
    
    if len(summary) > 0:
        best = summary.iloc[0]
        print(f"\n{'='*60}")
        print(f"BEST WALK-FORWARD COMBINED STRATEGY:")
        print(f"  RF >= {best['rf_thresh']:.2f}, Direction > {best['dir_thresh']:.2f}")
        print(f"  Total Signals: {int(best['signals'])}")
        print(f"  Accuracy: {best['accuracy']*100:.1f}%")
        print(f"{'='*60}")
        return best
    
    return None



def main():
    print("="*80)
    print("DIRECTION PREDICTION MODEL TRAINING")
    print("="*80)
    
    # Load and prepare data
    df = load_data()
    movement_df = prepare_direction_data(df)
    
    # Get features
    features = get_feature_columns(movement_df)
    X = movement_df[features]
    y = movement_df['direction']
    
    # Evaluate all models
    best_model, best_threshold = evaluate_all_models(X, y)
    
    # Train final model
    model, scaler = train_final_model(X, y, model_type=best_model)
    
    # Test LONG-only strategy
    probs = test_long_only_strategy(X, y, model, scaler)
    
    # Test combined strategy with movement model
    best_config = test_combined_strategy(df, features, model, scaler)
    
    # Walk-forward validation of combined strategy
    wf_best = walk_forward_combined_test(df, features, create_logistic_model, n_splits=5)
    
    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"""
Direction Prediction Model Trained!

Model Type: {best_model}
Features: {len(features)}
Training Samples: {len(y)} (movement bars only)

WALK-FORWARD VALIDATED STRATEGY:
{'  RF >= ' + str(wf_best['rf_thresh']) + ', Direction > ' + str(wf_best['dir_thresh']) if wf_best is not None else '  No reliable strategy found'}
{'  Accuracy: ' + str(round(wf_best['accuracy']*100, 1)) + '%' if wf_best is not None else ''}
{'  Signals: ' + str(int(wf_best['signals'])) if wf_best is not None else ''}

RECOMMENDED USAGE:
1. First check movement model (RF prob >= 0.75)
2. Then check direction model probability > 0.52
3. Only take LONG trades meeting both criteria
4. Expect ~55-60% accuracy on directional trades

Files saved:
- models/direction_model.pkl
- models/direction_scaler.pkl
- models/direction_features.txt
    """)



if __name__ == '__main__':
    main()
