# ============================================================
# HIGH COMPLEXITY REGRESSION - ACTUAL LABELS (WITH HYBRID)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

print("="*70)
print("HIGH COMPLEXITY REGRESSION - ACTUAL LABELS")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data_preprocessed")
ARTIFACTS_PATH = Path("artifacts")
LSTM_PATH = ARTIFACTS_PATH / "lstm_autoencoder"
ROUTER_PATH = ARTIFACTS_PATH / "router_with_embeddings"
MODEL_PATH = ARTIFACTS_PATH / "regression_models"
MODEL_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
MAX_SEQUENCE_LENGTH = 50

# ============================================================
# STEP 1: LOAD ARTIFACTS
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOAD ARTIFACTS")
print("="*70)

encoder = load_model(LSTM_PATH / 'lstm_encoder.keras')
print("✓ Loaded LSTM encoder")

with open(LSTM_PATH / 'lstm_artifacts.pkl', 'rb') as f:
    lstm_artifacts = pickle.load(f)
sequence_features = lstm_artifacts['sequence_features']
label_encoders = lstm_artifacts['label_encoders']

with open(ROUTER_PATH / "router_artifacts.pkl", 'rb') as f:
    router_artifacts = pickle.load(f)
scaler_seq = router_artifacts['scaler_seq']
print("✓ Loaded router's sequence scaler")

# ============================================================
# STEP 2: LOAD DATA
# ============================================================
print("\n" + "="*70)
print("STEP 2: LOAD DATA")
print("="*70)

df = pd.read_csv(DATA_PATH / "df_preprocessed.csv")
print(f"Loaded {len(df):,} visits")

df_claims = pd.read_csv(DATA_PATH / "df_claims_classified.csv")
print(f"Loaded {len(df_claims):,} claims")

valid_claims = set(df_claims['CLAIM_ID'].unique())
df = df[df['CLAIM_ID'].isin(valid_claims)]
df = df.sort_values(['CLAIM_ID', 'NO_OF_VISIT']).reset_index(drop=True)

claim_totals = df_claims.set_index('CLAIM_ID')['TOTAL_CLAIM_COST'].to_dict()

# ============================================================
# STEP 3: GET HIGH CLAIMS USING ACTUAL LABELS
# ============================================================
print("\n" + "="*70)
print("STEP 3: GET HIGH CLAIMS (ACTUAL LABELS)")
print("="*70)

# USE ACTUAL COMPLEXITY LABELS - NOT ROUTER!
high_claim_ids = df_claims[df_claims['COMPLEXITY'] == 'HIGH']['CLAIM_ID'].values
print(f"HIGH claims (ACTUAL): {len(high_claim_ids):,}")

high_costs = df_claims[df_claims['COMPLEXITY'] == 'HIGH']['TOTAL_CLAIM_COST']
print(f"\nHIGH Cost Statistics (ACTUAL):")
print(f"  Min:    ${high_costs.min():,.2f}")
print(f"  Max:    ${high_costs.max():,.2f}")
print(f"  Mean:   ${high_costs.mean():,.2f}")
print(f"  Median: ${high_costs.median():,.2f}")

df_high = df[df['CLAIM_ID'].isin(high_claim_ids)].copy()
print(f"HIGH visits: {len(df_high):,}")

# ============================================================
# STEP 4: PREPARE DATA
# ============================================================
print("\n" + "="*70)
print("STEP 4: PREPARE DATA")
print("="*70)

if 'ICD_COUNT' not in df_high.columns:
    if 'unique_icd_codes_count' in df_high.columns:
        df_high['ICD_COUNT'] = df_high['unique_icd_codes_count']
    elif 'has_ICD' in df_high.columns:
        df_high['ICD_COUNT'] = df_high['has_ICD'].astype(int)
    else:
        df_high['ICD_COUNT'] = 0

for col in label_encoders:
    enc_col = f'{col}_ENC'
    if col in df_high.columns and enc_col not in df_high.columns:
        le = label_encoders[col]
        df_high[enc_col] = df_high[col].fillna('UNKNOWN').astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

sequence_features_available = [f for f in sequence_features if f in df_high.columns]
print(f"Sequence features: {len(sequence_features_available)}")

for col in sequence_features_available:
    df_high[col] = df_high[col].fillna(0)

# ============================================================
# STEP 5: CREATE PARTIAL SEQUENCES
# ============================================================
print("\n" + "="*70)
print("STEP 5: CREATE PARTIAL SEQUENCES")
print("="*70)

partial_data = []

for claim_id, group in df_high.groupby('CLAIM_ID'):
    group = group.sort_values('NO_OF_VISIT')
    full_seq = group[sequence_features_available].values
    total_cost = claim_totals.get(claim_id, None)
    
    if total_cost is None:
        continue
    
    n_visits = len(full_seq)
    
    for end_idx in range(1, n_visits + 1):
        partial_seq = full_seq[:end_idx]
        partial_data.append({
            'sequence': partial_seq,
            'total_cost': total_cost
        })

print(f"Created {len(partial_data):,} partial sequence samples")

# ============================================================
# STEP 6: SCALE AND PAD
# ============================================================
print("\n" + "="*70)
print("STEP 6: SCALE AND PAD")
print("="*70)

all_seqs = [d['sequence'] for d in partial_data]
all_visits_flat = np.vstack(all_seqs)

all_visits_scaled = scaler_seq.transform(all_visits_flat)

idx = 0
scaled_seqs = []
for seq in all_seqs:
    seq_len = len(seq)
    scaled_seqs.append(all_visits_scaled[idx:idx+seq_len])
    idx += seq_len

X_padded = pad_sequences(
    scaled_seqs,
    maxlen=MAX_SEQUENCE_LENGTH,
    dtype='float32',
    padding='post',
    truncating='post',
    value=0.0
)

print(f"Padded shape: {X_padded.shape}")

# ============================================================
# STEP 7: GENERATE EMBEDDINGS
# ============================================================
print("\n" + "="*70)
print("STEP 7: GENERATE EMBEDDINGS")
print("="*70)

lstm_input_dim = encoder.input_shape[-1]
our_dim = X_padded.shape[-1]

if lstm_input_dim != our_dim:
    print(f"⚠️ Feature mismatch! Using mean-pooled embeddings")
    embeddings = np.mean(X_padded, axis=1)
else:
    embeddings = encoder.predict(X_padded, verbose=1)

print(f"Embeddings shape: {embeddings.shape}")

y = np.array([d['total_cost'] for d in partial_data])
y_log = np.log1p(y)

print(f"\nTarget stats:")
print(f"  Min:    ${y.min():,.2f}")
print(f"  Max:    ${y.max():,.2f}")
print(f"  Mean:   ${y.mean():,.2f}")
print(f"  Median: ${np.median(y):,.2f}")

# ============================================================
# STEP 8: TRAIN/TEST SPLIT
# ============================================================
print("\n" + "="*70)
print("STEP 8: TRAIN/TEST SPLIT")
print("="*70)

scaler_emb = StandardScaler()
X_scaled = scaler_emb.fit_transform(embeddings)

X_train, X_test, y_train, y_test, y_train_log, y_test_log = train_test_split(
    X_scaled, y, y_log, test_size=0.2, random_state=RANDOM_STATE
)

print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

# ============================================================
# STEP 9: TRAIN MODELS
# ============================================================
print("\n" + "="*70)
print("STEP 9: TRAIN MODELS WITH {}-FOLD CV".format(N_FOLDS))
print("="*70)

models = {
    'XGBoost': xgb.XGBRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0
    ),
    'LightGBM': lgb.LGBMRegressor(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_split=5,
        random_state=RANDOM_STATE, n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingRegressor(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        subsample=0.8, random_state=RANDOM_STATE
    ),
}

cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
results = {}

for name, model in models.items():
    print(f"\n{'-'*40}")
    print(f"Training: {name}")
    print(f"{'-'*40}")
    
    cv_scores = cross_val_score(model, X_scaled, y_log, cv=cv, scoring='r2', n_jobs=-1)
    print(f"  CV R² scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  CV R² mean:   {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_pred = np.clip(y_pred, 0, None)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std(),
        'test_r2': r2,
        'mae': mae,
        'rmse': rmse,
        'y_pred': y_pred
    }
    
    print(f"  Test R²:      {r2:.4f}")
    print(f"  Test MAE:     ${mae:,.2f}")
    print(f"  Test RMSE:    ${rmse:,.2f}")

# ============================================================
# STEP 10: HYBRID ENSEMBLE
# ============================================================
print("\n" + "="*70)
print("STEP 10: HYBRID ENSEMBLE")
print("="*70)

# Top 3 models for hybrid
top_3 = sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True)[:3]
print(f"Hybrid ensemble of: {[t[0] for t in top_3]}")

y_pred_hybrid = np.mean([results[t[0]]['y_pred'] for t in top_3], axis=0)

hybrid_mae = mean_absolute_error(y_test, y_pred_hybrid)
hybrid_rmse = np.sqrt(mean_squared_error(y_test, y_pred_hybrid))
hybrid_r2 = r2_score(y_test, y_pred_hybrid)

results['Hybrid'] = {
    'model': None,
    'cv_r2_mean': np.mean([results[t[0]]['cv_r2_mean'] for t in top_3]),
    'cv_r2_std': np.mean([results[t[0]]['cv_r2_std'] for t in top_3]),
    'test_r2': hybrid_r2,
    'mae': hybrid_mae,
    'rmse': hybrid_rmse,
    'y_pred': y_pred_hybrid,
    'ensemble_models': [t[0] for t in top_3]
}

print(f"  Hybrid R²:    {hybrid_r2:.4f}")
print(f"  Hybrid MAE:   ${hybrid_mae:,.2f}")
print(f"  Hybrid RMSE:  ${hybrid_rmse:,.2f}")

# ============================================================
# STEP 11: RESULTS COMPARISON
# ============================================================
print("\n" + "="*70)
print("STEP 11: RESULTS COMPARISON")
print("="*70)

print(f"\n{'Model':<18} {'CV R²':>12} {'Test R²':>10} {'MAE':>12} {'RMSE':>12}")
print("-" * 70)

for name, res in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
    cv_str = f"{res['cv_r2_mean']:.4f}±{res['cv_r2_std']:.3f}"
    print(f"{name:<18} {cv_str:>12} {res['test_r2']:>10.4f} ${res['mae']:>10,.2f} ${res['rmse']:>10,.2f}")

best_name = max(results, key=lambda x: results[x]['test_r2'])
best_r2 = results[best_name]['test_r2']
print(f"\n✓ BEST MODEL: {best_name} (R² = {best_r2:.4f})")

# ============================================================
# STEP 12: SAVE ARTIFACTS
# ============================================================
print("\n" + "="*70)
print("STEP 12: SAVE ARTIFACTS")
print("="*70)

# If hybrid is best, save the top models
if best_name == 'Hybrid':
    hybrid_models = {}
    for model_name in results['Hybrid']['ensemble_models']:
        hybrid_models[model_name] = results[model_name]['model']
    joblib.dump(hybrid_models, MODEL_PATH / 'high_model.joblib')
    print(f"Saved: {MODEL_PATH / 'high_model.joblib'} (Hybrid: {results['Hybrid']['ensemble_models']})")
else:
    joblib.dump(results[best_name]['model'], MODEL_PATH / 'high_model.joblib')
    print(f"Saved: {MODEL_PATH / 'high_model.joblib'} ({best_name})")

high_artifacts = {
    'best_model_name': best_name,
    'best_r2': results[best_name]['test_r2'],
    'best_mae': results[best_name]['mae'],
    'scaler_emb': scaler_emb,
    'sequence_features': sequence_features_available,
    'n_samples': len(partial_data),
    'n_claims': len(high_claim_ids),
    'cost_stats': {
        'min': y.min(),
        'max': y.max(),
        'mean': y.mean(),
        'median': np.median(y)
    },
    'all_results': {k: {key: val for key, val in v.items() if key not in ['model', 'y_pred']} 
                   for k, v in results.items()},
    'is_hybrid': best_name == 'Hybrid',
    'hybrid_models': results['Hybrid']['ensemble_models'] if best_name == 'Hybrid' else None
}

with open(MODEL_PATH / 'high_artifacts.pkl', 'wb') as f:
    pickle.dump(high_artifacts, f)
print(f"Saved: {MODEL_PATH / 'high_artifacts.pkl'}")

# ============================================================
# STEP 13: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 13: VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'HIGH Regression (ACTUAL Labels)\nBest: {best_name} (R² = {best_r2:.4f})', 
             fontsize=14, fontweight='bold')

# 1. Model Comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
r2_scores = [results[m]['test_r2'] for m in model_names]
colors = ['#e74c3c' if m == best_name else '#3498db' for m in model_names]
bars = ax1.bar(model_names, r2_scores, color=colors)
ax1.set_ylabel('R² Score')
ax1.set_title('Model Comparison - Test R²')
ax1.tick_params(axis='x', rotation=45)
for bar, r2 in zip(bars, r2_scores):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r2:.3f}', ha='center')

# 2. Predicted vs Actual
ax2 = axes[0, 1]
y_pred_best = results[best_name]['y_pred']
ax2.scatter(y_test, y_pred_best, alpha=0.3, s=10, c='#e74c3c')
max_val = np.percentile(np.concatenate([y_test, y_pred_best]), 99)
ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect')
ax2.set_xlabel('Actual ($)')
ax2.set_ylabel('Predicted ($)')
ax2.set_title(f'Predicted vs Actual - {best_name}')
ax2.legend()
ax2.set_xlim(0, max_val)
ax2.set_ylim(0, max_val)

# 3. Residuals
ax3 = axes[1, 0]
residuals = y_test - y_pred_best
ax3.hist(residuals, bins=50, color='#9b59b6', edgecolor='white', alpha=0.7)
ax3.axvline(0, color='red', linestyle='--', label='Zero Error')
ax3.axvline(residuals.mean(), color='blue', linestyle='--', label=f'Mean: ${residuals.mean():,.0f}')
ax3.set_xlabel('Residual ($)')
ax3.set_ylabel('Frequency')
ax3.set_title('Residuals Distribution')
ax3.legend()

# 4. Cost Distribution
ax4 = axes[1, 1]
ax4.hist(y, bins=50, color='#e74c3c', edgecolor='white', alpha=0.7)
ax4.axvline(y.mean(), color='red', linestyle='--', label=f'Mean: ${y.mean():,.0f}')
ax4.axvline(np.median(y), color='blue', linestyle='--', label=f'Median: ${np.median(y):,.0f}')
ax4.set_xlabel('Total Claim Cost ($)')
ax4.set_ylabel('Frequency')
ax4.set_title(f'HIGH Cost Distribution (n={len(y):,})')
ax4.legend()

plt.tight_layout()
plt.savefig(MODEL_PATH / 'high_results_actual.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {MODEL_PATH / 'high_results_actual.png'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("HIGH MODEL COMPLETE (ACTUAL LABELS)")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    HIGH MODEL SUMMARY                             ║
╠══════════════════════════════════════════════════════════════════╣
║  Training Data: ACTUAL HIGH labels (not router)                   ║
║  Claims:        {len(high_claim_ids):>10,}                                       ║
║  Samples:       {len(partial_data):>10,}                                       ║
║                                                                   ║
║  COST RANGE:                                                      ║
║    Min:         ${y.min():>10,.2f}                                      ║
║    Max:         ${y.max():>10,.2f}                                      ║
║    Mean:        ${y.mean():>10,.2f}                                      ║
║    Median:      ${np.median(y):>10,.2f}                                      ║
║                                                                   ║
║  RESULTS:                                                         ║""")

for name, res in sorted(results.items(), key=lambda x: x[1]['test_r2'], reverse=True):
    marker = "✓ BEST" if name == best_name else ""
    print(f"║    {name:<14} R²: {res['test_r2']:.4f}  MAE: ${res['mae']:>8,.2f} {marker:<6} ║")

print(f"""║                                                                   ║
║  BEST: {best_name} (R² = {best_r2:.4f})                                  ║
╚══════════════════════════════════════════════════════════════════╝

FILES SAVED:
  1. {MODEL_PATH / 'high_model.joblib'}
  2. {MODEL_PATH / 'high_artifacts.pkl'}
  3. {MODEL_PATH / 'high_results_actual.png'}

NEXT: Run final_inference.py
""")