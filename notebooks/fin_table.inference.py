# ============================================================
# FINAL INFERENCE - ALL SYNCED (ACTUAL LABELS + OWN SCALERS)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt

print("="*70)
print("FINAL INFERENCE - FULLY SYNCED")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data_preprocessed")
ARTIFACTS_PATH = Path("artifacts")
LSTM_PATH = ARTIFACTS_PATH / "lstm_autoencoder"
ROUTER_PATH = ARTIFACTS_PATH / "router_with_embeddings"
MODEL_PATH = ARTIFACTS_PATH / "regression_models"
OUTPUT_PATH = ARTIFACTS_PATH / "final_results"
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

MAX_SEQUENCE_LENGTH = 50
HIGH_COST_FLAG_THRESHOLD = 10000

# ============================================================
# STEP 1: LOAD ALL MODELS AND ARTIFACTS
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOAD ALL MODELS AND ARTIFACTS")
print("="*70)

# LSTM Encoder
encoder = load_model(LSTM_PATH / 'lstm_encoder.keras')
print("✓ Loaded LSTM encoder")

with open(LSTM_PATH / 'lstm_artifacts.pkl', 'rb') as f:
    lstm_artifacts = pickle.load(f)
sequence_features = lstm_artifacts['sequence_features']
label_encoders = lstm_artifacts['label_encoders']

# Router + its scaler
router_model = joblib.load(ROUTER_PATH / "router_xgb_best.joblib")
with open(ROUTER_PATH / "router_artifacts.pkl", 'rb') as f:
    router_artifacts = pickle.load(f)
scaler_seq = router_artifacts['scaler_seq']  # SHARED sequence scaler
print(f"✓ Loaded Router (Accuracy: {router_artifacts['best_accuracy']*100:.2f}%)")

# LOW model + its own embedding scaler
low_model = joblib.load(MODEL_PATH / 'low_model.joblib')
with open(MODEL_PATH / 'low_artifacts.pkl', 'rb') as f:
    low_artifacts = pickle.load(f)
low_scaler_emb = low_artifacts['scaler_emb']
print(f"✓ Loaded LOW model (R²: {low_artifacts['best_r2']:.4f}, {low_artifacts['best_model_name']})")

# MED model + its own embedding scaler
med_model = joblib.load(MODEL_PATH / 'med_model.joblib')
with open(MODEL_PATH / 'med_artifacts.pkl', 'rb') as f:
    med_artifacts = pickle.load(f)
med_scaler_emb = med_artifacts['scaler_emb']
print(f"✓ Loaded MED model (R²: {med_artifacts['best_r2']:.4f}, {med_artifacts['best_model_name']})")

# HIGH model + its own embedding scaler
high_model = joblib.load(MODEL_PATH / 'high_model.joblib')
with open(MODEL_PATH / 'high_artifacts.pkl', 'rb') as f:
    high_artifacts = pickle.load(f)
high_scaler_emb = high_artifacts['scaler_emb']
print(f"✓ Loaded HIGH model (R²: {high_artifacts['best_r2']:.4f}, {high_artifacts['best_model_name']})")

# Check if any model is hybrid
print(f"\nModel Types:")
print(f"  LOW:  {'Hybrid ' + str(low_artifacts.get('hybrid_models', '')) if low_artifacts.get('is_hybrid') else low_artifacts['best_model_name']}")
print(f"  MED:  {'Hybrid ' + str(med_artifacts.get('hybrid_models', '')) if med_artifacts.get('is_hybrid') else med_artifacts['best_model_name']}")
print(f"  HIGH: {'Hybrid ' + str(high_artifacts.get('hybrid_models', '')) if high_artifacts.get('is_hybrid') else high_artifacts['best_model_name']}")

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
claim_complexity = df_claims.set_index('CLAIM_ID')['COMPLEXITY'].to_dict()

# ============================================================
# STEP 3: PREPARE DATA
# ============================================================
print("\n" + "="*70)
print("STEP 3: PREPARE DATA")
print("="*70)

if 'ICD_COUNT' not in df.columns:
    if 'unique_icd_codes_count' in df.columns:
        df['ICD_COUNT'] = df['unique_icd_codes_count']
    elif 'has_ICD' in df.columns:
        df['ICD_COUNT'] = df['has_ICD'].astype(int)
    else:
        df['ICD_COUNT'] = 0

for col in label_encoders:
    enc_col = f'{col}_ENC'
    if col in df.columns and enc_col not in df.columns:
        le = label_encoders[col]
        df[enc_col] = df[col].fillna('UNKNOWN').astype(str).apply(
            lambda x: le.transform([x])[0] if x in le.classes_ else 0
        )

sequence_features_available = [f for f in sequence_features if f in df.columns]
print(f"Sequence features: {len(sequence_features_available)}")

for col in sequence_features_available:
    df[col] = df[col].fillna(0)

# ============================================================
# STEP 4: CREATE PARTIAL SEQUENCES (SAME AS TRAINING)
# ============================================================
print("\n" + "="*70)
print("STEP 4: CREATE PARTIAL SEQUENCES")
print("="*70)

partial_data = []

for claim_id, group in df.groupby('CLAIM_ID'):
    group = group.sort_values('NO_OF_VISIT')
    full_seq = group[sequence_features_available].values
    total_cost = claim_totals.get(claim_id, None)
    complexity = claim_complexity.get(claim_id, None)
    
    if total_cost is None:
        continue
    
    n_visits = len(full_seq)
    
    for end_idx in range(1, n_visits + 1):
        partial_seq = full_seq[:end_idx]
        current_spent = group['medical_amount'].iloc[:end_idx].sum()
        
        partial_data.append({
            'claim_id': claim_id,
            'sequence': partial_seq,
            'seq_length': end_idx,
            'total_visits': n_visits,
            'current_spent': current_spent,
            'actual_total': total_cost,
            'actual_complexity': complexity
        })

print(f"Created {len(partial_data):,} partial sequence samples")

# ============================================================
# STEP 5: SCALE AND PAD (USING ROUTER'S SCALER)
# ============================================================
print("\n" + "="*70)
print("STEP 5: SCALE AND PAD SEQUENCES")
print("="*70)

all_seqs = [d['sequence'] for d in partial_data]
all_visits_flat = np.vstack(all_seqs)

# Use ROUTER's sequence scaler (same as all training)
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
# STEP 6: GENERATE EMBEDDINGS
# ============================================================
print("\n" + "="*70)
print("STEP 6: GENERATE EMBEDDINGS")
print("="*70)

lstm_input_dim = encoder.input_shape[-1]
our_dim = X_padded.shape[-1]

if lstm_input_dim != our_dim:
    print(f"⚠️ Feature mismatch! Using mean-pooled embeddings")
    embeddings = np.mean(X_padded, axis=1)
else:
    embeddings = encoder.predict(X_padded, verbose=1)

print(f"Embeddings shape: {embeddings.shape}")

# ============================================================
# STEP 7: ROUTE CLAIMS
# ============================================================
print("\n" + "="*70)
print("STEP 7: ROUTE CLAIMS")
print("="*70)

# Scale embeddings for router
X_router_scaled = router_artifacts['scaler'].transform(embeddings)
router_preds = router_model.predict(X_router_scaled)
router_labels = router_artifacts['label_encoder'].inverse_transform(router_preds)

print(f"Router predictions:")
print(pd.Series(router_labels).value_counts())

# ============================================================
# STEP 8: PREDICT USING MODEL-SPECIFIC SCALERS
# ============================================================
print("\n" + "="*70)
print("STEP 8: PREDICT (MODEL-SPECIFIC SCALERS)")
print("="*70)

# Group by route for batch prediction
route_indices = {'LOW': [], 'MED': [], 'HIGH': []}
for i, route in enumerate(router_labels):
    route_indices[route].append(i)

print(f"LOW samples:  {len(route_indices['LOW']):,}")
print(f"MED samples:  {len(route_indices['MED']):,}")
print(f"HIGH samples: {len(route_indices['HIGH']):,}")

# Initialize predictions array
all_preds = np.zeros(len(partial_data))

# Helper function to predict (handles both single model and hybrid)
def predict_with_model(model, X, is_hybrid=False):
    if is_hybrid and isinstance(model, dict):
        # Hybrid: average predictions from all models
        preds = np.zeros(len(X))
        for name, m in model.items():
            preds += m.predict(X)
        preds /= len(model)
        return preds
    else:
        return model.predict(X)

# LOW predictions
if len(route_indices['LOW']) > 0:
    low_idx = np.array(route_indices['LOW'])
    low_emb = embeddings[low_idx]
    low_emb_scaled = low_scaler_emb.transform(low_emb)
    low_pred_log = predict_with_model(low_model, low_emb_scaled, low_artifacts.get('is_hybrid', False))
    low_pred = np.expm1(low_pred_log)
    low_pred = np.clip(low_pred, 0, None)
    all_preds[low_idx] = low_pred
    print(f"✓ LOW predictions complete")

# MED predictions
if len(route_indices['MED']) > 0:
    med_idx = np.array(route_indices['MED'])
    med_emb = embeddings[med_idx]
    med_emb_scaled = med_scaler_emb.transform(med_emb)
    med_pred_log = predict_with_model(med_model, med_emb_scaled, med_artifacts.get('is_hybrid', False))
    med_pred = np.expm1(med_pred_log)
    med_pred = np.clip(med_pred, 0, None)
    all_preds[med_idx] = med_pred
    print(f"✓ MED predictions complete")

# HIGH predictions
if len(route_indices['HIGH']) > 0:
    high_idx = np.array(route_indices['HIGH'])
    high_emb = embeddings[high_idx]
    high_emb_scaled = high_scaler_emb.transform(high_emb)
    high_pred_log = predict_with_model(high_model, high_emb_scaled, high_artifacts.get('is_hybrid', False))
    high_pred = np.expm1(high_pred_log)
    high_pred = np.clip(high_pred, 0, None)
    all_preds[high_idx] = high_pred
    print(f"✓ HIGH predictions complete")

# Build results dataframe
predictions = []
for i, (data, route, pred) in enumerate(zip(partial_data, router_labels, all_preds)):
    high_cost_flag = pred >= HIGH_COST_FLAG_THRESHOLD
    
    predictions.append({
        'CLAIM_ID': data['claim_id'],
        'SEQ_LENGTH': data['seq_length'],
        'TOTAL_VISITS': data['total_visits'],
        'CURRENT_SPENT': data['current_spent'],
        'ACTUAL_TOTAL': data['actual_total'],
        'ACTUAL_COMPLEXITY': data['actual_complexity'],
        'PREDICTED_COMPLEXITY': route,
        'PREDICTED_TOTAL': pred,
        'PREDICTED_REMAINING': max(0, pred - data['current_spent']),
        'ERROR': pred - data['actual_total'],
        'ABS_ERROR': abs(pred - data['actual_total']),
        'HIGH_COST_FLAG': high_cost_flag,
    })

df_results = pd.DataFrame(predictions)
print(f"\nInference complete: {len(df_results):,} samples")

# ============================================================
# STEP 9: CALCULATE METRICS
# ============================================================
print("\n" + "="*70)
print("STEP 9: CALCULATE METRICS")
print("="*70)

overall_r2 = r2_score(df_results['ACTUAL_TOTAL'], df_results['PREDICTED_TOTAL'])
overall_mae = mean_absolute_error(df_results['ACTUAL_TOTAL'], df_results['PREDICTED_TOTAL'])
overall_rmse = np.sqrt(mean_squared_error(df_results['ACTUAL_TOTAL'], df_results['PREDICTED_TOTAL']))

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    OVERALL MODEL PERFORMANCE                      ║
╠══════════════════════════════════════════════════════════════════╣
║  Total Samples:    {len(df_results):>10,}                                  ║
║  Unique Claims:    {df_results['CLAIM_ID'].nunique():>10,}                                  ║
║  Overall R²:       {overall_r2:>10.4f} ({overall_r2*100:.2f}%)                         ║
║  Overall MAE:      ${overall_mae:>10,.2f}                                 ║
║  Overall RMSE:     ${overall_rmse:>10,.2f}                                 ║
╚══════════════════════════════════════════════════════════════════╝
""")

# Cost statistics
print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                       COST STATISTICS                             ║
╠══════════════════════════════════════════════════════════════════╣
║                        ACTUAL              PREDICTED              ║
║  Min:            ${df_results['ACTUAL_TOTAL'].min():>10,.2f}        ${df_results['PREDICTED_TOTAL'].min():>10,.2f}       ║
║  Max:            ${df_results['ACTUAL_TOTAL'].max():>10,.2f}        ${df_results['PREDICTED_TOTAL'].max():>10,.2f}       ║
║  Mean:           ${df_results['ACTUAL_TOTAL'].mean():>10,.2f}        ${df_results['PREDICTED_TOTAL'].mean():>10,.2f}       ║
║  Median:         ${df_results['ACTUAL_TOTAL'].median():>10,.2f}        ${df_results['PREDICTED_TOTAL'].median():>10,.2f}       ║
╚══════════════════════════════════════════════════════════════════╝
""")

# Per-complexity metrics
print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                  PER-COMPLEXITY PERFORMANCE                       ║
╠══════════════════════════════════════════════════════════════════╣
║ Complexity │  Count  │    R²    │     MAE     │  Cost Range       ║
╠────────────┼─────────┼──────────┼─────────────┼───────────────────╣""")

complexity_metrics = {}

for comp in ['LOW', 'MED', 'HIGH']:
    df_comp = df_results[df_results['PREDICTED_COMPLEXITY'] == comp]
    if len(df_comp) > 1:
        r2 = r2_score(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'])
        mae = mean_absolute_error(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'])
        rmse = np.sqrt(mean_squared_error(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL']))
        min_cost = df_comp['ACTUAL_TOTAL'].min()
        max_cost = df_comp['ACTUAL_TOTAL'].max()
        
        complexity_metrics[comp] = {
            'r2': r2, 'mae': mae, 'rmse': rmse,
            'count': len(df_comp), 'min': min_cost, 'max': max_cost
        }
        print(f"║ {comp:<10} │ {len(df_comp):>7,} │ {r2:>8.4f} │ ${mae:>9,.2f} │ ${min_cost:>6,.0f} - ${max_cost:>6,.0f} ║")

print(f"╚══════════════════════════════════════════════════════════════════╝")

# Router accuracy
router_accuracy = (df_results['ACTUAL_COMPLEXITY'] == df_results['PREDICTED_COMPLEXITY']).mean()
print(f"\nRouter Accuracy: {router_accuracy*100:.2f}%")

# ============================================================
# STEP 10: HIGH COST FLAGS
# ============================================================
print("\n" + "="*70)
print("STEP 10: HIGH COST FLAGS (>$10,000)")
print("="*70)

flagged = df_results[df_results['HIGH_COST_FLAG'] == True]
print(f"\nFlagged Samples: {len(flagged):,} ({len(flagged)/len(df_results)*100:.2f}%)")

if len(flagged) > 1:
    print(f"\nFlagged Summary:")
    print(f"  Predicted Range: ${flagged['PREDICTED_TOTAL'].min():,.2f} - ${flagged['PREDICTED_TOTAL'].max():,.2f}")
    print(f"  Actual Range:    ${flagged['ACTUAL_TOTAL'].min():,.2f} - ${flagged['ACTUAL_TOTAL'].max():,.2f}")
    
    flagged_r2 = r2_score(flagged['ACTUAL_TOTAL'], flagged['PREDICTED_TOTAL'])
    flagged_mae = mean_absolute_error(flagged['ACTUAL_TOTAL'], flagged['PREDICTED_TOTAL'])
    print(f"\n  Performance on Flagged:")
    print(f"    R²:  {flagged_r2:.4f}")
    print(f"    MAE: ${flagged_mae:,.2f}")

# ============================================================
# STEP 11: COST BUCKET ANALYSIS
# ============================================================
print("\n" + "="*70)
print("STEP 11: COST BUCKET ANALYSIS")
print("="*70)

buckets = [
    (0, 100, '$0 - $100'),
    (100, 500, '$100 - $500'),
    (500, 1000, '$500 - $1,000'),
    (1000, 5000, '$1,000 - $5,000'),
    (5000, 10000, '$5,000 - $10,000'),
    (10000, float('inf'), '$10,000+')
]

print(f"\n{'Cost Bucket':<20} {'Count':>10} {'R²':>10} {'MAE':>12} {'Avg Error %':>12}")
print("-" * 70)

bucket_metrics = []

for low, high, label in buckets:
    df_bucket = df_results[(df_results['ACTUAL_TOTAL'] >= low) & (df_results['ACTUAL_TOTAL'] < high)]
    if len(df_bucket) > 1:
        r2 = r2_score(df_bucket['ACTUAL_TOTAL'], df_bucket['PREDICTED_TOTAL'])
        mae = mean_absolute_error(df_bucket['ACTUAL_TOTAL'], df_bucket['PREDICTED_TOTAL'])
        avg_pct_error = (df_bucket['ABS_ERROR'] / df_bucket['ACTUAL_TOTAL']).mean() * 100
        
        bucket_metrics.append({
            'bucket': label, 'count': len(df_bucket),
            'r2': r2, 'mae': mae, 'avg_pct_error': avg_pct_error
        })
        print(f"{label:<20} {len(df_bucket):>10,} {r2:>10.4f} ${mae:>10,.2f} {avg_pct_error:>11.1f}%")

# ============================================================
# STEP 12: SAVE RESULTS
# ============================================================
print("\n" + "="*70)
print("STEP 12: SAVE RESULTS")
print("="*70)

df_results.to_csv(OUTPUT_PATH / 'final_predictions.csv', index=False)
print(f"Saved: {OUTPUT_PATH / 'final_predictions.csv'}")

if len(flagged) > 0:
    flagged.to_csv(OUTPUT_PATH / 'flagged_high_cost_claims.csv', index=False)
    print(f"Saved: {OUTPUT_PATH / 'flagged_high_cost_claims.csv'}")

final_metrics = {
    'overall': {
        'r2': overall_r2,
        'mae': overall_mae,
        'rmse': overall_rmse,
        'n_samples': len(df_results),
        'n_claims': df_results['CLAIM_ID'].nunique()
    },
    'per_complexity': complexity_metrics,
    'router_accuracy': router_accuracy,
    'cost_buckets': bucket_metrics,
    'flagged': {
        'count': len(flagged),
        'threshold': HIGH_COST_FLAG_THRESHOLD
    },
    'training_r2': {
        'LOW': low_artifacts['best_r2'],
        'MED': med_artifacts['best_r2'],
        'HIGH': high_artifacts['best_r2']
    }
}

with open(OUTPUT_PATH / 'final_metrics.pkl', 'wb') as f:
    pickle.dump(final_metrics, f)
print(f"Saved: {OUTPUT_PATH / 'final_metrics.pkl'}")

# ============================================================
# STEP 13: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 13: VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle(f'Final Inference Results (Synced Models)\nOverall R² = {overall_r2:.4f} | MAE = ${overall_mae:,.0f}', 
             fontsize=14, fontweight='bold')

colors_map = {'LOW': '#2ecc71', 'MED': '#f39c12', 'HIGH': '#e74c3c'}

# 1. Predicted vs Actual
ax1 = axes[0, 0]
for comp in ['LOW', 'MED', 'HIGH']:
    df_comp = df_results[df_results['PREDICTED_COMPLEXITY'] == comp]
    ax1.scatter(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'],
                alpha=0.2, s=5, c=colors_map[comp], label=comp)
max_val = np.percentile(df_results['ACTUAL_TOTAL'], 99)
ax1.plot([0, max_val], [0, max_val], 'k--', label='Perfect')
ax1.set_xlabel('Actual ($)')
ax1.set_ylabel('Predicted ($)')
ax1.set_title('Predicted vs Actual')
ax1.legend()
ax1.set_xlim(0, max_val)
ax1.set_ylim(0, max_val)

# 2. R² by Complexity (Training vs Inference)
ax2 = axes[0, 1]
comps = ['LOW', 'MED', 'HIGH']
train_r2s = [low_artifacts['best_r2'], med_artifacts['best_r2'], high_artifacts['best_r2']]
inf_r2s = [complexity_metrics[c]['r2'] for c in comps]

x = np.arange(len(comps))
width = 0.35

bars1 = ax2.bar(x - width/2, train_r2s, width, label='Training R²', color='#3498db')
bars2 = ax2.bar(x + width/2, inf_r2s, width, label='Inference R²', color='#e74c3c')

ax2.set_ylabel('R² Score')
ax2.set_title('Training vs Inference R²')
ax2.set_xticks(x)
ax2.set_xticklabels(comps)
ax2.legend()
ax2.axhline(y=overall_r2, color='black', linestyle='--', label=f'Overall: {overall_r2:.3f}')

for bar, r2 in zip(bars1, train_r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r2:.2f}', ha='center', fontsize=9)
for bar, r2 in zip(bars2, inf_r2s):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{r2:.2f}', ha='center', fontsize=9)

# 3. MAE by Complexity
ax3 = axes[0, 2]
maes = [complexity_metrics[c]['mae'] for c in comps]
colors = [colors_map[c] for c in comps]
bars = ax3.bar(comps, maes, color=colors)
ax3.set_ylabel('MAE ($)')
ax3.set_title('MAE by Complexity')
ax3.axhline(y=overall_mae, color='black', linestyle='--', label=f'Overall: ${overall_mae:,.0f}')
ax3.legend()
for bar, mae in zip(bars, maes):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, f'${mae:,.0f}', ha='center')

# 4. Error Distribution
ax4 = axes[1, 0]
ax4.hist(df_results['ERROR'], bins=50, color='#3498db', edgecolor='white', alpha=0.7)
ax4.axvline(0, color='red', linestyle='--', label='Zero Error')
ax4.axvline(df_results['ERROR'].mean(), color='orange', linestyle='--', 
            label=f'Mean: ${df_results["ERROR"].mean():,.0f}')
ax4.set_xlabel('Error ($)')
ax4.set_ylabel('Frequency')
ax4.set_title('Error Distribution')
ax4.legend()

# 5. R² by Cost Bucket
ax5 = axes[1, 1]
if bucket_metrics:
    labels = [b['bucket'] for b in bucket_metrics]
    r2s_bucket = [b['r2'] for b in bucket_metrics]
    colors_bucket = ['#e74c3c' if r2 < 0 else '#2ecc71' for r2 in r2s_bucket]
    ax5.barh(labels, r2s_bucket, color=colors_bucket)
    ax5.set_xlabel('R² Score')
    ax5.set_title('R² by Cost Bucket')
    ax5.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax5.axvline(x=overall_r2, color='blue', linestyle='--', label=f'Overall: {overall_r2:.3f}')
    ax5.legend()

# 6. Distribution by Complexity
ax6 = axes[1, 2]
counts = [complexity_metrics[c]['count'] for c in comps]
ax6.pie(counts, labels=comps, colors=colors, autopct='%1.1f%%')
ax6.set_title('Samples by Predicted Complexity')

plt.tight_layout()
plt.savefig(OUTPUT_PATH / 'final_results_synced.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUTPUT_PATH / 'final_results_synced.png'}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    FINAL MODEL SUMMARY                            ║
╠══════════════════════════════════════════════════════════════════╣
║  DATASET:                                                         ║
║    Total Samples:       {len(df_results):>10,}                              ║
║    Unique Claims:       {df_results['CLAIM_ID'].nunique():>10,}                              ║
║                                                                   ║
║  OVERALL PERFORMANCE:                                             ║
║    R² Score:            {overall_r2:>10.4f} ({overall_r2*100:.2f}%)                     ║
║    MAE:                 ${overall_mae:>10,.2f}                             ║
║    RMSE:                ${overall_rmse:>10,.2f}                             ║
║                                                                   ║
║  ROUTER ACCURACY:       {router_accuracy*100:>10.2f}%                             ║
║                                                                   ║
║  TRAINING vs INFERENCE R²:                                        ║
║    LOW:   Train={low_artifacts['best_r2']:.4f}  Infer={complexity_metrics['LOW']['r2']:.4f}              ║
║    MED:   Train={med_artifacts['best_r2']:.4f}  Infer={complexity_metrics['MED']['r2']:.4f}              ║
║    HIGH:  Train={high_artifacts['best_r2']:.4f}  Infer={complexity_metrics['HIGH']['r2']:.4f}              ║
║                                                                   ║
║  COST RANGE:                                                      ║
║    Actual:    ${df_results['ACTUAL_TOTAL'].min():>8,.2f} - ${df_results['ACTUAL_TOTAL'].max():>10,.2f}              ║
║    Predicted: ${df_results['PREDICTED_TOTAL'].min():>8,.2f} - ${df_results['PREDICTED_TOTAL'].max():>10,.2f}              ║
║                                                                   ║
║  HIGH COST FLAGS (>${HIGH_COST_FLAG_THRESHOLD:,}):                                       ║
║    Count: {len(flagged):>10,} ({len(flagged)/len(df_results)*100:.1f}%)                               ║
╚══════════════════════════════════════════════════════════════════╝

FILES SAVED:
  1. {OUTPUT_PATH / 'final_predictions.csv'}
  2. {OUTPUT_PATH / 'final_metrics.pkl'}
  3. {OUTPUT_PATH / 'final_results_synced.png'}
  4. {OUTPUT_PATH / 'flagged_high_cost_claims.csv'}

SYSTEM STATUS: ✅ FULLY SYNCED
""")