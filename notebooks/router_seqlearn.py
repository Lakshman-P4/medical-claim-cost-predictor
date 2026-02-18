# ============================================================
# ROUTER TRAINING - PARTIAL SEQUENCES
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import xgboost as xgb

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ROUTER TRAINING - PARTIAL SEQUENCES")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data_preprocessed")
ARTIFACTS_PATH = Path("artifacts")
LSTM_PATH = ARTIFACTS_PATH / "lstm_autoencoder"
ROUTER_PATH = ARTIFACTS_PATH / "router_with_embeddings"
ROUTER_PATH.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_FOLDS = 5
MAX_SEQUENCE_LENGTH = 50

# ============================================================
# STEP 1: LOAD LSTM ENCODER
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOAD LSTM ENCODER")
print("="*70)

encoder = load_model(LSTM_PATH / 'lstm_encoder.keras')
print("✓ Loaded LSTM encoder")

with open(LSTM_PATH / 'lstm_artifacts.pkl', 'rb') as f:
    lstm_artifacts = pickle.load(f)
sequence_features = lstm_artifacts['sequence_features']
label_encoders = lstm_artifacts['label_encoders']
print(f"Sequence features: {len(sequence_features)}")

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

# Get ACTUAL complexity labels
claim_complexity = df_claims.set_index('CLAIM_ID')['COMPLEXITY'].to_dict()

print(f"\nActual Complexity Distribution:")
print(df_claims['COMPLEXITY'].value_counts())

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
print(f"Sequence features available: {len(sequence_features_available)}")

for col in sequence_features_available:
    df[col] = df[col].fillna(0)

# ============================================================
# STEP 4: CREATE PARTIAL SEQUENCES
# ============================================================
print("\n" + "="*70)
print("STEP 4: CREATE PARTIAL SEQUENCES")
print("="*70)

partial_data = []

for claim_id, group in df.groupby('CLAIM_ID'):
    group = group.sort_values('NO_OF_VISIT')
    full_seq = group[sequence_features_available].values
    complexity = claim_complexity.get(claim_id, None)
    
    if complexity is None:
        continue
    
    n_visits = len(full_seq)
    
    # Create partial sequences
    for end_idx in range(1, n_visits + 1):
        partial_seq = full_seq[:end_idx]
        
        partial_data.append({
            'claim_id': claim_id,
            'sequence': partial_seq,
            'seq_length': end_idx,
            'complexity': complexity  # ACTUAL complexity label
        })

print(f"Created {len(partial_data):,} partial sequence samples")

# Check distribution
complexity_counts = pd.Series([d['complexity'] for d in partial_data]).value_counts()
print(f"\nPartial Sequence Complexity Distribution:")
print(complexity_counts)

# ============================================================
# STEP 5: SCALE AND PAD
# ============================================================
print("\n" + "="*70)
print("STEP 5: SCALE AND PAD SEQUENCES")
print("="*70)

all_seqs = [d['sequence'] for d in partial_data]
all_visits_flat = np.vstack(all_seqs)

scaler_seq = StandardScaler()
all_visits_scaled = scaler_seq.fit_transform(all_visits_flat)

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
# STEP 7: PREPARE FOR TRAINING
# ============================================================
print("\n" + "="*70)
print("STEP 7: PREPARE FOR TRAINING")
print("="*70)

# Scale embeddings
scaler_emb = StandardScaler()
X = scaler_emb.fit_transform(embeddings)

# Encode labels
y_raw = np.array([d['complexity'] for d in partial_data])
le_complexity = LabelEncoder()
y = le_complexity.fit_transform(y_raw)

print(f"Features: {X.shape}")
print(f"Classes: {le_complexity.classes_}")
print(f"Class distribution: {np.bincount(y)}")

# ============================================================
# STEP 8: TRAIN ROUTER
# ============================================================
print("\n" + "="*70)
print("STEP 8: TRAIN ROUTER (XGBoost)")
print("="*70)

# Good params from before
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbosity=1,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

# Cross-validation
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

print("Running cross-validation...")
y_pred_cv = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

accuracy = accuracy_score(y, y_pred_cv)
print(f"\nCross-Validation Accuracy: {accuracy*100:.2f}%")

# Classification report
print("\nClassification Report:")
print(classification_report(y, y_pred_cv, target_names=le_complexity.classes_))

# Confusion matrix
cm = confusion_matrix(y, y_pred_cv)
print("\nConfusion Matrix:")
print(cm)

# Per-class accuracy
print("\nPer-Class Accuracy:")
for i, cls in enumerate(le_complexity.classes_):
    class_acc = cm[i, i] / cm[i].sum()
    print(f"  {cls}: {class_acc*100:.2f}%")

# ============================================================
# STEP 9: TRAIN FINAL MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 9: TRAIN FINAL MODEL")
print("="*70)

model.fit(X, y)
print(f"Final model trained on {len(X):,} samples")

# ============================================================
# STEP 10: SAVE ARTIFACTS
# ============================================================
print("\n" + "="*70)
print("STEP 10: SAVE ARTIFACTS")
print("="*70)

joblib.dump(model, ROUTER_PATH / 'router_xgb_best.joblib')
print(f"Saved: {ROUTER_PATH / 'router_xgb_best.joblib'}")

router_artifacts = {
    'best_accuracy': accuracy,
    'scaler': scaler_emb,
    'scaler_seq': scaler_seq,
    'label_encoder': le_complexity,
    'sequence_features': sequence_features_available,
    'confusion_matrix': cm,
    'n_samples': len(X),
    'n_folds': N_FOLDS,
}

with open(ROUTER_PATH / 'router_artifacts.pkl', 'wb') as f:
    pickle.dump(router_artifacts, f)
print(f"Saved: {ROUTER_PATH / 'router_artifacts.pkl'}")

# ============================================================
# STEP 11: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 11: VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle(f'Router (Partial Sequences) - Accuracy: {accuracy*100:.2f}%', 
             fontsize=14, fontweight='bold')

# Confusion Matrix
ax1 = axes[0]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_complexity.classes_,
            yticklabels=le_complexity.classes_, ax=ax1)
ax1.set_xlabel('Predicted')
ax1.set_ylabel('Actual')
ax1.set_title('Confusion Matrix')

# Per-Class Accuracy
ax2 = axes[1]
classes = le_complexity.classes_
class_accs = [cm[i, i] / cm[i].sum() * 100 for i in range(len(classes))]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
bars = ax2.bar(classes, class_accs, color=colors)
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Per-Class Accuracy')
ax2.axhline(y=accuracy*100, color='black', linestyle='--', label=f'Overall: {accuracy*100:.1f}%')
ax2.legend()
for bar, acc in zip(bars, class_accs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(ROUTER_PATH / 'router_partial_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {ROUTER_PATH / 'router_partial_results.png'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("ROUTER (PARTIAL SEQUENCES) COMPLETE")
print("="*70)

print(f"""
SUMMARY
-------
Samples: {len(X):,} partial sequences
Features: {X.shape[1]} (embeddings)
Folds: {N_FOLDS}

ACCURACY: {accuracy*100:.2f}%

PER-CLASS:
  LOW:  {cm[list(le_complexity.classes_).index('LOW'), list(le_complexity.classes_).index('LOW')] / cm[list(le_complexity.classes_).index('LOW')].sum() * 100:.2f}%
  MED:  {cm[list(le_complexity.classes_).index('MED'), list(le_complexity.classes_).index('MED')] / cm[list(le_complexity.classes_).index('MED')].sum() * 100:.2f}%
  HIGH: {cm[list(le_complexity.classes_).index('HIGH'), list(le_complexity.classes_).index('HIGH')] / cm[list(le_complexity.classes_).index('HIGH')].sum() * 100:.2f}%

FILES SAVED:
  1. {ROUTER_PATH / 'router_xgb_best.joblib'}
  2. {ROUTER_PATH / 'router_artifacts.pkl'}
  3. {ROUTER_PATH / 'router_partial_results.png'}

NEXT: Re-run final_inference.py
""")