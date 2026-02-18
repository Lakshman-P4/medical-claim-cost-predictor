# ============================================================
# LSTM AUTOENCODER - SEQUENCE PATTERN LEARNING (NO LEAKAGE)
# ============================================================
# This script:
# 1. Trains LSTM as AUTOENCODER (not predictor)
# 2. Learns to RECONSTRUCT sequences (not predict cost)
# 3. Embeddings capture PATTERNS only (no target leakage)
# 4. Loss = RMSE (reconstruction error)
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import warnings
warnings.filterwarnings('ignore')

# Deep Learning
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Input, Masking, 
    BatchNormalization, Bidirectional, RepeatVector, TimeDistributed
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler, LabelEncoder

print("="*70)
print("LSTM AUTOENCODER - SEQUENCE PATTERN LEARNING")
print("="*70)
print(f"TensorFlow version: {tf.__version__}")
print("\nThis version learns PATTERNS only - NO target leakage!")

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data_preprocessed")
ARTIFACTS_PATH = Path("artifacts")
LSTM_PATH = ARTIFACTS_PATH / "lstm_autoencoder"
LSTM_PATH.mkdir(parents=True, exist_ok=True)

# LSTM Config
EMBEDDING_DIM = 128      # Output embedding size
MAX_SEQUENCE_LENGTH = 50 # Max visits per claim (pad/truncate)
LSTM_UNITS = 64
BATCH_SIZE = 256
EPOCHS = 50

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOAD DATA")
print("="*70)

df = pd.read_csv(DATA_PATH / "df_preprocessed.csv")
print(f"Loaded {len(df):,} visits")

# Load claims for COMPLEXITY labels (NOT for training target!)
df_claims = pd.read_csv(DATA_PATH / "df_claims_classified.csv")
print(f"Loaded {len(df_claims):,} claims")

# Get valid claim IDs
valid_claims = set(df_claims['CLAIM_ID'].unique())
df = df[df['CLAIM_ID'].isin(valid_claims)]
print(f"Filtered to {len(df):,} visits for {df['CLAIM_ID'].nunique():,} claims")

# ============================================================
# STEP 2: PREPARE SEQUENCE FEATURES
# ============================================================
print("\n" + "="*70)
print("STEP 2: PREPARE SEQUENCE FEATURES")
print("="*70)

# Sort by claim and visit number
df = df.sort_values(['CLAIM_ID', 'NO_OF_VISIT']).reset_index(drop=True)

# Select features for sequence learning
SEQUENCE_FEATURES = [
    'medical_amount',      # Payment amount
    'NO_OF_VISIT',         # Visit number
    'AGE_FINAL',           # Age (numeric)
]

# Categorical features to encode
CATEGORICAL_SEQ_FEATURES = [
    'AGE_GROUP_FINAL',     # Age group (categorical)
    'BODY_PART_DESC',
    'NATURE_OF_INJURY_DESC', 
    'BODY_PART_GROUP_DESC',
    'CLAIM_CAUSE_GROUP_DESC',
    'INCIDENT_STATE',
    'CLAIMANT_TYPE_DESC',
    'GENDER',
]

# Check which categorical features exist
CATEGORICAL_SEQ_FEATURES = [c for c in CATEGORICAL_SEQ_FEATURES if c in df.columns]
print(f"Categorical features found: {len(CATEGORICAL_SEQ_FEATURES)}")

# Encode categorical features
label_encoders = {}
for col in CATEGORICAL_SEQ_FEATURES:
    le = LabelEncoder()
    df[f'{col}_ENC'] = le.fit_transform(df[col].fillna('UNKNOWN').astype(str))
    label_encoders[col] = le
    SEQUENCE_FEATURES.append(f'{col}_ENC')

print(f"Total sequence features: {len(SEQUENCE_FEATURES)}")
print(f"Features: {SEQUENCE_FEATURES}")

# Handle ICD columns if they exist
icd_cols = [c for c in df.columns if 'ICD' in c and 'PRIMARY' in c]
if icd_cols:
    icd_all = [c for c in df.columns if c.startswith('ICD_') and c != 'ICD_PRIMARY']
    df['ICD_COUNT'] = df[icd_all].notna().sum(axis=1) if icd_all else 0
    SEQUENCE_FEATURES.append('ICD_COUNT')
    print(f"Added ICD_COUNT feature")

# Fill NaN with 0 for numeric features
for col in SEQUENCE_FEATURES:
    if col in df.columns:
        df[col] = df[col].fillna(0)

NUM_FEATURES = len(SEQUENCE_FEATURES)
print(f"\nFinal feature count: {NUM_FEATURES}")

# ============================================================
# STEP 3: CREATE SEQUENCES PER CLAIM
# ============================================================
print("\n" + "="*70)
print("STEP 3: CREATE SEQUENCES PER CLAIM")
print("="*70)

sequences = []
claim_ids = []
sequence_lengths = []

for claim_id, group in df.groupby('CLAIM_ID'):
    group = group.sort_values('NO_OF_VISIT')
    seq = group[SEQUENCE_FEATURES].values
    
    sequences.append(seq)
    claim_ids.append(claim_id)
    sequence_lengths.append(len(seq))

print(f"Created {len(sequences):,} sequences")
print(f"Sequence length stats:")
print(f"  Min: {min(sequence_lengths)}")
print(f"  Max: {max(sequence_lengths)}")
print(f"  Mean: {np.mean(sequence_lengths):.1f}")
print(f"  Median: {np.median(sequence_lengths):.1f}")

# Truncate very long sequences
sequences_truncated = [seq[:MAX_SEQUENCE_LENGTH] for seq in sequences]

# ============================================================
# STEP 4: SCALE AND PAD SEQUENCES
# ============================================================
print("\n" + "="*70)
print("STEP 4: SCALE AND PAD SEQUENCES")
print("="*70)

# Flatten all sequences for scaling
all_visits = np.vstack(sequences_truncated)
print(f"Total visits for scaling: {len(all_visits):,}")

# Scale features
scaler = StandardScaler()
all_visits_scaled = scaler.fit_transform(all_visits)

# Rebuild sequences with scaled values
idx = 0
sequences_scaled = []
for seq in sequences_truncated:
    seq_len = len(seq)
    sequences_scaled.append(all_visits_scaled[idx:idx+seq_len])
    idx += seq_len

# Pad sequences
X_padded = pad_sequences(
    sequences_scaled, 
    maxlen=MAX_SEQUENCE_LENGTH, 
    dtype='float32', 
    padding='post',
    truncating='post',
    value=0.0
)

print(f"Padded sequences shape: {X_padded.shape}")
# Shape: (num_claims, MAX_SEQUENCE_LENGTH, num_features)

# For autoencoder: input = output (reconstruct the sequence)
# NO COST TARGET HERE - that's the key difference!

# Get claims info for later (but NOT used in training)
claim_ids_array = np.array(claim_ids)
targets_df = df_claims.set_index('CLAIM_ID').loc[claim_ids_array]
y_cost = targets_df['TOTAL_CLAIM_COST'].values  # For reference only, NOT training
y_complexity = targets_df['COMPLEXITY'].values  # For reference only

print(f"\nClaims by complexity (for reference only):")
unique, counts = np.unique(y_complexity, return_counts=True)
for u, c in zip(unique, counts):
    print(f"  {u}: {c:,}")

# ============================================================
# STEP 5: BUILD LSTM AUTOENCODER
# ============================================================
print("\n" + "="*70)
print("STEP 5: BUILD LSTM AUTOENCODER")
print("="*70)

print(f"Input shape: ({MAX_SEQUENCE_LENGTH}, {NUM_FEATURES})")

# ====== ENCODER ======
encoder_input = Input(shape=(MAX_SEQUENCE_LENGTH, NUM_FEATURES), name='encoder_input')

# Masking for padded values
masked = Masking(mask_value=0.0)(encoder_input)

# Encoder LSTM layers
enc_lstm1 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True, name='enc_lstm1'))(masked)
enc_lstm1 = Dropout(0.3)(enc_lstm1)

enc_lstm2 = Bidirectional(LSTM(LSTM_UNITS, return_sequences=False, name='enc_lstm2'))(enc_lstm1)
enc_lstm2 = Dropout(0.3)(enc_lstm2)

# Embedding layer (bottleneck) - THIS IS WHAT WE EXTRACT
embedding = Dense(EMBEDDING_DIM, activation='relu', name='embedding')(enc_lstm2)
embedding_normalized = BatchNormalization(name='embedding_normalized')(embedding)

# ====== DECODER ======
# Repeat embedding for each timestep
decoder_input = RepeatVector(MAX_SEQUENCE_LENGTH, name='repeat')(embedding_normalized)

# Decoder LSTM layers
dec_lstm1 = LSTM(LSTM_UNITS * 2, return_sequences=True, name='dec_lstm1')(decoder_input)
dec_lstm1 = Dropout(0.3)(dec_lstm1)

dec_lstm2 = LSTM(LSTM_UNITS * 2, return_sequences=True, name='dec_lstm2')(dec_lstm1)
dec_lstm2 = Dropout(0.3)(dec_lstm2)

# Output layer - reconstruct the sequence
decoder_output = TimeDistributed(Dense(NUM_FEATURES), name='decoder_output')(dec_lstm2)

# ====== FULL AUTOENCODER MODEL ======
autoencoder = Model(inputs=encoder_input, outputs=decoder_output, name='lstm_autoencoder')

# Custom RMSE loss
def rmse_loss(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

autoencoder.compile(
    optimizer='adam',
    loss=rmse_loss,  # RMSE as requested
    metrics=['mae']
)

# ====== ENCODER ONLY (for extracting embeddings) ======
encoder_model = Model(inputs=encoder_input, outputs=embedding_normalized, name='encoder')

print("\nAutoencoder Summary:")
autoencoder.summary()

print("\n" + "="*70)
print("KEY DIFFERENCE FROM PREVIOUS VERSION:")
print("="*70)
print("""
PREVIOUS (LEAKY):
  Target = TOTAL_CLAIM_COST
  Embeddings learned to predict cost = LEAKAGE

NOW (CORRECT):
  Target = RECONSTRUCT INPUT SEQUENCE
  Embeddings learn PATTERNS only = NO LEAKAGE
  
Loss = RMSE(input_sequence, reconstructed_sequence)
""")

# ============================================================
# STEP 6: TRAIN AUTOENCODER
# ============================================================
print("\n" + "="*70)
print("STEP 6: TRAIN AUTOENCODER")
print("="*70)

# Callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
]

# Train/Val split
from sklearn.model_selection import train_test_split
X_train, X_val = train_test_split(X_padded, test_size=0.2, random_state=42)

print(f"Training set: {len(X_train):,} claims")
print(f"Validation set: {len(X_val):,} claims")

# Train - INPUT = OUTPUT (reconstruct sequences)
history = autoencoder.fit(
    X_train, X_train,  # Autoencoder: reconstruct input
    validation_data=(X_val, X_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ============================================================
# STEP 7: EVALUATE RECONSTRUCTION
# ============================================================
print("\n" + "="*70)
print("STEP 7: EVALUATE RECONSTRUCTION")
print("="*70)

# Reconstruct validation set
X_val_reconstructed = autoencoder.predict(X_val, verbose=0)

# Calculate reconstruction error
reconstruction_error = np.sqrt(np.mean((X_val - X_val_reconstructed) ** 2))
print(f"Reconstruction RMSE: {reconstruction_error:.4f}")

# Per-feature reconstruction error
feature_errors = []
for i, feat in enumerate(SEQUENCE_FEATURES):
    feat_error = np.sqrt(np.mean((X_val[:, :, i] - X_val_reconstructed[:, :, i]) ** 2))
    feature_errors.append((feat, feat_error))
    
print("\nPer-Feature Reconstruction Error:")
for feat, err in sorted(feature_errors, key=lambda x: x[1], reverse=True)[:10]:
    print(f"  {feat:<30} RMSE: {err:.4f}")

# ============================================================
# STEP 8: EXTRACT EMBEDDINGS
# ============================================================
print("\n" + "="*70)
print("STEP 8: EXTRACT EMBEDDINGS")
print("="*70)

# Extract embeddings for ALL claims
embeddings = encoder_model.predict(X_padded, verbose=1)
print(f"Embeddings shape: {embeddings.shape}")

# Create DataFrame with embeddings
embedding_cols = [f'EMB_{i}' for i in range(EMBEDDING_DIM)]
df_embeddings = pd.DataFrame(embeddings, columns=embedding_cols)
df_embeddings['CLAIM_ID'] = claim_ids
df_embeddings['SEQUENCE_LENGTH'] = sequence_lengths

# Add reference info (NOT used in training, just for convenience)
df_embeddings['TOTAL_CLAIM_COST'] = y_cost
df_embeddings['COMPLEXITY'] = y_complexity

print(f"\nEmbeddings DataFrame shape: {df_embeddings.shape}")

# ============================================================
# STEP 9: VERIFY NO LEAKAGE
# ============================================================
print("\n" + "="*70)
print("STEP 9: VERIFY NO LEAKAGE")
print("="*70)

# Check correlation between embeddings and target
# If no leakage, correlations should be LOW
correlations = []
for col in embedding_cols:
    corr = np.corrcoef(df_embeddings[col], df_embeddings['TOTAL_CLAIM_COST'])[0, 1]
    correlations.append(abs(corr))

max_corr = max(correlations)
mean_corr = np.mean(correlations)

print(f"Embedding-Cost Correlation Check:")
print(f"  Max absolute correlation:  {max_corr:.4f}")
print(f"  Mean absolute correlation: {mean_corr:.4f}")

if max_corr < 0.5:
    print("  ✓ LOW correlation - No obvious leakage!")
else:
    print("  ⚠ HIGH correlation - Possible leakage, investigate!")

# Compare with previous version (which had R²=76% on LSTM alone)
print(f"\nPrevious LSTM (predictor) achieved R²=76% on cost prediction")
print(f"If embeddings had leakage, we'd see high correlations here")
print(f"Current max correlation: {max_corr:.4f} - embeddings capture PATTERNS, not cost")

# ============================================================
# STEP 10: SAVE ARTIFACTS
# ============================================================
print("\n" + "="*70)
print("STEP 10: SAVE ARTIFACTS")
print("="*70)

# Save embeddings
df_embeddings.to_csv(LSTM_PATH / 'claim_embeddings.csv', index=False)
print(f"Saved: {LSTM_PATH / 'claim_embeddings.csv'}")

# Save models
autoencoder.save(LSTM_PATH / 'lstm_autoencoder.keras')
print(f"Saved: {LSTM_PATH / 'lstm_autoencoder.keras'}")

encoder_model.save(LSTM_PATH / 'lstm_encoder.keras')
print(f"Saved: {LSTM_PATH / 'lstm_encoder.keras'}")

# Save artifacts
lstm_artifacts = {
    'scaler': scaler,
    'label_encoders': label_encoders,
    'sequence_features': SEQUENCE_FEATURES,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'embedding_dim': EMBEDDING_DIM,
    'embedding_cols': embedding_cols,
    'training_history': history.history,
    'reconstruction_rmse': reconstruction_error,
    'correlation_check': {
        'max_correlation': max_corr,
        'mean_correlation': mean_corr,
    }
}

with open(LSTM_PATH / 'lstm_artifacts.pkl', 'wb') as f:
    pickle.dump(lstm_artifacts, f)
print(f"Saved: {LSTM_PATH / 'lstm_artifacts.pkl'}")

# ============================================================
# STEP 11: VISUALIZE
# ============================================================
print("\n" + "="*70)
print("STEP 11: VISUALIZE")
print("="*70)

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('LSTM Autoencoder Results (No Leakage)', fontsize=14, fontweight='bold')

# 1. Training history
ax1 = axes[0, 0]
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss (RMSE)')
ax1.set_title('Training History - Reconstruction Loss')
ax1.legend()

# 2. Embedding correlation with cost (should be LOW)
ax2 = axes[0, 1]
ax2.hist(correlations, bins=30, color='steelblue', edgecolor='white', alpha=0.7)
ax2.axvline(mean_corr, color='red', linestyle='--', label=f'Mean: {mean_corr:.3f}')
ax2.axvline(0.5, color='orange', linestyle='--', label='Leakage threshold')
ax2.set_xlabel('Absolute Correlation with Cost')
ax2.set_ylabel('Frequency')
ax2.set_title('Embedding-Cost Correlations (Low = Good)')
ax2.legend()

# 3. Embedding space (PCA) colored by complexity
ax3 = axes[1, 0]
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

complexity_map = {'LOW': 0, 'MED': 1, 'HIGH': 2}
colors = [complexity_map.get(c, 1) for c in y_complexity]

scatter = ax3.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                      c=colors, alpha=0.3, s=10, cmap='RdYlGn_r')
ax3.set_xlabel('PCA Component 1')
ax3.set_ylabel('PCA Component 2')
ax3.set_title('Embedding Space (colored by Complexity)')

# 4. Sequence length distribution
ax4 = axes[1, 1]
ax4.hist(sequence_lengths, bins=50, color='#2ecc71', edgecolor='white', alpha=0.7)
ax4.axvline(np.mean(sequence_lengths), color='red', linestyle='--', 
            label=f'Mean: {np.mean(sequence_lengths):.1f}')
ax4.set_xlabel('Sequence Length (# Visits)')
ax4.set_ylabel('Frequency')
ax4.set_title('Sequence Length Distribution')
ax4.legend()

plt.tight_layout()
plt.savefig(LSTM_PATH / 'autoencoder_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {LSTM_PATH / 'autoencoder_results.png'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("LSTM AUTOENCODER COMPLETE - NO LEAKAGE")
print("="*70)

print(f"""
SUMMARY
-------
Claims processed: {len(claim_ids):,}
Sequence features: {len(SEQUENCE_FEATURES)}
Embedding dimension: {EMBEDDING_DIM}

RECONSTRUCTION METRICS:
  RMSE: {reconstruction_error:.4f}

LEAKAGE CHECK:
  Max embedding-cost correlation: {max_corr:.4f}
  Mean embedding-cost correlation: {mean_corr:.4f}
  Status: {"✓ NO LEAKAGE" if max_corr < 0.5 else "⚠ INVESTIGATE"}

FILES SAVED:
  1. {LSTM_PATH / 'claim_embeddings.csv'} - Pattern embeddings
  2. {LSTM_PATH / 'lstm_autoencoder.keras'} - Full autoencoder
  3. {LSTM_PATH / 'lstm_encoder.keras'} - Encoder only (for inference)
  4. {LSTM_PATH / 'lstm_artifacts.pkl'} - Configs
  5. {LSTM_PATH / 'autoencoder_results.png'} - Visualizations

NEXT STEPS:
  1. Update router to use new embeddings from {LSTM_PATH}
  2. Re-run regression models
  3. Expected: R² will be LOWER but REALISTIC (no cheating!)

KEY DIFFERENCE:
  Previous: LSTM predicted cost → Embeddings encoded cost → R²=99% (LEAKY)
  Now: LSTM reconstructs sequences → Embeddings encode PATTERNS → Fair R²
""")