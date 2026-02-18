# ============================================================
# ROUTER V2 - MULTIPLE ALGORITHMS WITH ENGINEERED FEATURES
# ============================================================
# This script:
# 1. Loads engineered features from df_engineered.csv
# 2. Loads COMPLEXITY labels from df_claims_classified.csv (training target only)
# 3. Tries XGBoost, LightGBM, CatBoost, RandomForest routers
# 4. Hyperparameter tuning for each
# 5. Target: 75%+ accuracy
# 6. Saves best router model
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

print("="*70)
print("ROUTER V2 - MULTIPLE ALGORITHMS")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data_preprocessed")
ARTIFACTS_PATH = Path("artifacts")
ROUTER_PATH = ARTIFACTS_PATH / "router_v2"
ROUTER_PATH.mkdir(parents=True, exist_ok=True)

N_FOLDS = 5
N_ITER = 50  # Number of random combinations to try per model
TARGET_ACCURACY = 0.75

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOAD DATA")
print("="*70)

# Load engineered features
df = pd.read_csv(DATA_PATH / "df_engineered.csv")
print(f"Loaded engineered data: {len(df):,} claims")

# Load COMPLEXITY labels (from KNN classification - training target only)
df_classified = pd.read_csv(DATA_PATH / "df_claims_classified.csv")
print(f"Loaded classified data: {len(df_classified):,} claims")

# Merge COMPLEXITY into df
df = df.merge(
    df_classified[['CLAIM_ID', 'COMPLEXITY']],
    on='CLAIM_ID',
    how='inner'
)
print(f"After merge: {len(df):,} claims with COMPLEXITY labels")
print(f"\nCOMPLEXITY distribution:")
print(df['COMPLEXITY'].value_counts())

# ============================================================
# STEP 2: PREPARE FEATURES
# ============================================================
print("\n" + "="*70)
print("STEP 2: PREPARE FEATURES")
print("="*70)

# Load feature engineering artifacts
with open(ARTIFACTS_PATH / "fe_artifacts.pkl", 'rb') as f:
    fe_artifacts = pickle.load(f)

# Get all engineered features
ALL_FEATURES = fe_artifacts['feature_lists']['all']

# Filter to features that exist in df
FEATURE_COLS = [c for c in ALL_FEATURES if c in df.columns]
print(f"Total features available: {len(FEATURE_COLS)}")

# Prepare X and y
X = df[FEATURE_COLS].fillna(0).values
y_raw = df['COMPLEXITY'].values

# Encode target
le_complexity = LabelEncoder()
y = le_complexity.fit_transform(y_raw)
print(f"Classes: {le_complexity.classes_}")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nFeature matrix shape: {X_scaled.shape}")
print(f"Target distribution: {np.bincount(y)}")

# ============================================================
# STEP 3: MODEL DEFINITIONS (OPTIMIZED)
# ============================================================
print("\n" + "="*70)
print("STEP 3: MODEL DEFINITIONS (WITH CLASS WEIGHTS FOR MED)")
print("="*70)

# Calculate class weights to boost MED
from sklearn.utils.class_weight import compute_class_weight
class_weights_array = compute_class_weight('balanced', classes=np.unique(y), y=y)
class_weight_dict = {i: w for i, w in enumerate(class_weights_array)}
print(f"Class weights: {class_weight_dict}")

# Boost MED class weight further (MED is usually class 1 or middle class)
# Find which class is MED
med_class_idx = list(le_complexity.classes_).index('MED')
class_weight_dict[med_class_idx] *= 1.5  # Boost MED by 50%
print(f"Boosted MED (class {med_class_idx}) weight: {class_weight_dict}")

# Convert to sample weights for XGBoost (it uses scale_pos_weight differently)
# For multi-class, we'll use sample_weight in fit

models = {}

# --- XGBoost (with class weights) ---
models['XGBoost'] = {
    'model': xgb.XGBClassifier(random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='mlogloss'),
    'params': {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'min_child_weight': [1, 3],
        'colsample_bytree': [0.8, 1.0],
    }
}

# --- LightGBM (with class weights) ---
models['LightGBM'] = {
    'model': lgb.LGBMClassifier(random_state=42, n_jobs=-1, verbose=-1, class_weight=class_weight_dict),
    'params': {
        'max_depth': [6, 8, 10],
        'learning_rate': [0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'num_leaves': [31, 50],
    }
}

# --- RandomForest (with class weights) ---
models['RandomForest'] = {
    'model': RandomForestClassifier(random_state=42, n_jobs=-1, class_weight=class_weight_dict),
    'params': {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
    }
}

print(f"Models to test: {list(models.keys())}")

# ============================================================
# STEP 4: TRAIN AND EVALUATE EACH MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 4: TRAIN AND EVALUATE MODELS")
print("="*70)

results = {}
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for name, config in models.items():
    print(f"\n{'='*50}")
    print(f"Training: {name}")
    print(f"{'='*50}")
    
    param_grid = config['params']
    
    # RandomizedSearchCV - samples N_ITER random combinations
    random_search = RandomizedSearchCV(
        config['model'],
        param_grid,
        n_iter=N_ITER,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_scaled, y)
    
    # Store results
    results[name] = {
        'best_model': random_search.best_estimator_,
        'best_params': random_search.best_params_,
        'best_score': random_search.best_score_,
        'cv_results': random_search.cv_results_
    }
    
    print(f"\n{name} Results:")
    print(f"  Best Accuracy: {random_search.best_score_:.4f} ({random_search.best_score_*100:.2f}%)")
    print(f"  Best Params: {random_search.best_params_}")
    
    # Check if target met
    if random_search.best_score_ >= TARGET_ACCURACY:
        print(f"  ✓ TARGET MET (>= {TARGET_ACCURACY*100}%)")
    else:
        print(f"  ✗ Below target ({TARGET_ACCURACY*100}%)")

# ============================================================
# STEP 5: COMPARE MODELS
# ============================================================
print("\n" + "="*70)
print("STEP 5: MODEL COMPARISON")
print("="*70)

print(f"\n{'Model':<15} {'Accuracy':>12} {'Status':<15}")
print("-" * 45)

for name, res in sorted(results.items(), key=lambda x: x[1]['best_score'], reverse=True):
    status = "✓ TARGET MET" if res['best_score'] >= TARGET_ACCURACY else "✗ Below"
    print(f"{name:<15} {res['best_score']*100:>11.2f}% {status:<15}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['best_score'])
best_model = results[best_model_name]['best_model']
best_accuracy = results[best_model_name]['best_score']

print(f"\n✓ BEST MODEL: {best_model_name}")
print(f"  Accuracy: {best_accuracy*100:.2f}%")

# ============================================================
# STEP 6: DETAILED EVALUATION OF BEST MODEL
# ============================================================
print("\n" + "="*70)
print("STEP 6: DETAILED EVALUATION")
print("="*70)

# Cross-validation predictions for confusion matrix
y_pred_cv = np.zeros_like(y)
y_proba_cv = np.zeros((len(y), len(le_complexity.classes_)))

for train_idx, val_idx in cv.split(X_scaled, y):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Clone model with best params
    if best_model_name == 'XGBoost':
        model = xgb.XGBClassifier(**results[best_model_name]['best_params'], random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='mlogloss')
    elif best_model_name == 'LightGBM':
        model = lgb.LGBMClassifier(**results[best_model_name]['best_params'], random_state=42, n_jobs=-1, verbose=-1, class_weight=class_weight_dict)
    else:
        model = RandomForestClassifier(**results[best_model_name]['best_params'], random_state=42, n_jobs=-1, class_weight=class_weight_dict)
    
    model.fit(X_train, y_train)
    y_pred_cv[val_idx] = model.predict(X_val)
    y_proba_cv[val_idx] = model.predict_proba(X_val)

# Classification report
print("\nClassification Report (Default Thresholds):")
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
# STEP 6B: THRESHOLD TUNING FOR MED
# ============================================================
print("\n" + "="*70)
print("STEP 6B: THRESHOLD TUNING FOR MED")
print("="*70)

# Find MED class index
med_idx = list(le_complexity.classes_).index('MED')
low_idx = list(le_complexity.classes_).index('LOW')
high_idx = list(le_complexity.classes_).index('HIGH')

print(f"Class indices - LOW: {low_idx}, MED: {med_idx}, HIGH: {high_idx}")

# Try different thresholds for MED
best_threshold = 0.0
best_med_accuracy = 0
best_overall_accuracy = accuracy_score(y, y_pred_cv)
best_threshold_results = None

print("\nTesting MED threshold adjustments...")
print(f"{'Threshold':<12} {'MED Acc':>10} {'Overall Acc':>12} {'LOW Acc':>10} {'HIGH Acc':>10}")
print("-" * 60)

for threshold_boost in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]:
    # Adjust probabilities - boost MED
    adjusted_proba = y_proba_cv.copy()
    adjusted_proba[:, med_idx] += threshold_boost
    
    # Get new predictions
    y_pred_adjusted = np.argmax(adjusted_proba, axis=1)
    
    # Calculate accuracies
    overall_acc = accuracy_score(y, y_pred_adjusted)
    cm_adjusted = confusion_matrix(y, y_pred_adjusted)
    
    med_acc = cm_adjusted[med_idx, med_idx] / cm_adjusted[med_idx].sum() if cm_adjusted[med_idx].sum() > 0 else 0
    low_acc = cm_adjusted[low_idx, low_idx] / cm_adjusted[low_idx].sum() if cm_adjusted[low_idx].sum() > 0 else 0
    high_acc = cm_adjusted[high_idx, high_idx] / cm_adjusted[high_idx].sum() if cm_adjusted[high_idx].sum() > 0 else 0
    
    print(f"{threshold_boost:<12.2f} {med_acc*100:>9.2f}% {overall_acc*100:>11.2f}% {low_acc*100:>9.2f}% {high_acc*100:>9.2f}%")
    
    # Track best balanced result (prioritize MED improvement without destroying others)
    if med_acc > best_med_accuracy and overall_acc >= best_overall_accuracy * 0.95:  # Allow 5% overall drop
        best_threshold = threshold_boost
        best_med_accuracy = med_acc
        best_threshold_results = {
            'cm': cm_adjusted,
            'med_acc': med_acc,
            'low_acc': low_acc,
            'high_acc': high_acc,
            'overall_acc': overall_acc
        }

print(f"\n✓ Best MED threshold boost: {best_threshold}")
if best_threshold_results:
    print(f"  MED Accuracy: {best_threshold_results['med_acc']*100:.2f}%")
    print(f"  Overall Accuracy: {best_threshold_results['overall_acc']*100:.2f}%")

# ============================================================
# STEP 6C: ONE-VS-REST MED CLASSIFIER
# ============================================================
print("\n" + "="*70)
print("STEP 6C: ONE-VS-REST MED CLASSIFIER")
print("="*70)

# Create binary target: MED vs NOT_MED
y_binary_med = (y == med_idx).astype(int)
print(f"MED vs NOT_MED distribution: NOT_MED={np.sum(1-y_binary_med):,}, MED={np.sum(y_binary_med):,}")

# Train a separate binary classifier for MED
from sklearn.model_selection import cross_val_predict

print("\nTraining binary MED classifier...")

# Use XGBoost for binary classification with class weight
med_weight = len(y_binary_med) / (2 * np.sum(y_binary_med))  # Balance classes
not_med_weight = len(y_binary_med) / (2 * np.sum(1 - y_binary_med))

binary_clf = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=med_weight / not_med_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss'
)

# Cross-val predict for binary
y_binary_pred = cross_val_predict(binary_clf, X_scaled, y_binary_med, cv=cv)
y_binary_proba = cross_val_predict(binary_clf, X_scaled, y_binary_med, cv=cv, method='predict_proba')

binary_acc = accuracy_score(y_binary_med, y_binary_pred)
print(f"Binary MED classifier accuracy: {binary_acc*100:.2f}%")

# Binary classification report
from sklearn.metrics import precision_recall_fscore_support
prec, rec, f1, _ = precision_recall_fscore_support(y_binary_med, y_binary_pred, average='binary')
print(f"  Precision: {prec*100:.2f}%")
print(f"  Recall: {rec*100:.2f}%")
print(f"  F1-Score: {f1*100:.2f}%")

# ============================================================
# STEP 6D: COMBINED APPROACH - USE BINARY TO REFINE MULTICLASS
# ============================================================
print("\n" + "="*70)
print("STEP 6D: COMBINED APPROACH")
print("="*70)

# Strategy: If binary classifier is confident about MED, override multiclass prediction
y_pred_combined = y_pred_cv.copy()
med_proba_threshold = 0.6  # If binary says >60% MED, trust it

# Get MED probability from binary classifier
med_binary_proba = y_binary_proba[:, 1]

# Override predictions where binary is confident about MED
confident_med_mask = med_binary_proba >= med_proba_threshold
override_count = 0

for i in range(len(y_pred_combined)):
    if confident_med_mask[i] and y_pred_combined[i] != med_idx:
        y_pred_combined[i] = med_idx
        override_count += 1

print(f"Overrode {override_count:,} predictions to MED based on binary classifier")

# Evaluate combined approach
combined_acc = accuracy_score(y, y_pred_combined)
cm_combined = confusion_matrix(y, y_pred_combined)

print(f"\nCombined Approach Results:")
print(f"  Overall Accuracy: {combined_acc*100:.2f}%")

print("\nPer-Class Accuracy (Combined):")
for i, cls in enumerate(le_complexity.classes_):
    class_acc = cm_combined[i, i] / cm_combined[i].sum()
    print(f"  {cls}: {class_acc*100:.2f}%")

# ============================================================
# STEP 6E: COMPARISON OF APPROACHES
# ============================================================
print("\n" + "="*70)
print("STEP 6E: COMPARISON OF ALL APPROACHES")
print("="*70)

approaches = {
    'Default': {'cm': cm, 'overall_acc': accuracy_score(y, y_pred_cv)},
    'Threshold Tuned': best_threshold_results,
    'Combined (Binary+Multi)': {'cm': cm_combined, 'overall_acc': combined_acc}
}

print(f"\n{'Approach':<25} {'Overall':>10} {'LOW':>10} {'MED':>10} {'HIGH':>10}")
print("-" * 70)

best_approach_name = 'Default'
best_approach_med = 0

for name, data in approaches.items():
    if data is None:
        continue
    cm_temp = data['cm']
    overall = data['overall_acc']
    low = cm_temp[low_idx, low_idx] / cm_temp[low_idx].sum()
    med = cm_temp[med_idx, med_idx] / cm_temp[med_idx].sum()
    high = cm_temp[high_idx, high_idx] / cm_temp[high_idx].sum()
    
    print(f"{name:<25} {overall*100:>9.2f}% {low*100:>9.2f}% {med*100:>9.2f}% {high*100:>9.2f}%")
    
    # Track best for MED while keeping overall decent
    if med > best_approach_med and overall >= 0.65:
        best_approach_med = med
        best_approach_name = name

print(f"\n✓ Best approach for MED: {best_approach_name} ({best_approach_med*100:.2f}%)")

# Save the best approach info for later use
best_approach_data = approaches[best_approach_name]

# ============================================================
# STEP 7: FEATURE IMPORTANCE
# ============================================================
print("\n" + "="*70)
print("STEP 7: FEATURE IMPORTANCE")
print("="*70)

# Train final model on all data
if best_model_name == 'XGBoost':
    final_model = xgb.XGBClassifier(**results[best_model_name]['best_params'], random_state=42, n_jobs=-1, verbosity=0, use_label_encoder=False, eval_metric='mlogloss')
elif best_model_name == 'LightGBM':
    final_model = lgb.LGBMClassifier(**results[best_model_name]['best_params'], random_state=42, n_jobs=-1, verbose=-1, class_weight=class_weight_dict)
else:
    final_model = RandomForestClassifier(**results[best_model_name]['best_params'], random_state=42, n_jobs=-1, class_weight=class_weight_dict)

final_model.fit(X_scaled, y)

# Train final binary MED classifier
final_binary_clf = xgb.XGBClassifier(
    max_depth=8,
    learning_rate=0.1,
    n_estimators=200,
    scale_pos_weight=med_weight / not_med_weight,
    random_state=42,
    n_jobs=-1,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss'
)
final_binary_clf.fit(X_scaled, y_binary_med)

# Get feature importance
if hasattr(final_model, 'feature_importances_'):
    importance = final_model.feature_importances_
    feat_imp = pd.DataFrame({
        'Feature': FEATURE_COLS,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    print("\nTop 20 Most Important Features for Routing:")
    print("-" * 50)
    for i, row in feat_imp.head(20).iterrows():
        print(f"  {row['Feature']:<40} {row['Importance']:.4f}")
    
    # Save feature importance
    feat_imp.to_csv(ROUTER_PATH / 'router_feature_importance.csv', index=False)

# ============================================================
# STEP 8: SAVE ARTIFACTS
# ============================================================
print("\n" + "="*70)
print("STEP 8: SAVE ARTIFACTS")
print("="*70)

# Save best model (use joblib for all models - more reliable)
joblib.dump(final_model, ROUTER_PATH / 'best_router.joblib')
print(f"  Saved: {ROUTER_PATH / 'best_router.joblib'}")

# Save binary MED classifier
joblib.dump(final_binary_clf, ROUTER_PATH / 'binary_med_classifier.joblib')
print(f"  Saved: {ROUTER_PATH / 'binary_med_classifier.joblib'}")

# Save all artifacts
router_artifacts = {
    'best_model_name': best_model_name,
    'best_params': results[best_model_name]['best_params'],
    'best_accuracy': best_accuracy,
    'feature_cols': FEATURE_COLS,
    'scaler': scaler,
    'label_encoder': le_complexity,
    'class_weight_dict': class_weight_dict,
    'all_results': {name: {'best_score': res['best_score'], 'best_params': res['best_params']} 
                   for name, res in results.items()},
    'confusion_matrix': cm,
    # Threshold tuning results
    'best_threshold': best_threshold,
    'best_threshold_results': best_threshold_results,
    # Binary classifier info
    'binary_clf_accuracy': binary_acc,
    'binary_clf_precision': prec,
    'binary_clf_recall': rec,
    'med_proba_threshold': med_proba_threshold,
    # Best approach
    'best_approach_name': best_approach_name,
    'best_approach_med_accuracy': best_approach_med,
    # Class indices
    'class_indices': {'LOW': low_idx, 'MED': med_idx, 'HIGH': high_idx},
}

with open(ROUTER_PATH / 'router_v2_artifacts.pkl', 'wb') as f:
    pickle.dump(router_artifacts, f)
print(f"  Saved: {ROUTER_PATH / 'router_v2_artifacts.pkl'}")

# ============================================================
# STEP 9: VISUALIZATION
# ============================================================
print("\n" + "="*70)
print("STEP 9: VISUALIZATION")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle(f'Router V2 Results - Best: {best_model_name} ({best_accuracy*100:.2f}%)', fontsize=14, fontweight='bold')

# 1. Model Comparison
ax1 = axes[0, 0]
model_names = list(results.keys())
accuracies = [results[m]['best_score'] * 100 for m in model_names]
colors = ['#2ecc71' if acc >= TARGET_ACCURACY*100 else '#e74c3c' for acc in accuracies]
bars = ax1.bar(model_names, accuracies, color=colors, edgecolor='white')
ax1.axhline(y=TARGET_ACCURACY*100, color='black', linestyle='--', label=f'Target: {TARGET_ACCURACY*100}%')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Model Comparison')
ax1.legend()
for bar, acc in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
             f'{acc:.1f}%', ha='center', va='bottom')

# 2. Confusion Matrix
ax2 = axes[0, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_complexity.classes_, 
            yticklabels=le_complexity.classes_, ax=ax2)
ax2.set_xlabel('Predicted')
ax2.set_ylabel('Actual')
ax2.set_title('Confusion Matrix')

# 3. Feature Importance (top 15)
ax3 = axes[1, 0]
if hasattr(final_model, 'feature_importances_'):
    top_feat = feat_imp.head(15)
    ax3.barh(range(len(top_feat)), top_feat['Importance'], color='steelblue')
    ax3.set_yticks(range(len(top_feat)))
    ax3.set_yticklabels(top_feat['Feature'])
    ax3.set_xlabel('Importance')
    ax3.set_title('Top 15 Feature Importance')
    ax3.invert_yaxis()

# 4. Per-Class Accuracy
ax4 = axes[1, 1]
class_accuracies = [cm[i, i] / cm[i].sum() * 100 for i in range(len(le_complexity.classes_))]
colors = ['#3498db', '#2ecc71', '#e74c3c']
bars = ax4.bar(le_complexity.classes_, class_accuracies, color=colors, edgecolor='white')
ax4.axhline(y=TARGET_ACCURACY*100, color='black', linestyle='--', label=f'Target: {TARGET_ACCURACY*100}%')
ax4.set_ylabel('Accuracy (%)')
ax4.set_title('Per-Class Accuracy')
ax4.legend()
for bar, acc in zip(bars, class_accuracies):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{acc:.1f}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig(ROUTER_PATH / 'router_v2_results.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {ROUTER_PATH / 'router_v2_results.png'}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("ROUTER V2 COMPLETE")
print("="*70)

print(f"""
SUMMARY
-------
Features Used: {len(FEATURE_COLS)}
Models Tested: {len(models)}

RESULTS:
""")

for name, res in sorted(results.items(), key=lambda x: x[1]['best_score'], reverse=True):
    status = "✓" if res['best_score'] >= TARGET_ACCURACY else "✗"
    print(f"  {status} {name}: {res['best_score']*100:.2f}%")

print(f"""
BEST MODEL: {best_model_name}
  Accuracy: {best_accuracy*100:.2f}%
  Target:   {TARGET_ACCURACY*100:.2f}%
  Status:   {'✓ TARGET MET!' if best_accuracy >= TARGET_ACCURACY else '✗ Below target'}

Previous Router (CatBoost): 69.42%
New Router ({best_model_name}): {best_accuracy*100:.2f}%
Improvement: {(best_accuracy - 0.6942) / 0.6942 * 100:.1f}%

Files Saved:
  1. {ROUTER_PATH / 'best_router.joblib'}
  2. {ROUTER_PATH / 'router_v2_artifacts.pkl'}
  3. {ROUTER_PATH / 'router_feature_importance.csv'}
  4. {ROUTER_PATH / 'router_v2_results.png'}

Next Steps:
  1. Run train_low.py - Train LOW complexity model
  2. Run train_med.py - Train MED complexity model  
  3. Run train_high.py - Train HIGH complexity model
  4. Run inference_final.py - Generate predictions
""")
