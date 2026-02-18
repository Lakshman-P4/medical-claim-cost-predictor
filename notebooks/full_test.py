# ============================================================
# FULL MODEL TESTING - ALL HELD-OUT TEST DATA
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("="*70)
print("FULL MODEL TESTING - ALL HELD-OUT TEST DATA")
print("="*70)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = Path("data_preprocessed")
ARTIFACTS_PATH = Path("artifacts")
OUTPUT_PATH = ARTIFACTS_PATH / "final_results"
MODEL_PATH = ARTIFACTS_PATH / "regression_models"

RANDOM_STATE = 42

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n" + "="*70)
print("STEP 1: LOAD DATA")
print("="*70)

df_claims = pd.read_csv(DATA_PATH / "df_claims_classified.csv")
print(f"Total claims: {len(df_claims):,}")

df_predictions = pd.read_csv(OUTPUT_PATH / "final_predictions.csv")
print(f"Total predictions: {len(df_predictions):,}")

# ============================================================
# STEP 2: RECREATE TRAIN/TEST SPLIT
# ============================================================
print("\n" + "="*70)
print("STEP 2: RECREATE TRAIN/TEST SPLIT")
print("="*70)

train_claim_ids = set()
test_claim_ids = set()

for complexity in ['LOW', 'MED', 'HIGH']:
    complexity_claims = df_claims[df_claims['COMPLEXITY'] == complexity]['CLAIM_ID'].values
    
    train_claims, test_claims = train_test_split(
        complexity_claims, 
        test_size=0.2, 
        random_state=RANDOM_STATE
    )
    
    train_claim_ids.update(train_claims)
    test_claim_ids.update(test_claims)
    
    print(f"  {complexity}: Train={len(train_claims):,} | Test={len(test_claims):,}")

print(f"\nTotal Train Claims: {len(train_claim_ids):,}")
print(f"Total Test Claims: {len(test_claim_ids):,}")

# ============================================================
# STEP 3: SPLIT PREDICTIONS INTO TRAIN/TEST
# ============================================================
print("\n" + "="*70)
print("STEP 3: SPLIT PREDICTIONS")
print("="*70)

df_train = df_predictions[df_predictions['CLAIM_ID'].isin(train_claim_ids)]
df_test = df_predictions[df_predictions['CLAIM_ID'].isin(test_claim_ids)]

print(f"Train predictions: {len(df_train):,} samples ({len(df_train['CLAIM_ID'].unique()):,} claims)")
print(f"Test predictions: {len(df_test):,} samples ({len(df_test['CLAIM_ID'].unique()):,} claims)")

# ============================================================
# STEP 4: CALCULATE TEST METRICS - OVERALL
# ============================================================
print("\n" + "="*70)
print("STEP 4: TEST SET METRICS - OVERALL")
print("="*70)

test_r2 = r2_score(df_test['ACTUAL_TOTAL'], df_test['PREDICTED_TOTAL'])
test_mae = mean_absolute_error(df_test['ACTUAL_TOTAL'], df_test['PREDICTED_TOTAL'])
test_rmse = np.sqrt(mean_squared_error(df_test['ACTUAL_TOTAL'], df_test['PREDICTED_TOTAL']))

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                 TEST SET PERFORMANCE (UNSEEN DATA)                ║
╠══════════════════════════════════════════════════════════════════╣
║  Test Samples:       {len(df_test):>10,}                                  ║
║  Test Claims:        {df_test['CLAIM_ID'].nunique():>10,}                                  ║
║                                                                   ║
║  TEST R²:            {test_r2:>10.4f} ({test_r2*100:.2f}%)                       ║
║  TEST MAE:           ${test_mae:>10,.2f}                                 ║
║  TEST RMSE:          ${test_rmse:>10,.2f}                                 ║
╚══════════════════════════════════════════════════════════════════╝
""")

# ============================================================
# STEP 5: TEST METRICS - PER COMPLEXITY
# ============================================================
print("\n" + "="*70)
print("STEP 5: TEST SET METRICS - PER COMPLEXITY")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║              TEST SET PERFORMANCE BY COMPLEXITY                   ║
╠══════════════════════════════════════════════════════════════════╣
║ Complexity │  Samples  │    R²     │     MAE      │    RMSE      ║
╠────────────┼───────────┼───────────┼──────────────┼──────────────╣""")

test_complexity_metrics = {}

for complexity in ['LOW', 'MED', 'HIGH']:
    # Filter by ACTUAL complexity
    df_comp = df_test[df_test['ACTUAL_COMPLEXITY'] == complexity]
    
    if len(df_comp) > 1:
        r2 = r2_score(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'])
        mae = mean_absolute_error(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'])
        rmse = np.sqrt(mean_squared_error(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL']))
        
        test_complexity_metrics[complexity] = {
            'samples': len(df_comp),
            'claims': df_comp['CLAIM_ID'].nunique(),
            'r2': r2,
            'mae': mae,
            'rmse': rmse
        }
        
        print(f"║ {complexity:<10} │ {len(df_comp):>9,} │ {r2:>9.4f} │ ${mae:>10,.2f} │ ${rmse:>10,.2f} ║")

print(f"╚══════════════════════════════════════════════════════════════════╝")

# ============================================================
# STEP 6: TEST METRICS - BY PREDICTED COMPLEXITY (ROUTING)
# ============================================================
print("\n" + "="*70)
print("STEP 6: TEST SET METRICS - BY PREDICTED COMPLEXITY")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║          TEST SET PERFORMANCE BY PREDICTED (ROUTED)               ║
╠══════════════════════════════════════════════════════════════════╣
║ Routed To  │  Samples  │    R²     │     MAE      │  Cost Range  ║
╠────────────┼───────────┼───────────┼──────────────┼──────────────╣""")

for complexity in ['LOW', 'MED', 'HIGH']:
    df_comp = df_test[df_test['PREDICTED_COMPLEXITY'] == complexity]
    
    if len(df_comp) > 1:
        r2 = r2_score(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'])
        mae = mean_absolute_error(df_comp['ACTUAL_TOTAL'], df_comp['PREDICTED_TOTAL'])
        min_cost = df_comp['ACTUAL_TOTAL'].min()
        max_cost = df_comp['ACTUAL_TOTAL'].max()
        
        print(f"║ {complexity:<10} │ {len(df_comp):>9,} │ {r2:>9.4f} │ ${mae:>10,.2f} │ ${min_cost:>5,.0f}-${max_cost:>5,.0f} ║")

print(f"╚══════════════════════════════════════════════════════════════════╝")

# ============================================================
# STEP 7: ROUTER ACCURACY ON TEST SET
# ============================================================
print("\n" + "="*70)
print("STEP 7: ROUTER ACCURACY ON TEST SET")
print("="*70)

router_correct = (df_test['ACTUAL_COMPLEXITY'] == df_test['PREDICTED_COMPLEXITY']).sum()
router_total = len(df_test)
router_accuracy = router_correct / router_total

print(f"\nOverall Router Accuracy: {router_accuracy*100:.2f}%")
print(f"  Correct: {router_correct:,} / {router_total:,}")

print(f"\nPer-Class Router Accuracy:")
for complexity in ['LOW', 'MED', 'HIGH']:
    df_comp = df_test[df_test['ACTUAL_COMPLEXITY'] == complexity]
    correct = (df_comp['ACTUAL_COMPLEXITY'] == df_comp['PREDICTED_COMPLEXITY']).sum()
    acc = correct / len(df_comp) if len(df_comp) > 0 else 0
    print(f"  {complexity}: {acc*100:.2f}% ({correct:,}/{len(df_comp):,})")

# ============================================================
# STEP 8: COST BUCKET ANALYSIS ON TEST SET
# ============================================================
print("\n" + "="*70)
print("STEP 8: COST BUCKET ANALYSIS (TEST SET)")
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
    df_bucket = df_test[(df_test['ACTUAL_TOTAL'] >= low) & (df_test['ACTUAL_TOTAL'] < high)]
    
    if len(df_bucket) > 1:
        r2 = r2_score(df_bucket['ACTUAL_TOTAL'], df_bucket['PREDICTED_TOTAL'])
        mae = mean_absolute_error(df_bucket['ACTUAL_TOTAL'], df_bucket['PREDICTED_TOTAL'])
        avg_pct_error = (df_bucket['ABS_ERROR'] / df_bucket['ACTUAL_TOTAL']).mean() * 100
        
        bucket_metrics.append({
            'bucket': label,
            'count': len(df_bucket),
            'r2': r2,
            'mae': mae,
            'avg_pct_error': avg_pct_error
        })
        
        print(f"{label:<20} {len(df_bucket):>10,} {r2:>10.4f} ${mae:>10,.2f} {avg_pct_error:>11.1f}%")

# ============================================================
# STEP 9: COMPARE TRAIN VS TEST PERFORMANCE
# ============================================================
print("\n" + "="*70)
print("STEP 9: TRAIN VS TEST COMPARISON")
print("="*70)

# Calculate train metrics
train_r2 = r2_score(df_train['ACTUAL_TOTAL'], df_train['PREDICTED_TOTAL'])
train_mae = mean_absolute_error(df_train['ACTUAL_TOTAL'], df_train['PREDICTED_TOTAL'])
train_rmse = np.sqrt(mean_squared_error(df_train['ACTUAL_TOTAL'], df_train['PREDICTED_TOTAL']))

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                   TRAIN vs TEST COMPARISON                        ║
╠══════════════════════════════════════════════════════════════════╣
║                 │     TRAIN        │      TEST        │   DIFF   ║
╠─────────────────┼──────────────────┼──────────────────┼──────────╣
║  Samples        │ {len(df_train):>16,} │ {len(df_test):>16,} │          ║
║  Claims         │ {df_train['CLAIM_ID'].nunique():>16,} │ {df_test['CLAIM_ID'].nunique():>16,} │          ║
║  R²             │ {train_r2:>16.4f} │ {test_r2:>16.4f} │ {test_r2-train_r2:>+8.4f} ║
║  MAE            │ ${train_mae:>15,.2f} │ ${test_mae:>15,.2f} │ ${test_mae-train_mae:>+7,.0f} ║
║  RMSE           │ ${train_rmse:>15,.2f} │ ${test_rmse:>15,.2f} │ ${test_rmse-train_rmse:>+7,.0f} ║
╚══════════════════════════════════════════════════════════════════╝
""")

# Check for overfitting
r2_diff = train_r2 - test_r2
if r2_diff > 0.1:
    print("⚠️  WARNING: Possible overfitting (Train R² >> Test R²)")
elif r2_diff < -0.05:
    print("✅ Good: Test performance matches or exceeds training")
else:
    print("✅ Good: Train and Test performance are similar (no significant overfitting)")

# ============================================================
# STEP 10: SAMPLE PREDICTIONS FROM TEST SET
# ============================================================
print("\n" + "="*70)
print("STEP 10: SAMPLE PREDICTIONS FROM TEST SET")
print("="*70)

sample_criteria = {
    'LOW': [
        {'name': 'Small', 'min': 0, 'max': 200},
        {'name': 'Medium', 'min': 200, 'max': 1000},
        {'name': 'Large', 'min': 1000, 'max': 5000},
    ],
    'MED': [
        {'name': 'Small', 'min': 0, 'max': 1000},
        {'name': 'Medium', 'min': 1000, 'max': 5000},
        {'name': 'Large', 'min': 5000, 'max': 20000},
    ],
    'HIGH': [
        {'name': 'Small', 'min': 0, 'max': 5000},
        {'name': 'Medium', 'min': 5000, 'max': 20000},
        {'name': 'Large', 'min': 20000, 'max': 50000},
    ],
}

showcase_samples = []

for complexity in ['LOW', 'MED', 'HIGH']:
    df_comp = df_test[df_test['ACTUAL_COMPLEXITY'] == complexity]
    
    for criteria in sample_criteria[complexity]:
        df_range = df_comp[
            (df_comp['ACTUAL_TOTAL'] >= criteria['min']) & 
            (df_comp['ACTUAL_TOTAL'] < criteria['max'])
        ]
        
        if len(df_range) == 0:
            continue
        
        unique_claims = df_range['CLAIM_ID'].unique()
        
        if len(unique_claims) == 0:
            continue
        
        np.random.seed(42)
        selected_claim = np.random.choice(unique_claims)
        
        claim_data = df_range[df_range['CLAIM_ID'] == selected_claim]
        final_row = claim_data[claim_data['SEQ_LENGTH'] == claim_data['SEQ_LENGTH'].max()].iloc[0]
        
        error = final_row['PREDICTED_TOTAL'] - final_row['ACTUAL_TOTAL']
        pct_error = (error / final_row['ACTUAL_TOTAL']) * 100 if final_row['ACTUAL_TOTAL'] > 0 else 0
        
        showcase_samples.append({
            'Complexity': complexity,
            'Cost Range': criteria['name'],
            'Claim ID': selected_claim,
            'Visits': int(final_row['TOTAL_VISITS']),
            'Actual ($)': round(final_row['ACTUAL_TOTAL'], 2),
            'Predicted ($)': round(final_row['PREDICTED_TOTAL'], 2),
            'Error ($)': round(error, 2),
            'Error (%)': round(pct_error, 1),
            'Routed To': final_row['PREDICTED_COMPLEXITY'],
            'Routing Correct': 'Yes' if final_row['ACTUAL_COMPLEXITY'] == final_row['PREDICTED_COMPLEXITY'] else 'No',
        })

df_showcase = pd.DataFrame(showcase_samples)

print(f"""
╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════╗
║                                    SAMPLE TEST PREDICTIONS                                                   ║
╠═══════════╤═══════════╤══════════════╤════════╤═══════════════╤═══════════════╤════════════╤════════════════╣
║ Complexity│ Range     │ Claim ID     │ Visits │ Actual ($)    │ Predicted ($) │ Error ($)  │ Routing        ║
╠═══════════╪═══════════╪══════════════╪════════╪═══════════════╪═══════════════╪════════════╪════════════════╣""")

for _, row in df_showcase.iterrows():
    routing_status = "✅ Correct" if row['Routing Correct'] == 'Yes' else f"❌ → {row['Routed To']}"
    print(f"║ {row['Complexity']:<9} │ {row['Cost Range']:<9} │ {str(row['Claim ID']):<12} │ {row['Visits']:>6} │ ${row['Actual ($)']:>11,.2f} │ ${row['Predicted ($)']:>11,.2f} │ ${row['Error ($)']:>8,.0f} │ {routing_status:<14} ║")

print(f"╚═══════════╧═══════════╧══════════════╧════════╧═══════════════╧═══════════════╧════════════╧════════════════╝")

# ============================================================
# STEP 11: SAVE TEST RESULTS
# ============================================================
print("\n" + "="*70)
print("STEP 11: SAVE TEST RESULTS")
print("="*70)

# Save full test predictions
df_test.to_csv(OUTPUT_PATH / 'test_predictions.csv', index=False)
print(f"✓ Saved: {OUTPUT_PATH / 'test_predictions.csv'}")

# Save showcase samples
df_showcase.to_csv(OUTPUT_PATH / 'test_showcase_samples.csv', index=False)
print(f"✓ Saved: {OUTPUT_PATH / 'test_showcase_samples.csv'}")

# Save test metrics
test_metrics = {
    'overall': {
        'r2': test_r2,
        'mae': test_mae,
        'rmse': test_rmse,
        'n_samples': len(df_test),
        'n_claims': df_test['CLAIM_ID'].nunique()
    },
    'per_complexity': test_complexity_metrics,
    'router_accuracy': router_accuracy,
    'cost_buckets': bucket_metrics,
    'train_vs_test': {
        'train_r2': train_r2,
        'test_r2': test_r2,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'r2_diff': train_r2 - test_r2,
        'mae_diff': test_mae - train_mae
    }
}

with open(OUTPUT_PATH / 'test_metrics.pkl', 'wb') as f:
    pickle.dump(test_metrics, f)
print(f"✓ Saved: {OUTPUT_PATH / 'test_metrics.pkl'}")

# Save to Excel
excel_path = OUTPUT_PATH / 'full_test_results.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    df_test.to_excel(writer, sheet_name='All Test Predictions', index=False)
    df_showcase.to_excel(writer, sheet_name='Sample Claims', index=False)
    
    # Summary sheet
    summary_data = {
        'Metric': ['R²', 'MAE', 'RMSE', 'Samples', 'Claims', 'Router Accuracy'],
        'Train': [train_r2, train_mae, train_rmse, len(df_train), df_train['CLAIM_ID'].nunique(), '-'],
        'Test': [test_r2, test_mae, test_rmse, len(df_test), df_test['CLAIM_ID'].nunique(), f"{router_accuracy*100:.2f}%"],
    }
    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)
    
print(f"✓ Saved: {excel_path}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "="*70)
print("FULL TEST RESULTS SUMMARY")
print("="*70)

print(f"""
╔══════════════════════════════════════════════════════════════════╗
║                    FULL TEST RESULTS SUMMARY                      ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  TEST SET (UNSEEN DATA):                                          ║
║    Samples:         {len(df_test):>10,}                                  ║
║    Claims:          {df_test['CLAIM_ID'].nunique():>10,}                                  ║
║                                                                   ║
║  TEST PERFORMANCE:                                                ║
║    R² Score:        {test_r2:>10.4f} ({test_r2*100:.2f}%)                       ║
║    MAE:             ${test_mae:>10,.2f}                                 ║
║    RMSE:            ${test_rmse:>10,.2f}                                 ║
║                                                                   ║
║  ROUTER ACCURACY:   {router_accuracy*100:>10.2f}%                               ║
║                                                                   ║
║  PER-COMPLEXITY TEST R²:                                          ║
║    LOW:             {test_complexity_metrics.get('LOW', {}).get('r2', 0):>10.4f}                                  ║
║    MED:             {test_complexity_metrics.get('MED', {}).get('r2', 0):>10.4f}                                  ║
║    HIGH:            {test_complexity_metrics.get('HIGH', {}).get('r2', 0):>10.4f}                                  ║
║                                                                   ║
║  OVERFITTING CHECK:                                               ║
║    Train R²:        {train_r2:>10.4f}                                  ║
║    Test R²:         {test_r2:>10.4f}                                  ║
║    Difference:      {train_r2 - test_r2:>+10.4f}                                  ║
║    Status:          {'⚠️ Overfitting' if (train_r2 - test_r2) > 0.1 else '✅ Good'}                                 ║
║                                                                   ║
╚══════════════════════════════════════════════════════════════════╝

FILES SAVED:
  1. {OUTPUT_PATH / 'test_predictions.csv'}
  2. {OUTPUT_PATH / 'test_showcase_samples.csv'}
  3. {OUTPUT_PATH / 'test_metrics.pkl'}
  4. {OUTPUT_PATH / 'full_test_results.xlsx'}
""")
