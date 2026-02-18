# ============================================================
# R² SCORE ANALYSIS - AS REQUESTED BY SAI
# ============================================================

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from pathlib import Path

print("="*60)
print("R² SCORE ANALYSIS")
print("="*60)

DATA_PATH = Path("data_preprocessed")

# Load inference results (with predicted routing)
df = pd.read_csv(DATA_PATH / "inference_results.csv")

print(f"\nLoaded {len(df):,} predictions")

# ============================================================
# OVERALL R² SCORE
# ============================================================
print("\n" + "="*60)
print("OVERALL METRICS")
print("="*60)

r2_overall = r2_score(df['ACTUAL_COST'], df['PREDICTED_COST'])
mae_overall = mean_absolute_error(df['ACTUAL_COST'], df['PREDICTED_COST'])
rmse_overall = np.sqrt(mean_squared_error(df['ACTUAL_COST'], df['PREDICTED_COST']))

print(f"\nR² Score:  {r2_overall:.4f}")
print(f"MAE:       ${mae_overall:,.2f}")
print(f"RMSE:      ${rmse_overall:,.2f}")

# Interpretation
print(f"\n--- INTERPRETATION ---")
if r2_overall > 0.7:
    print("✓ R² > 0.7: Model explains variance well")
elif r2_overall > 0.5:
    print("⚠️ R² 0.5-0.7: Model explains some variance, room for improvement")
elif r2_overall > 0:
    print("⚠️ R² 0-0.5: Model explains little variance, needs improvement")
else:
    print("✗ R² < 0: Model performs worse than predicting the mean!")

# ============================================================
# R² BY COMPLEXITY
# ============================================================
print("\n" + "="*60)
print("R² BY COMPLEXITY")
print("="*60)

print(f"\n{'Complexity':<12} {'R² Score':>12} {'MAE':>12} {'RMSE':>12} {'Avg Cost':>12}")
print("-" * 65)

for complexity in ['LOW', 'MED', 'HIGH']:
    subset = df[df['COMPLEXITY'] == complexity]
    if len(subset) == 0:
        continue
    
    r2 = r2_score(subset['ACTUAL_COST'], subset['PREDICTED_COST'])
    mae = mean_absolute_error(subset['ACTUAL_COST'], subset['PREDICTED_COST'])
    rmse = np.sqrt(mean_squared_error(subset['ACTUAL_COST'], subset['PREDICTED_COST']))
    avg_cost = subset['ACTUAL_COST'].mean()
    
    print(f"{complexity:<12} {r2:>12.4f} ${mae:>10,.2f} ${rmse:>10,.2f} ${avg_cost:>10,.2f}")

# ============================================================
# R² BY COST RANGE
# ============================================================
print("\n" + "="*60)
print("R² BY COST RANGE")
print("="*60)

cost_ranges = [
    (0, 100, "$0 - $100"),
    (100, 250, "$100 - $250"),
    (250, 500, "$250 - $500"),
    (500, 1000, "$500 - $1,000"),
    (1000, 2500, "$1,000 - $2,500"),
    (2500, 5000, "$2,500 - $5,000"),
    (5000, 50000, "$5,000+"),
]

print(f"\n{'Cost Range':<20} {'# Claims':>10} {'R² Score':>12} {'MAE':>12}")
print("-" * 60)

for low, high, label in cost_ranges:
    subset = df[(df['ACTUAL_COST'] >= low) & (df['ACTUAL_COST'] < high)]
    if len(subset) > 10:  # Need enough samples for R²
        r2 = r2_score(subset['ACTUAL_COST'], subset['PREDICTED_COST'])
        mae = mean_absolute_error(subset['ACTUAL_COST'], subset['PREDICTED_COST'])
        print(f"{label:<20} {len(subset):>10,} {r2:>12.4f} ${mae:>10,.2f}")

# ============================================================
# DIAGNOSIS
# ============================================================
print("\n" + "="*60)
print("DIAGNOSIS FOR SAI")
print("="*60)

print(f"""
Results:
--------
Overall R² = {r2_overall:.4f}

What this means:
- R² of {r2_overall:.2f} means the model explains {r2_overall*100:.1f}% of variance
- Remaining {(1-r2_overall)*100:.1f}% is unexplained (error)

If R² is low (< 0.5):
- Model is NOT capturing the relationship well
- Algorithm may need changing
- Features may not be predictive enough
- First-visit data may simply not predict total cost well

Sai's concern:
- MAE not improving across LOW→MED→HIGH
- This could indicate the model is predicting similar values regardless of complexity
- Need to check if predictions vary appropriately with actual costs
""")

# ============================================================
# PREDICTION VARIANCE CHECK
# ============================================================
print("\n" + "="*60)
print("PREDICTION VARIANCE CHECK")
print("="*60)

print(f"\n{'Complexity':<12} {'Actual Mean':>14} {'Actual Std':>14} {'Pred Mean':>14} {'Pred Std':>14}")
print("-" * 75)

for complexity in ['LOW', 'MED', 'HIGH']:
    subset = df[df['COMPLEXITY'] == complexity]
    if len(subset) == 0:
        continue
    
    actual_mean = subset['ACTUAL_COST'].mean()
    actual_std = subset['ACTUAL_COST'].std()
    pred_mean = subset['PREDICTED_COST'].mean()
    pred_std = subset['PREDICTED_COST'].std()
    
    print(f"{complexity:<12} ${actual_mean:>12,.2f} ${actual_std:>12,.2f} ${pred_mean:>12,.2f} ${pred_std:>12,.2f}")

print(f"""
If Pred Std << Actual Std:
  → Model is NOT varying predictions enough
  → Predicting similar values for all claims
  → This is a major issue

If Pred Mean ≠ Actual Mean:
  → Model has systematic bias (over/under predicting)
""")