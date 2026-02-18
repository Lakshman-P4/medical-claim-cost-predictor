# ============================================================
# PREPROCESSING PIPELINE
# ============================================================
# This script:
# 1. Loads raw data
# 2. Removes unnecessary columns
# 3. Handles missing values
# 4. Creates derived features (ICD, Age groups, Visit features)
# 5. Removes MEDICAL_PAYMENT_TOTAL (unreliable)
# 6. Calculates TOTAL_CLAIM_COST from sum of medical_amount per claim
# 7. Filters: removes claims with total cost <= 0 or > 50,000
# 8. Saves preprocessed data
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

print("="*60)
print("PREPROCESSING PIPELINE")
print("="*60)

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_DIR = Path(".")
DATA_RAW = PROJECT_DIR / "data_raw" / "cleaned_dataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data_preprocessed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("\n" + "="*60)
print("STEP 1: LOAD DATA")
print("="*60)

df = pd.read_csv(DATA_RAW)
print(f"Loaded: {len(df):,} rows, {df['CLAIM_ID'].nunique():,} unique claims")
print(f"Columns: {df.columns.tolist()}")

# ============================================================
# STEP 2: DROP UNNECESSARY COLUMNS
# ============================================================
print("\n" + "="*60)
print("STEP 2: DROP UNNECESSARY COLUMNS")
print("="*60)

drop_cols = [
    "CLAIM_NUMBER", 
    "cost_trend_strength",
    "cost_volatility", 
    "avg_daily_cost_change", 
    "inflation_adjusted_amount",
    "source", 
    "icd_codes_per_visit",
    "MEDICAL_PAYMENT_TOTAL"  # Removing - we'll calculate from medical_amount
]

existing_to_drop = [c for c in drop_cols if c in df.columns]
df = df.drop(columns=existing_to_drop)
print(f"Dropped columns: {existing_to_drop}")
print(f"Shape after drop: {df.shape}")

# ============================================================
# STEP 3: HANDLE ICD CODES
# ============================================================
print("\n" + "="*60)
print("STEP 3: HANDLE ICD CODES")
print("="*60)

# Create HAS_ICD flag
df['HAS_ICD'] = df['icd_code_comb'].notna().astype(int)
print(f"HAS_ICD distribution:\n{df['HAS_ICD'].value_counts()}")

# Extract primary ICD code
def extract_primary_icd(icd_str):
    if pd.isna(icd_str):
        return 'NONE'
    codes = str(icd_str).split(',')
    return codes[0].strip() if codes else 'NONE'

df['ICD_PRIMARY'] = df['icd_code_comb'].apply(extract_primary_icd)
print(f"ICD_PRIMARY unique values: {df['ICD_PRIMARY'].nunique()}")

# Fill missing unique_icd_codes_count
df['unique_icd_codes_count'] = df['unique_icd_codes_count'].fillna(0).astype(int)

# ============================================================
# STEP 4: HANDLE AGE WITH MODEL IMPUTATION (CLAIM LEVEL)
# ============================================================
print("\n" + "="*60)
print("STEP 4: HANDLE AGE WITH MODEL IMPUTATION (CLAIM LEVEL)")
print("="*60)

# --- Age Group Boundaries (from data analysis) ---
AGE_GROUP_BOUNDARIES = {
    'Young_Adult': {'min': 18, 'max': 25, 'median': 21},
    'Adult': {'min': 26, 'max': 40, 'median': 38},
    'Middle_Aged': {'min': 41, 'max': 55, 'median': 51},
    'Senior_Citizen': {'min': 56, 'max': 125, 'median': 71},
}

def get_age_group_from_age(age):
    """Derive AGE_GROUP from AGE based on boundaries"""
    if pd.isna(age):
        return None
    age = int(age)
    if 18 <= age <= 25:
        return 'Young_Adult'
    elif 26 <= age <= 40:
        return 'Adult'
    elif 41 <= age <= 55:
        return 'Middle_Aged'
    elif age >= 56:
        return 'Senior_Citizen'
    else:
        return None  # Age < 18

def get_age_from_age_group(age_group):
    """Derive AGE from AGE_GROUP using median (rounded)"""
    if pd.isna(age_group) or age_group not in AGE_GROUP_BOUNDARIES:
        return None
    return int(round(AGE_GROUP_BOUNDARIES[age_group]['median']))

# --- Step 4a: Get ONE age/age_group per CLAIM_ID ---
print("\n--- Step 4a: Extract Claim-Level Age Info ---")

# For each claim, get the first non-null AGE and AGE_GROUP
claim_age_info = df.groupby('CLAIM_ID').agg(
    AGE_FIRST=('AGE', 'first'),
    AGE_GROUP_FIRST=('AGE_GROUP', 'first')
).reset_index()

# Also try to get any non-null value (in case first is null but others aren't)
claim_age_any = df.groupby('CLAIM_ID').agg(
    AGE_ANY=('AGE', lambda x: x.dropna().iloc[0] if x.dropna().any() else None),
    AGE_GROUP_ANY=('AGE_GROUP', lambda x: x.dropna().iloc[0] if x.dropna().any() else None)
).reset_index()

# Merge and use ANY if FIRST is null
claim_age_info = claim_age_info.merge(claim_age_any, on='CLAIM_ID')
claim_age_info['AGE_CLAIM'] = claim_age_info['AGE_FIRST'].fillna(claim_age_info['AGE_ANY'])
claim_age_info['AGE_GROUP_CLAIM'] = claim_age_info['AGE_GROUP_FIRST'].fillna(claim_age_info['AGE_GROUP_ANY'])
claim_age_info = claim_age_info[['CLAIM_ID', 'AGE_CLAIM', 'AGE_GROUP_CLAIM']]

print(f"Total claims: {len(claim_age_info):,}")
print(f"Claims with AGE: {claim_age_info['AGE_CLAIM'].notna().sum():,}")
print(f"Claims with AGE_GROUP: {claim_age_info['AGE_GROUP_CLAIM'].notna().sum():,}")

# --- Step 4b: For claims with AGE but missing AGE_GROUP, derive AGE_GROUP ---
print("\n--- Step 4b: Derive AGE_GROUP from AGE ---")
mask_has_age_no_group = claim_age_info['AGE_CLAIM'].notna() & claim_age_info['AGE_GROUP_CLAIM'].isna()
claim_age_info.loc[mask_has_age_no_group, 'AGE_GROUP_CLAIM'] = claim_age_info.loc[mask_has_age_no_group, 'AGE_CLAIM'].apply(get_age_group_from_age)
print(f"Derived AGE_GROUP for {mask_has_age_no_group.sum():,} claims")

# --- Step 4c: For claims with AGE_GROUP but missing AGE, derive AGE ---
print("\n--- Step 4c: Derive AGE from AGE_GROUP median ---")
mask_has_group_no_age = claim_age_info['AGE_CLAIM'].isna() & claim_age_info['AGE_GROUP_CLAIM'].notna()
claim_age_info.loc[mask_has_group_no_age, 'AGE_CLAIM'] = claim_age_info.loc[mask_has_group_no_age, 'AGE_GROUP_CLAIM'].apply(get_age_from_age_group)
print(f"Derived AGE for {mask_has_group_no_age.sum():,} claims")

# --- Step 4d: For claims missing BOTH, use model imputation ---
print("\n--- Step 4d: Model Imputation for Claims Missing Both ---")

mask_missing_both = claim_age_info['AGE_CLAIM'].isna() & claim_age_info['AGE_GROUP_CLAIM'].isna()
n_missing_both = mask_missing_both.sum()

if n_missing_both > 0:
    print(f"Claims missing both AGE and AGE_GROUP: {n_missing_both:,}")
    
    # Get claim-level features for imputation (from first visit of each claim)
    claim_features = df.groupby('CLAIM_ID').first().reset_index()
    
    # Features for imputation
    impute_features = [
        'GENDER', 'BODY_PART_DESC', 'BODY_PART_GROUP_DESC',
        'NATURE_OF_INJURY_DESC', 'INCIDENT_STATE', 'CLAIM_CAUSE_GROUP_DESC',
        'CLAIMANT_TYPE_DESC'
    ]
    
    # Merge age info with claim features
    claim_features = claim_features.merge(claim_age_info[['CLAIM_ID', 'AGE_GROUP_CLAIM']], on='CLAIM_ID', how='left')
    
    # Prepare data for model
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    
    # Split into known and unknown
    df_known = claim_features[claim_features['AGE_GROUP_CLAIM'].notna()].copy()
    df_unknown = claim_features[claim_features['AGE_GROUP_CLAIM'].isna()].copy()
    
    print(f"Training data: {len(df_known):,} claims")
    print(f"To impute: {len(df_unknown):,} claims")
    
    # Encode features
    encoders = {}
    X_train = pd.DataFrame()
    X_unknown = pd.DataFrame()
    
    for col in impute_features:
        if col in claim_features.columns:
            le = LabelEncoder()
            # Fit on all data to handle all categories
            all_values = claim_features[col].fillna('MISSING').astype(str)
            le.fit(all_values)
            encoders[col] = le
            
            X_train[col] = le.transform(df_known[col].fillna('MISSING').astype(str))
            X_unknown[col] = le.transform(df_unknown[col].fillna('MISSING').astype(str))
    
    # Target
    y_train = df_known['AGE_GROUP_CLAIM']
    
    # Train model
    print("Training RandomForest for AGE_GROUP imputation...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    # Predict missing AGE_GROUP
    predicted_age_groups = rf_model.predict(X_unknown)
    
    # Update claim_age_info with predictions
    unknown_claim_ids = df_unknown['CLAIM_ID'].values
    for claim_id, pred_group in zip(unknown_claim_ids, predicted_age_groups):
        mask = claim_age_info['CLAIM_ID'] == claim_id
        claim_age_info.loc[mask, 'AGE_GROUP_CLAIM'] = pred_group
        claim_age_info.loc[mask, 'AGE_CLAIM'] = get_age_from_age_group(pred_group)
    
    print(f"Imputed AGE_GROUP and derived AGE for {n_missing_both:,} claims")
    
    # Show distribution of imputed values
    print(f"\nImputed AGE_GROUP distribution:")
    imputed_distribution = pd.Series(predicted_age_groups).value_counts()
    print(imputed_distribution)
else:
    print("No claims missing both AGE and AGE_GROUP")

# --- Step 4e: Apply claim-level age info back to all visits ---
print("\n--- Step 4e: Apply Claim-Level Age to All Visits ---")

# Rename columns for final
claim_age_info = claim_age_info.rename(columns={
    'AGE_CLAIM': 'AGE_FINAL',
    'AGE_GROUP_CLAIM': 'AGE_GROUP_FINAL'
})

# Drop original AGE and AGE_GROUP from df
if 'AGE' in df.columns:
    df = df.drop(columns=['AGE'])
if 'AGE_GROUP' in df.columns:
    df = df.drop(columns=['AGE_GROUP'])

# Merge claim-level age info to all visits
df = df.merge(claim_age_info[['CLAIM_ID', 'AGE_FINAL', 'AGE_GROUP_FINAL']], on='CLAIM_ID', how='left')

# --- Verify ---
print(f"\n--- After Imputation ---")
print(f"Missing AGE_FINAL: {df['AGE_FINAL'].isna().sum()}")
print(f"Missing AGE_GROUP_FINAL: {df['AGE_GROUP_FINAL'].isna().sum()}")

# Verify same age per claim
age_per_claim = df.groupby('CLAIM_ID')['AGE_FINAL'].nunique()
claims_with_multiple_ages = (age_per_claim > 1).sum()
print(f"\nClaims with multiple AGE values: {claims_with_multiple_ages} (should be 0)")

print(f"\nAGE_GROUP_FINAL distribution:")
print(df['AGE_GROUP_FINAL'].value_counts())

print(f"\nAGE_FINAL stats:")
print(df['AGE_FINAL'].describe())

# ============================================================
# STEP 5: FILL MISSING CATEGORICALS
# ============================================================
print("\n" + "="*60)
print("STEP 5: FILL MISSING CATEGORICALS")
print("="*60)

categorical_cols = [
    'CLAIM_CAUSE_GROUP_DESC',
    'BODY_PART_DESC',
    'BODY_PART_GROUP_DESC',
    'NATURE_OF_INJURY_DESC',
    'INCIDENT_STATE',
    'GENDER',
    'CLAIMANT_TYPE_DESC'
]

for col in categorical_cols:
    if col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            df[col] = df[col].fillna('MISSING')
            print(f"{col}: {missing_count} missing → filled with 'MISSING'")
        else:
            print(f"{col}: No missing values")

# ============================================================
# STEP 6: CREATE VISIT FEATURES
# ============================================================
print("\n" + "="*60)
print("STEP 6: CREATE VISIT FEATURES")
print("="*60)

# Sort by claim and visit number
df = df.sort_values(['CLAIM_ID', 'NO_OF_VISIT']).reset_index(drop=True)

# Calculate visit features per claim
visit_features = df.groupby('CLAIM_ID').agg(
    TOTAL_VISITS_TRUE=('CLAIM_ID', 'count'),
    TOTAL_PAID_VISITS=('medical_amount', lambda x: (x > 0).sum()),
    TOTAL_ZERO_VISITS=('medical_amount', lambda x: (x == 0).sum()),
    TOTAL_NEGATIVE_VISITS=('medical_amount', lambda x: (x < 0).sum()),
).reset_index()

visit_features['TOTAL_ZERO_NEG_VISITS'] = visit_features['TOTAL_ZERO_VISITS'] + visit_features['TOTAL_NEGATIVE_VISITS']
visit_features['PAID_VISIT_RATIO'] = visit_features['TOTAL_PAID_VISITS'] / visit_features['TOTAL_VISITS_TRUE']

print(f"Visit features calculated for {len(visit_features):,} claims")

# Merge back to main dataframe
df = df.merge(visit_features, on='CLAIM_ID', how='left')
print(f"Shape after merge: {df.shape}")

# ============================================================
# STEP 7: FILTER VISITS & CALCULATE TOTAL CLAIM COST
# ============================================================
print("\n" + "="*60)
print("STEP 7: FILTER VISITS & CALCULATE TOTAL CLAIM COST")
print("="*60)

# --- First: Remove all rows where medical_amount <= 0 ---
print(f"\n--- Filtering Visit Rows ---")
print(f"Rows before filtering: {len(df):,}")

rows_zero = (df['medical_amount'] == 0).sum()
rows_negative = (df['medical_amount'] < 0).sum()
print(f"Rows with medical_amount = 0: {rows_zero:,}")
print(f"Rows with medical_amount < 0: {rows_negative:,}")

# Keep only rows where medical_amount > 0
df = df[df['medical_amount'] > 0].copy()
print(f"Rows after filtering (medical_amount > 0): {len(df):,}")
print(f"Unique claims after filtering: {df['CLAIM_ID'].nunique():,}")

# --- Now: Calculate TOTAL_CLAIM_COST per claim (sum of positive medical_amount) ---
claim_totals = df.groupby('CLAIM_ID')['medical_amount'].sum().reset_index()
claim_totals.columns = ['CLAIM_ID', 'TOTAL_CLAIM_COST']

print(f"\n--- Before Filtering ---")
print(f"Total claims: {len(claim_totals):,}")
print(f"Min cost:     ${claim_totals['TOTAL_CLAIM_COST'].min():,.2f}")
print(f"Max cost:     ${claim_totals['TOTAL_CLAIM_COST'].max():,.2f}")
print(f"Mean cost:    ${claim_totals['TOTAL_CLAIM_COST'].mean():,.2f}")
print(f"Median cost:  ${claim_totals['TOTAL_CLAIM_COST'].median():,.2f}")

# Count problematic claims
negative_zero_claims = (claim_totals['TOTAL_CLAIM_COST'] <= 0).sum()
over_50k_claims = (claim_totals['TOTAL_CLAIM_COST'] > 50000).sum()
print(f"\nClaims with total cost <= $0: {negative_zero_claims:,}")
print(f"Claims with total cost > $50,000: {over_50k_claims:,}")

# Filter: Keep only claims with 0 < total <= 50,000
valid_claims = claim_totals[
    (claim_totals['TOTAL_CLAIM_COST'] > 0) & 
    (claim_totals['TOTAL_CLAIM_COST'] <= 50000)
]

print(f"\n--- After Filtering ---")
print(f"Valid claims: {len(valid_claims):,}")
print(f"Removed: {len(claim_totals) - len(valid_claims):,} claims")

# Merge TOTAL_CLAIM_COST to main dataframe
df = df.merge(valid_claims, on='CLAIM_ID', how='inner')

print(f"\n--- Final Dataset ---")
print(f"Rows: {len(df):,}")
print(f"Unique claims: {df['CLAIM_ID'].nunique():,}")

# Verify cost distribution
print(f"\nTOTAL_CLAIM_COST stats:")
print(f"  Min:    ${df.groupby('CLAIM_ID')['TOTAL_CLAIM_COST'].first().min():,.2f}")
print(f"  Max:    ${df.groupby('CLAIM_ID')['TOTAL_CLAIM_COST'].first().max():,.2f}")
print(f"  Mean:   ${df.groupby('CLAIM_ID')['TOTAL_CLAIM_COST'].first().mean():,.2f}")
print(f"  Median: ${df.groupby('CLAIM_ID')['TOTAL_CLAIM_COST'].first().median():,.2f}")

# ============================================================
# STEP 8: FINAL COLUMN CHECK
# ============================================================
print("\n" + "="*60)
print("STEP 8: FINAL COLUMN CHECK")
print("="*60)

print(f"\nTotal columns: {len(df.columns)}")
print(f"\nColumns:")
for i, col in enumerate(df.columns):
    dtype = df[col].dtype
    print(f"  {i+1}. {col} ({dtype})")

# Verify no missing values in critical columns
print("\n--- Missing Value Check ---")
critical_cols = ['CLAIM_ID', 'medical_amount', 'TOTAL_CLAIM_COST', 'NO_OF_VISIT']
for col in critical_cols:
    missing = df[col].isna().sum()
    print(f"  {col}: {missing} missing")

# ============================================================
# STEP 9: SAVE PREPROCESSED DATA
# ============================================================
print("\n" + "="*60)
print("STEP 9: SAVE PREPROCESSED DATA")
print("="*60)

output_file = OUTPUT_DIR / "df_preprocessed.csv"
df.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   Shape: {df.shape}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)

print("")
print("Summary:")
print("--------")
print(f"Input:  {DATA_RAW}")
print(f"Output: {output_file}")
print("")
print("Dataset:")
print(f"  - Rows: {len(df):,}")
print(f"  - Unique claims: {df['CLAIM_ID'].nunique():,}")
print(f"  - Columns: {len(df.columns)}")
print("")
print("Target Variable:")
print("  - TOTAL_CLAIM_COST (sum of medical_amount per claim)")
print("  - Range: $0 < cost <= $50,000")
print("")
print("Key Features:")
print("  - Visit features (TOTAL_VISITS_TRUE, PAID_VISIT_RATIO, etc.)")
print("  - ICD features (HAS_ICD, ICD_PRIMARY, unique_icd_codes_count)")
print("  - Demographics (AGE_FINAL, AGE_GROUP_FINAL, GENDER)")
print("  - Claim characteristics (BODY_PART_DESC, NATURE_OF_INJURY_DESC, etc.)")
print("")
print("Next Steps:")
print("  1. Run classify_complexity.py to create complexity groups")
print("  2. Run regress_low.py, regress_med.py, regress_high.py to train models")
print("  3. Run inference_pipeline.py to make predictions")