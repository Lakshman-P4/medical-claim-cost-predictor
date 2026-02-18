# ============================================================
# EXPLORE BODY PART TO BODY PART GROUP MAPPING
# Save as: explore_data.py
# ============================================================

import pandas as pd
from pathlib import Path

print("="*70)
print("BODY PART → BODY PART GROUP MAPPING")
print("="*70)

# Load data
DATA_PATH = Path("data_preprocessed")
df = pd.read_csv(DATA_PATH / "df_preprocessed.csv")

print(f"\nLoaded {len(df):,} records")

# ============================================================
# CREATE MAPPING TABLE
# ============================================================

# Get unique combinations
mapping = df.groupby(['BODY_PART_DESC', 'BODY_PART_GROUP_DESC']).size().reset_index(name='COUNT')

# Sort by group, then by body part
mapping = mapping.sort_values(['BODY_PART_GROUP_DESC', 'BODY_PART_DESC'])

# Print table
print("\n" + "="*100)
print(f"{'BODY PART':<50} {'GROUP':<25} {'COUNT':>10}")
print("="*100)

current_group = None
for _, row in mapping.iterrows():
    if row['BODY_PART_GROUP_DESC'] != current_group:
        current_group = row['BODY_PART_GROUP_DESC']
        print(f"\n--- {current_group} ---")
    
    print(f"{row['BODY_PART_DESC']:<50} {row['BODY_PART_GROUP_DESC']:<25} {row['COUNT']:>10,}")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nUnique Body Parts: {df['BODY_PART_DESC'].nunique()}")
print(f"Unique Body Part Groups: {df['BODY_PART_GROUP_DESC'].nunique()}")

# Group counts
print("\nRecords per Group:")
group_counts = df['BODY_PART_GROUP_DESC'].value_counts()
for group, count in group_counts.items():
    num_parts = mapping[mapping['BODY_PART_GROUP_DESC'] == group]['BODY_PART_DESC'].nunique()
    print(f"  {group}: {count:,} records ({num_parts} body parts)")

# Check for body parts in multiple groups
multi_group = mapping.groupby('BODY_PART_DESC')['BODY_PART_GROUP_DESC'].nunique()
multi_group = multi_group[multi_group > 1]

if len(multi_group) > 0:
    print(f"\n⚠️ Body Parts appearing in MULTIPLE groups: {len(multi_group)}")
    for bp in multi_group.index:
        groups = mapping[mapping['BODY_PART_DESC'] == bp]['BODY_PART_GROUP_DESC'].tolist()
        print(f"  - {bp}: {groups}")
else:
    print("\n✓ Each body part belongs to exactly ONE group")

print("\n" + "="*70)
print("DONE!")
print("="*70)