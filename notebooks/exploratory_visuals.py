#RICH HISTORIC DATA VISUALIZATION
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

print("="*70)
print("RICH HISTORIC DATA VISUALIZATION")
print("="*70)

# CONFIGURATION
DATA_PATH = Path("data_preprocessed")
OUTPUT_PATH = Path("artifacts/visualizations")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

# LOAD DATA
print("\nLoading data...")

df = pd.read_csv(DATA_PATH / "df_preprocessed.csv")
df_claims = pd.read_csv(DATA_PATH / "df_claims_classified.csv")

print(f"Total Visits: {len(df):,}")
print(f"Total Claims: {len(df_claims):,}")

# KEY STATISTICS
print("\n" + "="*70)
print("KEY STATISTICS")
print("="*70)

total_visits = len(df)
total_claims = len(df_claims)
avg_visits_per_claim = total_visits / total_claims
total_cost = df_claims['TOTAL_CLAIM_COST'].sum()
avg_cost = df_claims['TOTAL_CLAIM_COST'].mean()
median_cost = df_claims['TOTAL_CLAIM_COST'].median()

complexity_dist = df_claims['COMPLEXITY'].value_counts()

print(f"""
DATA SUMMARY:
  Total Claims:         {total_claims:,}
  Total Visits:         {total_visits:,}
  Avg Visits/Claim:     {avg_visits_per_claim:.1f}
  
  Total Cost:           ${total_cost:,.2f}
  Average Claim Cost:   ${avg_cost:,.2f}
  Median Claim Cost:    ${median_cost:,.2f}
  
  Complexity Distribution:
    LOW:  {complexity_dist.get('LOW', 0):,} ({complexity_dist.get('LOW', 0)/total_claims*100:.1f}%)
    MED:  {complexity_dist.get('MED', 0):,} ({complexity_dist.get('MED', 0)/total_claims*100:.1f}%)
    HIGH: {complexity_dist.get('HIGH', 0):,} ({complexity_dist.get('HIGH', 0)/total_claims*100:.1f}%)
""")

#visual1
print("\nCreating main infographic...")

fig = plt.figure(figsize=(16, 12))
fig.patch.set_facecolor('#f8f9fa')

# Title
fig.suptitle('Rich Historic Data: The Foundation of Accurate Predictions', 
             fontsize=24, fontweight='bold', color='#2c3e50', y=0.98)

# Subtitle
fig.text(0.5, 0.93, 'Workers Compensation Claims Dataset Overview', 
         fontsize=14, ha='center', color='#7f8c8d')

# TOP ROW: BIG NUMBERS
# Create grid
gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3, 
                       left=0.05, right=0.95, top=0.88, bottom=0.08)

# Big stat boxes
stats = [
    (f"{total_claims:,}", "Total Claims", "#3498db", "📋"),
    (f"{total_visits:,}", "Total Visits", "#2ecc71", "🏥"),
    (f"${total_cost/1e6:.1f}M", "Total Value", "#9b59b6", "💰"),
    (f"{avg_visits_per_claim:.1f}", "Avg Visits/Claim", "#e74c3c", "📊"),
]

for i, (value, label, color, icon) in enumerate(stats):
    ax = fig.add_subplot(gs[0, i])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Background box
    rect = mpatches.FancyBboxPatch((0.5, 0.5), 9, 9, 
                                    boxstyle="round,pad=0.1,rounding_size=0.5",
                                    facecolor=color, alpha=0.15, edgecolor=color, linewidth=3)
    ax.add_patch(rect)
    
    # Value
    ax.text(5, 6, value, fontsize=28, fontweight='bold', ha='center', va='center', color=color)
    
    # Label
    ax.text(5, 2.5, label, fontsize=12, ha='center', va='center', color='#2c3e50')
  
# MIDDLE LEFT: Complexity Distribution (Pie)
ax1 = fig.add_subplot(gs[1, :2])

colors_complexity = ['#2ecc71', '#f39c12', '#e74c3c']
sizes = [complexity_dist.get('LOW', 0), complexity_dist.get('MED', 0), complexity_dist.get('HIGH', 0)]
labels = ['LOW\nSimple Claims', 'MED\nModerate Claims', 'HIGH\nComplex Claims']

wedges, texts, autotexts = ax1.pie(sizes, colors=colors_complexity, autopct='%1.1f%%',
                                    startangle=90, explode=(0.02, 0.02, 0.02),
                                    textprops={'fontsize': 11})

for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')

ax1.set_title('Claim Complexity Distribution', fontsize=14, fontweight='bold', pad=10)
ax1.legend(wedges, labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

# MIDDLE RIGHT: Cost Distribution (Histogram)
ax2 = fig.add_subplot(gs[1, 2:])

# Cap at 99th percentile for visualization
cost_cap = np.percentile(df_claims['TOTAL_CLAIM_COST'], 99)
costs_capped = df_claims['TOTAL_CLAIM_COST'].clip(upper=cost_cap)

ax2.hist(costs_capped, bins=50, color='#3498db', edgecolor='white', alpha=0.7)
ax2.axvline(avg_cost, color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: ${avg_cost:,.0f}')
ax2.axvline(median_cost, color='#2ecc71', linestyle='--', linewidth=2, label=f'Median: ${median_cost:,.0f}')

ax2.set_xlabel('Total Claim Cost ($)', fontsize=11)
ax2.set_ylabel('Number of Claims', fontsize=11)
ax2.set_title('Cost Distribution Across Claims', fontsize=14, fontweight='bold', pad=10)
ax2.legend(fontsize=10)
ax2.set_xlim(0, cost_cap)

# Format x-axis
ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# BOTTOM LEFT: Visits per Claim Distribution
ax3 = fig.add_subplot(gs[2, :2])

visits_per_claim = df.groupby('CLAIM_ID').size()
visits_capped = visits_per_claim.clip(upper=30)

ax3.hist(visits_capped, bins=30, color='#9b59b6', edgecolor='white', alpha=0.7)
ax3.axvline(visits_per_claim.mean(), color='#e74c3c', linestyle='--', linewidth=2, 
            label=f'Mean: {visits_per_claim.mean():.1f}')
ax3.axvline(visits_per_claim.median(), color='#2ecc71', linestyle='--', linewidth=2,
            label=f'Median: {visits_per_claim.median():.1f}')

ax3.set_xlabel('Number of Visits per Claim', fontsize=11)
ax3.set_ylabel('Number of Claims', fontsize=11)
ax3.set_title('Visit Frequency Distribution', fontsize=14, fontweight='bold', pad=10)
ax3.legend(fontsize=10)

# BOTTOM RIGHT: Cost by Complexity (Box Plot)
ax4 = fig.add_subplot(gs[2, 2:])

complexity_order = ['LOW', 'MED', 'HIGH']
box_data = [df_claims[df_claims['COMPLEXITY'] == c]['TOTAL_CLAIM_COST'].clip(upper=20000) 
            for c in complexity_order]

bp = ax4.boxplot(box_data, labels=complexity_order, patch_artist=True)

for patch, color in zip(bp['boxes'], colors_complexity):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)

ax4.set_ylabel('Total Claim Cost ($)', fontsize=11)
ax4.set_title('Cost Range by Complexity', fontsize=14, fontweight='bold', pad=10)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))

# SAVE
plt.savefig(OUTPUT_PATH / 'rich_data_infographic.png', dpi=200, bbox_inches='tight',
            facecolor='#f8f9fa', edgecolor='none')
plt.close()
print(f"Saved: {OUTPUT_PATH / 'rich_data_infographic.png'}")

#visual2
print("\nCreating data enabler summary...")

fig2, ax = plt.subplots(figsize=(14, 8))
fig2.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
ax.axis('off')

# Title
ax.text(0.5, 0.92, 'RICH HISTORIC DATA', fontsize=32, fontweight='bold', 
        ha='center', va='center', color='white', transform=ax.transAxes)

ax.text(0.5, 0.84, 'Enabling Accurate Workers Compensation Predictions', 
        fontsize=16, ha='center', va='center', color='#a0a0a0', transform=ax.transAxes)

# Main stats in boxes
box_data = [
    (0.15, 0.55, f"{total_claims:,}", "Claims", "#3498db"),
    (0.40, 0.55, f"{total_visits:,}", "Visits", "#2ecc71"),
    (0.65, 0.55, f"${total_cost/1e6:.1f}M", "Total Value", "#9b59b6"),
    (0.90, 0.55, f"{avg_visits_per_claim:.1f}", "Avg Visits", "#e74c3c"),
]

for x, y, value, label, color in box_data:
    # Box
    rect = mpatches.FancyBboxPatch((x-0.10, y-0.15), 0.18, 0.30, 
                                    transform=ax.transAxes,
                                    boxstyle="round,pad=0.02,rounding_size=0.02",
                                    facecolor=color, alpha=0.2, edgecolor=color, linewidth=2)
    ax.add_patch(rect)
    
    # Value
    ax.text(x, y+0.05, value, fontsize=24, fontweight='bold', 
            ha='center', va='center', color='white', transform=ax.transAxes)
    
    # Label
    ax.text(x, y-0.08, label, fontsize=12, ha='center', va='center', 
            color='#a0a0a0', transform=ax.transAxes)

# Bottom section - Key insights
insights = [
    "✓ 100K+ historical claims for pattern learning",
    "✓ 630K+ visits capturing claim progression",
    "✓ Diverse complexity levels (LOW / MED / HIGH)",
    "✓ Rich features: costs, diagnoses, demographics",
]

for i, insight in enumerate(insights):
    ax.text(0.5, 0.28 - i*0.07, insight, fontsize=14, ha='center', va='center',
            color='white', transform=ax.transAxes)

# Footer
ax.text(0.5, 0.05, 'Data enables LSTM sequence learning → Pattern recognition → Accurate predictions',
        fontsize=12, ha='center', va='center', color='#f39c12', style='italic',
        transform=ax.transAxes)

plt.savefig(OUTPUT_PATH / 'data_enabler_dark.png', dpi=200, bbox_inches='tight',
            facecolor='#1a1a2e', edgecolor='none')
plt.close()
print(f"Saved: {OUTPUT_PATH / 'data_enabler_dark.png'}")

#visual 3
print("\nCreating light version...")

fig3, ax = plt.subplots(figsize=(14, 8))
fig3.patch.set_facecolor('white')
ax.set_facecolor('white')
ax.axis('off')

# Title
ax.text(0.5, 0.92, 'Rich Historic Data: The Enabler', fontsize=28, fontweight='bold', 
        ha='center', va='center', color='#2c3e50', transform=ax.transAxes)

ax.text(0.5, 0.84, 'Foundation for Accurate Workers Compensation Predictions', 
        fontsize=14, ha='center', va='center', color='#7f8c8d', transform=ax.transAxes)

# Main stats
box_data = [
    (0.15, 0.55, f"{total_claims:,}", "Total Claims", "#3498db"),
    (0.40, 0.55, f"{total_visits:,}", "Total Visits", "#2ecc71"),
    (0.65, 0.55, f"${total_cost/1e6:.1f}M", "Total Value", "#9b59b6"),
    (0.90, 0.55, f"{avg_visits_per_claim:.1f}", "Avg Visits/Claim", "#e74c3c"),
]

for x, y, value, label, color in box_data:
    rect = mpatches.FancyBboxPatch((x-0.10, y-0.15), 0.18, 0.30, 
                                    transform=ax.transAxes,
                                    boxstyle="round,pad=0.02,rounding_size=0.02",
                                    facecolor=color, alpha=0.15, edgecolor=color, linewidth=3)
    ax.add_patch(rect)
    
    ax.text(x, y+0.05, value, fontsize=26, fontweight='bold', 
            ha='center', va='center', color=color, transform=ax.transAxes)
    
    ax.text(x, y-0.08, label, fontsize=11, ha='center', va='center', 
            color='#2c3e50', transform=ax.transAxes)

ax.text(0.5, 0.28, 'Claim Complexity Breakdown', fontsize=16, fontweight='bold',
        ha='center', va='center', color='#2c3e50', transform=ax.transAxes)

# Horizontal bar for complexity
bar_y = 0.18
bar_height = 0.06
total = complexity_dist.sum()  # FIXED

low_pct = complexity_dist.get('LOW', 0) / total
med_pct = complexity_dist.get('MED', 0) / total
high_pct = complexity_dist.get('HIGH', 0) / total

# Draw bars
ax.barh([bar_y], [low_pct], height=bar_height, color='#2ecc71', left=0.1, transform=ax.transAxes)
ax.barh([bar_y], [med_pct], height=bar_height, color='#f39c12', left=0.1+low_pct, transform=ax.transAxes)
ax.barh([bar_y], [high_pct], height=bar_height, color='#e74c3c', left=0.1+low_pct+med_pct, transform=ax.transAxes)

# Labels
ax.text(0.1 + low_pct/2, bar_y, f'LOW\n{low_pct*100:.0f}%', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', transform=ax.transAxes)
ax.text(0.1 + low_pct + med_pct/2, bar_y, f'MED\n{med_pct*100:.0f}%', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', transform=ax.transAxes)
ax.text(0.1 + low_pct + med_pct + high_pct/2, bar_y, f'HIGH\n{high_pct*100:.0f}%', fontsize=10, ha='center', va='center',
        color='white', fontweight='bold', transform=ax.transAxes)

# Footer
ax.text(0.5, 0.05, 'More Data → Better Patterns → More Accurate Predictions',
        fontsize=14, ha='center', va='center', color='#3498db', fontweight='bold',
        transform=ax.transAxes)

plt.savefig(OUTPUT_PATH / 'data_enabler_light.png', dpi=200, bbox_inches='tight',
            facecolor='white', edgecolor='none')
plt.close()
print(f"Saved: {OUTPUT_PATH / 'data_enabler_light.png'}")


print("VISUALIZATION COMPLETE")

print(f"""
FILES SAVED:
  1. {OUTPUT_PATH / 'rich_data_infographic.png'} (detailed infographic)
  2. {OUTPUT_PATH / 'data_enabler_dark.png'} (dark theme summary)
  3. {OUTPUT_PATH / 'data_enabler_light.png'} (light theme summary)

USE FOR PRESENTATION:
  - Slide showing data foundation
  - "Why our predictions work" narrative
  - Rich historic data = accurate models
""")
