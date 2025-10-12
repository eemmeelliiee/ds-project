"""
Data Exploration and Visualization
All visualizations are proportional, ordered, labeled, and color-coded (Seaborn ≥ 0.14)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import matplotlib.patches as mpatches

# -------------------------------------------------------------------
# Load and validate data
# -------------------------------------------------------------------
def ReadPreviousData(location: str):
    Filepath = Path(location)
    if Filepath.exists():
        data = pd.read_csv(location)
        return data, True
    else:
        data = pd.DataFrame()
        return data, False

df, df_exists = ReadPreviousData('data/water_quality_cols_complete_rows_rel_cols_all_num_less_corr.csv')

if not df_exists:
    sys.exit("ERROR: Not all required data found.")
print("All files found")

# -------------------------------------------------------------------
# Class order, labels, and color palette
# -------------------------------------------------------------------
class_order = [1, 2, 0]
class_labels = {1: 'A', 2: 'C', 0: 'Other'}
x_tick_labels = [class_labels[i] for i in class_order]

# Color-blind–friendly blue → light blue → red
palette_custom = {
    1: "#364B9A",  # Blue  → Class A (Good)
    2: "#6EA6CD",  # Light Blue → Class C (Moderate)
    0: "#DD3D2D"   # Red   → Class Other (Poor)
}

# small helper to annotate n above stacked bars
def annotate_counts_above_bars(ax, n_series, y_offset=0.08, fmt="n={n}"):
    """
    Add n-labels clearly above stacked bars.
    - y_offset: controls how far above 1.0 the labels appear (as a proportion)
    - Expands ylim to keep labels inside the visible area
    """
    for i, (cat, n) in enumerate(n_series.items()):
        ax.text(
            i,
            1 + y_offset,               # place well above top of bar
            fmt.format(n=int(n)),
            ha='center',
            va='bottom',
            fontsize=9
        )
    ax.set_ylim(0, 1 + y_offset + 0.1)  # adds generous headroom

# -------------------------------------------------------------------
# Figure 1 – Distribution of Samples by Water Quality Class (Count)
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))

# Count samples per water quality class
count_df = (
    df['water_quality']
    .value_counts()
    .reindex(class_order)
    .reset_index()
)
count_df.columns = ['water_quality', 'count']

# Barplot of counts
sns.barplot(
    data=count_df,
    x='water_quality',
    y='count',
    hue='water_quality',
    hue_order=class_order,
    order=class_order,
    palette=palette_custom,
    legend=False
)

plt.title('Distribution of Samples by Water Quality Class')
plt.xlabel('Water Quality Class')
plt.ylabel('Number of Samples')
plt.xticks(ticks=range(len(class_order)), labels=x_tick_labels)
plt.tight_layout()
plt.savefig('figures/data-exploration/01_water_quality_distribution_count.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# Figure 2 – Maximum Temperature by Water Quality
# -------------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x='water_quality', y='Temperature (C) - Max',
    hue='water_quality',
    hue_order=class_order,
    order=class_order,
    palette=palette_custom,
    legend=False
)
plt.title('Temperature (°C) - Max by Water Quality Class')
plt.xlabel('Water Quality Class')
plt.ylabel('Temperature (°C) - Max')
plt.xticks(ticks=range(len(class_order)), labels=x_tick_labels)
plt.tight_layout()
plt.savefig('figures/data-exploration/02_temp_max_vs_quality.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# Figure 3 – Water Quality Distribution by Water Body Type (proportions)
# Deterministic stacking: A (bottom) → C (middle) → Other (top)
# -------------------------------------------------------------------
# Build readable 'water_body_type' from one-hot columns
body_cols = [c for c in df.columns if c.startswith('Type Water Body_')]
df['water_body_type'] = df[body_cols].idxmax(axis=1).str.replace('Type Water Body_', '', regex=False)

# counts & proportions per body type
counts_bt = df.groupby(['water_body_type', 'water_quality']).size().unstack(fill_value=0)
props_bt  = counts_bt.div(counts_bt.sum(axis=1), axis=0).fillna(0.0)

# enforce deterministic class order and handle dtypes
props_bt = props_bt.reindex(columns=class_order, fill_value=0.0)
props_bt.columns = pd.CategoricalIndex(props_bt.columns, ordered=True, categories=class_order)
props_bt = props_bt.sort_index(axis=1)

# sort bars by share of Class A (optional for readability)
order_bt = props_bt.sort_values(by=1, ascending=False).index if 1 in props_bt.columns else props_bt.index
props_bt_sorted = props_bt.loc[order_bt]
n_per_bt = counts_bt.loc[order_bt].sum(axis=1)

ax = props_bt_sorted.plot(
    kind='bar',
    stacked=True,
    figsize=(10, 6),
    width=0.85,
    edgecolor='white',
    linewidth=0.5,
    color=[palette_custom[c] for c in props_bt_sorted.columns]  # [1, 2, 0]
)

plt.title('Proportion of Water Quality Classes by Water Body Type', fontsize=14)
plt.xlabel('Water Body Type', fontsize=12)
plt.ylabel('Proportion of Samples', fontsize=12)
plt.xticks(rotation=45, ha='right')

# annotate n above bars and add legend
annotate_counts_above_bars(ax, n_per_bt, y_offset=0.02)
handles = [mpatches.Patch(color=palette_custom[c], label=class_labels[c]) for c in class_order]
plt.legend(handles=handles, title='Water Quality', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('figures/data-exploration/03_water_quality_vs_body_type.png', dpi=300)
plt.show()

# -------------------------------------------------------------------
# Figure 4 – pH Range by Water Quality
# -------------------------------------------------------------------
df['pH_range'] = df['pH - Max'] - df['pH - Min']

plt.figure(figsize=(7, 5))
sns.boxplot(
    data=df,
    x='water_quality', y='pH_range',
    hue='water_quality',
    hue_order=class_order,
    order=class_order,
    palette=palette_custom,
    legend=False
)
plt.title('pH Range (Max - Min) by Water Quality Class')
plt.xlabel('Water Quality Class')
plt.ylabel('pH Range (Max - Min)')
plt.xticks(ticks=range(len(class_order)), labels=x_tick_labels)
plt.tight_layout()
plt.savefig('figures/data-exploration/04_ph_range_vs_quality.png', dpi=300)
plt.show()


# -------------------------------------------------------------------
# Figure 5 – Proportion of Water Quality Levels Across States (proportions)
# Deterministic stacking: A (bottom) → C (middle) → Other (top)
# -------------------------------------------------------------------
state_cols = [col for col in df.columns if col.startswith('State Name_')]
df['state'] = df[state_cols].idxmax(axis=1).str.replace('State Name_', '', regex=False)

# counts & proportions per state
counts_st = df.groupby(['state', 'water_quality']).size().unstack(fill_value=0)
props_st  = counts_st.div(counts_st.sum(axis=1), axis=0).fillna(0.0)

# enforce deterministic class order and handle dtypes
props_st = props_st.reindex(columns=class_order, fill_value=0.0)
props_st.columns = pd.CategoricalIndex(props_st.columns, ordered=True, categories=class_order)
props_st = props_st.sort_index(axis=1)

# sort states by share of Class A (optional for readability)
order_st = props_st.sort_values(by=1, ascending=False).index if 1 in props_st.columns else props_st.index
props_st_sorted = props_st.loc[order_st]
n_per_st = counts_st.loc[order_st].sum(axis=1)

ax = props_st_sorted.plot(
    kind='bar',
    stacked=True,
    figsize=(12, 6),
    width=0.85,
    edgecolor='white',
    linewidth=0.5,
    color=[palette_custom[c] for c in props_st_sorted.columns]  # [1, 2, 0]
)

plt.title('Proportion of Water Quality Levels Across Indian States', fontsize=14)
plt.xlabel('State', fontsize=12)
plt.ylabel('Proportion of Samples', fontsize=12)
plt.xticks(rotation=45, ha='right')

# annotate n above bars and add legend
annotate_counts_above_bars(ax, n_per_st, y_offset=0.02)
handles = [mpatches.Patch(color=palette_custom[c], label=class_labels[c]) for c in class_order]
plt.legend(handles=handles, title='Water Quality', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig('figures/data-exploration/05_state_vs_quality_proportion.png', dpi=300)
plt.show()
