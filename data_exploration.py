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

def ExploreData():

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

    df, df_exists = ReadPreviousData('data/cleaned_dataset.csv')

    if not df_exists:
        sys.exit("ERROR: Not all required data found.")
    print("All files found")

    # -------------------------------------------------------------------
    # Class order, labels, and color palette
    # -------------------------------------------------------------------
    class_order = [2, 1, 0]
    class_labels = {2: 'A', 1: 'C', 0: 'Other'}
    x_tick_labels = [class_labels[i] for i in class_order]

    # Color-blind–friendly blue → light blue → red
    palette_custom = {
        2: "#364B9A",  # Blue  → Class A (Good)
        1: "#6EA6CD",  # Light Blue → Class C (Moderate)
        0: "#DD3D2D"   # Red   → Class Other (Poor)
    }

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------
    def annotate_counts(ax, counts, offset=0.08):
        for i, n in enumerate(counts):
            ax.text(i, 1 + offset, f"n={int(n)}", ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.1 + offset)

    # -------------------------------------------------------------------
    # Figure 1 – Distribution by Class
    # -------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    counts = df['water_quality'].value_counts().reindex(class_order).reset_index()
    counts.columns = ['water_quality', 'count']

    sns.barplot(
        data=counts,
        x='water_quality',
        y='count',
        hue='water_quality',
        order=class_order,
        hue_order=class_order,
        palette=palette_custom,
        legend=False
    )

    plt.title('Distribution of Samples by Water Quality Class')
    plt.xlabel('Water Quality Class')
    plt.ylabel('Number of Samples')
    plt.xticks(range(len(class_order)), x_tick_labels)
    plt.savefig('figures/data-exploration/01_water_quality_distribution_count.png', dpi=300)


    # -------------------------------------------------------------------
    # Figure 2 – Max Temperature by Class
    # -------------------------------------------------------------------
    plt.figure(figsize=(10, 10))
    sns.boxplot(
        data=df,
        x='water_quality',
        y='Temperature (C) - Max',
        order=class_order,
        hue='water_quality',
        hue_order=class_order,
        palette=palette_custom,
        legend=False
    )
    plt.title('Temperature (°C) - Max by Water Quality Class')
    plt.xlabel('Water Quality Class')
    plt.ylabel('Temperature (°C) - Max')
    plt.xticks(range(len(class_order)), x_tick_labels)
    plt.savefig('figures/data-exploration/02_temp_max_vs_quality.png')


    # -------------------------------------------------------------------
    # Figure 3 – Proportion by Water Body Type
    # -------------------------------------------------------------------
    body_cols = [c for c in df if c.startswith('Type Water Body_')]
    df['water_body_type'] = df[body_cols].idxmax(axis=1).str.replace('Type Water Body_', '', regex=False)

    counts_bt = df.groupby(['water_body_type', 'water_quality']).size().unstack(fill_value=0)
    props_bt = counts_bt.div(counts_bt.sum(axis=1), axis=0).reindex(columns=class_order, fill_value=0)

    # Order water body types by total number of samples (descending)
    order_bt = counts_bt.sum(axis=1).sort_values(ascending=False).index
    n_per_bt = counts_bt.loc[order_bt].sum(axis=1)

    ax = props_bt.loc[order_bt].plot(
        kind='bar', stacked=True, figsize=(20,13),
        color=[palette_custom[c] for c in class_order], edgecolor='white', linewidth=0.5
    )
    plt.title('Proportion of Water Quality Classes by Water Body Type')
    plt.xlabel('Water Body Type')
    plt.ylabel('Proportion of Samples')
    plt.xticks(rotation=45, ha='right')
    annotate_counts(ax, n_per_bt)
    handles = [mpatches.Patch(color=palette_custom[c], label=class_labels[c]) for c in class_order]
    plt.legend(handles=handles, title='Water Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('figures/data-exploration/03_water_quality_vs_body_type.png')

    # -------------------------------------------------------------------
    # Figure 4 – pH Range by Class
    # -------------------------------------------------------------------
    df['pH_range'] = df['pH - Max'] - df['pH - Min']

    plt.figure(figsize=(10, 10))
    sns.boxplot(
        data=df,
        x='water_quality',
        y='pH_range',
        order=class_order,
        hue='water_quality',
        hue_order=class_order,
        palette=palette_custom,
        legend=False
    )
    plt.title('pH Range (Max - Min) by Water Quality Class')
    plt.xlabel('Water Quality Class')
    plt.ylabel('pH Range (Max - Min)')
    plt.xticks(range(len(class_order)), x_tick_labels)
    plt.savefig('figures/data-exploration/04_ph_range_vs_quality.png')


    # -------------------------------------------------------------------
    # Figure 5 – Proportion by State
    # -------------------------------------------------------------------
    state_cols = [c for c in df if c.startswith('State Name_')]
    df['state'] = df[state_cols].idxmax(axis=1).str.replace('State Name_', '', regex=False)

    counts_st = df.groupby(['state', 'water_quality']).size().unstack(fill_value=0)
    props_st = counts_st.div(counts_st.sum(axis=1), axis=0).reindex(columns=class_order, fill_value=0)

    # Order states by total number of samples (descending)
    order_st = counts_st.sum(axis=1).sort_values(ascending=False).index
    n_per_st = counts_st.loc[order_st].sum(axis=1)

    ax = props_st.loc[order_st].plot(
        kind='bar', stacked=True, figsize=(20,13),
        color=[palette_custom[c] for c in class_order], edgecolor='white', linewidth=0.5
    )
    plt.title('Proportion of Water Quality Levels Across Indian States')
    plt.xlabel('State')
    plt.ylabel('Proportion of Samples')
    plt.xticks(rotation=45, ha='right')
    annotate_counts(ax, n_per_st)
    handles = [mpatches.Patch(color=palette_custom[c], label=class_labels[c]) for c in class_order]
    plt.legend(handles=handles, title='Water Quality', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig('figures/data-exploration/05_state_vs_quality_proportion.png')