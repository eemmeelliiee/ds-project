from pathlib import Path
import sys
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,balanced_accuracy_score, precision_score, recall_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.base import clone as clone
import matplotlib.patches as mpatches

# Config
RAW_CSV = "data/original_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

FIG_DIR = Path("figures")

# SHAP Based feature pruning
ENABLE_SHAP_FEATURE_PRUNE = True  # set to False to disable
SHAP_PRUNE_THRESHOLD = 0.0  # keep features with mean SHAP > threshold
SHAP_PRUNE_MIN_FEATURES = 25 # keep at least this many (by top-importance)

# Columns not needed to determine water quality class
EARLY_DROP = [
    'STN code','Monitoring Location','Year',
    'Conductivity (¬µmho/cm) - Min','Conductivity (¬µmho/cm) - Max',
    'NitrateN (mg/L) - Min','NitrateN (mg/L) - Max',
    'Fecal Coliform (MPN/100ml) - Min','Fecal Coliform (MPN/100ml) - Max',
    'Fecal - Min','Fecal - Max'
]

# Object columns to be converted to float
OBJ_TO_FLOAT = ['Dissolved - Min','BOD (mg/L) - Min','Total Coliform (MPN/100ml) - Min']

# Columns not relevant after determining water quality class (target column)
DROP_AFTER_TARGET = [
    'Total Coliform (MPN/100ml) - Min','Total Coliform (MPN/100ml) - Max',
    'BOD (mg/L) - Min','BOD (mg/L) - Max','Dissolved - Max'
]

# Categorical columns to be dummy encoded
CATEGORICAL_COLS = ['Type Water Body', 'State Name']

# Path used for data exploration/visualization
CLEANED_PATH = Path("data/cleaned_dataset.csv")


# Utilities
def load_raw_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: Input file not found at {path}")
    return pd.read_csv(p)

# Build water_quality target using CPCB rules
def make_target_column(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()

    # Columns needed to determine water quality
    cols_needed = [
        'pH - Min', 'pH - Max',
        'Dissolved - Min',
        'BOD (mg/L) - Max',
        'Total Coliform (MPN/100ml) - Max'
    ]

    # Print obj
    print(tmp.select_dtypes(include=['object']).columns)

    # Replace BDL in relevant object columns
    if 'Dissolved - Min' in tmp.columns:
        tmp['Dissolved - Min'] = tmp['Dissolved - Min'].replace('BDL', 0.3/2)  # = DO LOD/2
    if 'BOD (mg/L) - Max' in tmp.columns:
        tmp['BOD (mg/L) - Max'] = tmp['BOD (mg/L) - Max'].replace('BDL', 1.0/2)  # = BOD LOD/2

    # Coerce needed columns to numeric
    for c in cols_needed:
        if c in tmp.columns:
            tmp[c] = pd.to_numeric(tmp[c], errors='coerce')

    # CPCB rules
    class_a = (
        (tmp['pH - Min'] >= 6.5) & (tmp['pH - Max'] <= 8.5) &
        (tmp['Dissolved - Min'] >= 6) &
        (tmp['BOD (mg/L) - Max'] <= 2) &
        (tmp['Total Coliform (MPN/100ml) - Max'] <= 50)
    )
    class_c = (
        (tmp['pH - Min'] >= 6) & (tmp['pH - Max'] <= 9) &
        (tmp['Dissolved - Min'] >= 4) &
        (tmp['BOD (mg/L) - Max'] <= 3) &
        (tmp['Total Coliform (MPN/100ml) - Max'] <= 5000)
    )
    return np.select([class_a, class_c], [2, 1], default=0)

def explore_data():
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

    class_order = [2, 1, 0]
    class_labels = {2: 'A', 1: 'C', 0: 'Other'}
    x_tick_labels = [class_labels[i] for i in class_order]

    palette_custom = {
        2: "#364B9A",  # Class A (Good)
        1: "#6EA6CD",  # Class C (Moderate)
        0: "#DD3D2D"   # Class Other (Poor)
    }

    def annotate_counts(ax, counts, offset=0.08):
        for i, n in enumerate(counts):
            ax.text(i, 1 + offset, f"n={int(n)}", ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, 1.1 + offset)

    # Figure 1 – Distribution by Class
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
    plt.close()

    # Figure 2 – Max Temperature by Class
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
    plt.close()

    # Figure 3 – Proportion by Water Body Type
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
    plt.close()


    # Figure 4 – pH Range by Class
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
    plt.close()

    # Figure 5 – Proportion by State
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
    plt.close()

# Custom sklearn transformer
class PandasPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_ = None

    def _base_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Drop irrelevant columns (early)
        cols_to_drop = [c for c in EARLY_DROP if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Handle BDL replacements (no row drops)
        df.replace(to_replace={'Dissolved - Min': 'BDL'}, value=0.3/2, inplace=True)
        df.replace(to_replace={'BOD (mg/L) - Max': 'BDL'}, value=1.0/2, inplace=True)

        # Cast "numerical" objects to numeric (NaNs allowed, later impute)
        for c in OBJ_TO_FLOAT:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Drop post-target (leakage) columns
        cols_to_drop2 = [c for c in DROP_AFTER_TARGET if c in df.columns]
        df.drop(columns=cols_to_drop2, inplace=True, errors='ignore')

        return df

    def fit(self, X, y=None):
        df = self._base_prepare(X)

        # Dummy on fit, learn final column set
        existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, dtype=int)

        # Lock the column order for later transforms
        self.feature_names_ = df.columns.tolist()
        return self

    def transform(self, X):
        df = self._base_prepare(X)

        existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, dtype=int)

        # Align to fitted columns (add missing with 0, drop extras, order columns)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names_]

        return df

    def get_feature_names_out(self):
        return np.array(self.feature_names_ if self.feature_names_ is not None else [])

# Select a subset of columns by name (Sklearn-clone friendly)
class ColumnSelector(BaseEstimator, TransformerMixin):
    def __init__(self, columns=None):
        self.columns = columns  # Store exactly what was passed; do not coerce/mutate here or clone() will complain.

    def fit(self, X, y=None):
        self.columns_ = None if self.columns is None else list(self.columns) # Create an internal, immutable snapshot for transform-time use
        return self

    def transform(self, X):
        import pandas as _pd
        if self.columns_ is None:
            return X
        df = X if hasattr(X, "columns") else _pd.DataFrame(X, columns=self.columns_)
        cols = [c for c in self.columns_ if c in df.columns]
        return df[cols]
    
def main():
    raw = load_raw_df(RAW_CSV)
    y = make_target_column(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    #Create and save a cleaned dataset for analysis
    preprocessor = PandasPreprocessor()
    preprocessor.fit(raw) # fit preprocessing steps (e.g., dummy variables)
    cleaned_df = preprocessor.transform(raw)

    # Add the target column
    cleaned_df['water_quality'] = make_target_column(raw)
    CLEANED_PATH.parent.mkdir(exist_ok=True, parents=True)
    cleaned_df.to_csv(CLEANED_PATH, index=False)

    print(f"Cleaned dataset saved to: {CLEANED_PATH}")

    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            criterion="gini",
            max_depth=6,
            min_samples_leaf=5,
            min_samples_split=2
        ),
    }

    # Evaluation
    for name, model in models.items():
        print(f"\n=== Evaluating: {name} ===")

        pipe = Pipeline([
        ("prep", PandasPreprocessor()), # Fitted clone, ready for transform() and predict()
        ("imputer", SimpleImputer(strategy="median")),  
        ("clf", model)
        ])

        # Cross-validation
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "f1_weighted": "f1_weighted",
            "balanced_accuracy": "balanced_accuracy",
            'precision_weighted': 'precision_weighted',
            "recall_weighted": "recall_weighted",
            "neg_log_loss": "neg_log_loss"
        }

        cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

        print(f"CV f1_weighted: {cv_results['test_f1_weighted'].mean():.3f} ± {cv_results['test_f1_weighted'].std():.3f}")
        print(f"CV balanced_accuracy: {cv_results['test_balanced_accuracy'].mean():.3f} ± {cv_results['test_balanced_accuracy'].std():.3f}")
        print(f"CV precision_weighted: {cv_results['test_precision_weighted'].mean():.3f} ± {cv_results['test_precision_weighted'].std():.3f}")
        print(f"CV recall_weighted: {cv_results['test_recall_weighted'].mean():.3f} ± {cv_results['test_recall_weighted'].std():.3f}")
        print(f"CV log_loss: {-cv_results['test_neg_log_loss'].mean():.3f} ± {-cv_results['test_neg_log_loss'].std():.3f}")


        # Train and evaluate on test set
        pipe.fit(X_train, y_train)

        # === Predict-proba section (main model) ===
        y_proba = pipe.predict_proba(X_test)                 # (n_samples, n_classes)
        classes = pipe.named_steps["clf"].classes_
        proba_cols = [f"proba_{int(c)}" for c in classes]

        proba_df = pd.DataFrame(y_proba, columns=proba_cols)
        proba_df["y_true"] = np.array(y_test)
        proba_df["y_pred"] = pipe.predict(X_test)
        proba_df["y_pred_confidence"] = y_proba.max(axis=1)

        FIG_DIR.mkdir(exist_ok=True, parents=True)
        proba_csv = FIG_DIR / f"test_pred_proba_{name.replace(' ', '_').lower()}.csv"
        proba_df.to_csv(proba_csv, index=False)
        print(f"[Proba] Saved per-sample probabilities to: {proba_csv}")

        # === Calibration (Reliability) Curve ===
        from sklearn.calibration import calibration_curve
        import matplotlib.pyplot as plt

        # 1. Compare predicted confidence vs. actual correctness
        y_true_binary = (proba_df["y_true"] == proba_df["y_pred"]).astype(int)
        probs = proba_df["y_pred_confidence"]

        prob_true, prob_pred = calibration_curve(y_true_binary, probs, n_bins=10)

        plt.figure(figsize=(6,6))
        plt.plot(prob_pred, prob_true, marker='o', label='Model calibration')
        plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
        plt.xlabel("Predicted confidence")
        plt.ylabel("Observed accuracy")
        plt.title(f"Calibration Curve – {name}")
        plt.legend()
        plt.tight_layout()

        calib_path = FIG_DIR / f"calibration_curve_{name.replace(' ', '_').lower()}.png"
        plt.savefig(calib_path, dpi=300)
        plt.close()

        print(f"[Proba] Saved calibration curve to: {calib_path}")


        # Optional: quick confidence histogram
        plt.figure(figsize=(6, 4))
        sns.histplot(proba_df["y_pred_confidence"], bins=20, edgecolor="white")
        plt.title(f"Prediction Confidence (max class prob) — {name}")
        plt.xlabel("Max predicted probability")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confidence_hist_{name.replace(' ', '_').lower()}.png", dpi=300)
        plt.close()

        y_pred = pipe.predict(X_test)

        print("\nTest Metrics:")
        print(f"Test f1_weighted        : {f1_score(y_test, y_pred, average='weighted'): .3f}")
        print(f"Test balanced_accuracy    : {balanced_accuracy_score(y_test, y_pred):.3f}")
        print(f"Test precision_weighted    : {precision_score(y_test, y_pred, average='weighted'):.3f}")
        print(f"Test recall_weighted    : {recall_score(y_test, y_pred, average='weighted'):.3f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        from sklearn.metrics import log_loss

        y_proba = pipe.predict_proba(X_test)
        test_logloss = log_loss(y_test, y_proba)
        print(f"Test log loss           : {test_logloss:.3f}")

        # Optional: Save confusion matrix per model
        cm = confusion_matrix(y_test, y_pred) 
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png", dpi=300)
        plt.close()
        
        print("Computing SHAP values...")

        try:
            # Get fitted model and preprocessed features
            fitted_model = pipe.named_steps["clf"]
            prep = pipe.named_steps["prep"]
            imputer = pipe.named_steps["imputer"]

            # Apply same preprocessing to X_train
            X_train_prepped = prep.transform(X_train)
            X_train_imputed = imputer.transform(X_train_prepped)

            feature_names = prep.get_feature_names_out()
            X_df = pd.DataFrame(X_train_imputed, columns=feature_names)

            if "tree" in str(type(fitted_model)).lower():
                explainer = shap.TreeExplainer(fitted_model)
                shap_vals = explainer.shap_values(X_df)
            else:
                explainer = shap.LinearExplainer(fitted_model, X_df)
                shap_vals = explainer.shap_values(X_df) 

            # Determine target classes
            unique_classes = np.unique(y_train)
            class_idx = 2 if 2 in unique_classes else 1

            # Handle SHAP output types
            if isinstance(shap_vals, list):
                shap_class_vals = shap_vals[class_idx]
            elif shap_vals.ndim == 3:
                shap_class_vals = shap_vals[:, :, class_idx]
            else:
                shap_class_vals = shap_vals  # binary/regression

            print(f"SHAP shape: {shap_class_vals.shape} | Features: {X_df.shape}")

            # Plot and save SHAP summary
            shap.summary_plot(
                shap_class_vals,
                features=X_df,
                feature_names=feature_names,
                show=False
            )
            plt.title(f"SHAP Summary - {name}")
            plt.tight_layout()
            shap_path = FIG_DIR / f"shap_summary_{name.replace(' ', '_').lower()}.png"
            plt.savefig(shap_path, dpi=300)
            plt.close()
            print(f"SHAP summary plot saved to: {shap_path}")

            # Feature pruning based on SHAP importance
            if ENABLE_SHAP_FEATURE_PRUNE:
                try:
                    # Compute mean SHAP per feature
                    mean_abs = np.mean(np.abs(shap_class_vals), axis=0)
                    shap_importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)

                    # Decide which to keep
                    keep_by_thresh = shap_importance[shap_importance > SHAP_PRUNE_THRESHOLD].index.tolist()
                    if len(keep_by_thresh) < SHAP_PRUNE_MIN_FEATURES:
                        keep_features = shap_importance.index.tolist()[:SHAP_PRUNE_MIN_FEATURES]
                    else:
                        keep_features = keep_by_thresh

                    # Save the importances for traceability
                    imp_path = FIG_DIR / f"shap_importance_{name.replace(' ', '_').lower()}.csv"
                    shap_importance.to_csv(imp_path, header=["mean_abs_shap"])
                    print(f"[SHAP] Saved importance CSV to: {imp_path}")
                    print(f"[SHAP] Keeping {len(keep_features)} features (threshold={SHAP_PRUNE_THRESHOLD}, floor={SHAP_PRUNE_MIN_FEATURES})")

                    # Refit and evaluate the same model on the reduced feature set
                    from sklearn.base import clone as _clone
                    reduced_model = _clone(model)

                    reduced_pipe = Pipeline([
                        ("prep", PandasPreprocessor()),
                        ("select", ColumnSelector(keep_features)),
                        ("imputer", SimpleImputer(strategy="median")),
                        ("clf", reduced_model)
                    ])

                    # CV on reduced
                    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
                    scoring = {
                        "f1_weighted": "f1_weighted",
                        "balanced_accuracy": "balanced_accuracy",
                        'precision_weighted': 'precision_weighted',
                        "recall_weighted": "recall_weighted",
                        "neg_log_loss": "neg_log_loss"
                        
                    }
                    cv_reduced = cross_validate(reduced_pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)
                    print(f"[Reduced] CV f1_weighted: {cv_reduced['test_f1_weighted'].mean():.3f} ± {cv_reduced['test_f1_weighted'].std():.3f}")
                    print(f"[Reduced] CV balanced_acc: {cv_reduced['test_balanced_accuracy'].mean():.3f} ± {cv_reduced['test_balanced_accuracy'].std():.3f}")
                    print(f"[Reduced] CV precision_weighted: {cv_reduced['test_precision_weighted'].mean():.3f} ± {cv_reduced['test_precision_weighted'].std():.3f}")
                    print(f"[Reduced] CV recall_weighted: {cv_reduced['test_recall_weighted'].mean():.3f} ± {cv_reduced['test_recall_weighted'].std():.3f}")
                    mean_logloss = -cv_reduced['test_neg_log_loss'].mean()
                    std_logloss  =  cv_reduced['test_neg_log_loss'].std()   # keep positive
                    print(f"[Reduced] CV log_loss: {mean_logloss:.3f} ± {std_logloss:.3f}")

                    # Test on reduced
                    reduced_pipe.fit(X_train, y_train)
                    y_pred_red  = reduced_pipe.predict(X_test)      # ← move this up
                    # === Predict-proba section (reduced model) ===
                    y_proba_red = reduced_pipe.predict_proba(X_test)
                    classes_red = reduced_pipe.named_steps["clf"].classes_
                    proba_cols_red = [f"proba_{int(c)}" for c in classes_red]

                    proba_df_red = pd.DataFrame(y_proba_red, columns=proba_cols_red)
                    proba_df_red["y_true"] = np.array(y_test)
                    proba_df_red["y_pred"] = y_pred_red
                    proba_df_red["y_pred_confidence"] = y_proba_red.max(axis=1)

                    proba_csv_red = FIG_DIR / f"test_pred_proba_reduced_{name.replace(' ', '_').lower()}.csv"
                    proba_df_red.to_csv(proba_csv_red, index=False)
                    print(f"[Proba][Reduced] Saved per-sample probabilities to: {proba_csv_red}")

                    # === Calibration (Reliability) Curve ===
                    from sklearn.calibration import calibration_curve
                    import matplotlib.pyplot as plt

                   # 1) Compare predicted confidence vs. actual correctness (REDUCED model)
                    y_true_binary_red = (proba_df_red["y_true"] == proba_df_red["y_pred"]).astype(int)
                    probs_red = proba_df_red["y_pred_confidence"]

                    # If not already imported above:
                    # from sklearn.calibration import calibration_curve

                    prob_true_red, prob_pred_red = calibration_curve(y_true_binary_red, probs_red, n_bins=10)

                    plt.figure(figsize=(6,6))
                    plt.plot(prob_pred_red, prob_true_red, marker='o', label='Model calibration (reduced)')
                    plt.plot([0,1], [0,1], 'k--', label='Perfect calibration')
                    plt.xlabel("Predicted confidence")
                    plt.ylabel("Observed accuracy")
                    plt.title(f"Calibration Curve – {name} (Reduced)")
                    plt.legend()
                    plt.tight_layout()

                    calib_path_red = FIG_DIR / f"calibration_curve_reduced_{name.replace(' ', '_').lower()}.png"
                    plt.savefig(calib_path_red, dpi=300)
                    plt.close()

                    print(f"[Proba][Reduced] Saved calibration curve to: {calib_path_red}")


                    print(f"[Proba] Saved calibration curve to: {calib_path}")


                    y_pred_red = reduced_pipe.predict(X_test)
                    print("\n[Reduced] Test Metrics:")
                    print("\nTest Metrics:")
                    print(f"Test f1_weighted        : {f1_score(y_test,  y_pred_red, average='weighted'): .3f}")
                    print(f"Test balanced_accuracy    : {balanced_accuracy_score(y_test,  y_pred_red):.3f}")
                    print(f"Test precision_weighted    : {precision_score(y_test,  y_pred_red, average='weighted'):.3f}")
                    print(f"Test recall_weighted    : {recall_score(y_test,  y_pred_red, average='weighted'):.3f}")
                    print(classification_report(y_test, y_pred_red))
                    # === Log Loss Evaluation (Reduced model) ===
                    y_proba_red = reduced_pipe.predict_proba(X_test)
                    test_logloss_red = log_loss(y_test, y_proba_red)
                    print(f"Test log loss (reduced) : {test_logloss_red:.3f}")


                except Exception as _e:
                    print(f"[SHAP] Pruning step failed for {name}: {_e}")

        except Exception as e:
            print(f"[SHAP] Skipped SHAP summary for {name} due to error:\n{e}")
    explore_data()
    print("\nArtifacts saved in ./figures")

if __name__ == "__main__":
    main()
