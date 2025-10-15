import os
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
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, f1_score,
    balanced_accuracy_score
)
from sklearn.tree import DecisionTreeClassifier, plot_tree

# -----------------------------
# Config
# -----------------------------
RAW_CSV = "data/original_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Columns you previously dropped early
DROP_EARLY = [
    'STN code','Monitoring Location','Year',
    'Conductivity (¬µmho/cm) - Min','Conductivity (¬µmho/cm) - Max',
    'NitrateN (mg/L) - Min','NitrateN (mg/L) - Max',
    'Fecal Coliform (MPN/100ml) - Min','Fecal Coliform (MPN/100ml) - Max',
    'Fecal - Min','Fecal - Max'
]

# Object columns that should be cast to numeric after BDL handling
OBJ_TO_FLOAT = ['Dissolved - Min','BOD (mg/L) - Min','Total Coliform (MPN/100ml) - Min']

# Columns to drop AFTER creating the target (to avoid leakage)
DROP_AFTER_TARGET = [
    'Total Coliform (MPN/100ml) - Min','Total Coliform (MPN/100ml) - Max',
    'BOD (mg/L) - Min','BOD (mg/L) - Max','Dissolved - Max'
]

# Categorical columns for one-hot encoding (we’ll use pandas.get_dummies)
CATEGORICAL_COLS = ['Type Water Body', 'State Name']

# One-hot dummy you wanted removed for dimensionality/correlation reasons
RAW_WATER_DUMMY = 'Type Water Body_WATER TREATMENT PLANT (RAW WATER)'


# -----------------------------
# Utilities
# -----------------------------
def load_raw_df(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"ERROR: Input file not found at {path}")
    return pd.read_csv(p)


def make_target_column(df: pd.DataFrame) -> pd.Series:
    """
    Build 'water_quality' target using CPCB rules.
    Handles 'BDL' safely and coerces to numeric for rule columns.
    """
    tmp = df.copy()

    cols_needed = [
        'pH - Min', 'pH - Max',
        'Dissolved - Min',
        'BOD (mg/L) - Max',
        'Total Coliform (MPN/100ml) - Max'
    ]

    # Replace BDL where it could appear
    if 'Dissolved - Min' in tmp.columns:
        tmp['Dissolved - Min'] = tmp['Dissolved - Min'].replace('BDL', 0.3/2)  # DO LOD/2
    if 'BOD (mg/L) - Max' in tmp.columns:
        tmp['BOD (mg/L) - Max'] = tmp['BOD (mg/L) - Max'].replace('BDL', 1.0/2)  # conservative

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


# -----------------------------
# Custom sklearn transformer
# -----------------------------
class PandasPreprocessor(BaseEstimator, TransformerMixin):
    """
    Preprocessing without changing sample count:
    - drop early/leakage columns
    - handle BDL text -> numeric placeholders
    - cast numeric-like objects to numeric (leave NaNs, impute later)
    - one-hot encode with stable columns learned on fit
    - optionally drop the RAW WATER dummy
    """
    def __init__(self, drop_raw_water_dummy: bool = True):
        self.drop_raw_water_dummy = drop_raw_water_dummy
        self.feature_names_ = None

    def _base_prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Drop irrelevant columns (early)
        cols_to_drop = [c for c in DROP_EARLY if c in df.columns]
        df.drop(columns=cols_to_drop, inplace=True, errors='ignore')

        # Handle "BDL" replacements (no row drops)
        df.replace(to_replace={'Dissolved - Min': 'BDL'}, value=0.3/2, inplace=True)
        df.replace(to_replace={'BOD (mg/L) - Min': 'BDL'}, value=1.0/2, inplace=True)

        # Cast numeric-ish objects to numeric (NaNs allowed; will impute)
        for c in OBJ_TO_FLOAT:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # Drop post-target/leakage columns
        cols_to_drop2 = [c for c in DROP_AFTER_TARGET if c in df.columns]
        df.drop(columns=cols_to_drop2, inplace=True, errors='ignore')

        return df

    def fit(self, X, y=None):
        df = self._base_prepare(X)

        # One-hot on fit, learn final column set
        existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, dtype=int)

        # Optionally drop RAW WATER dummy
        if self.drop_raw_water_dummy and RAW_WATER_DUMMY in df.columns:
            df.drop(columns=[RAW_WATER_DUMMY], inplace=True)

        # Lock the column order for later transforms
        self.feature_names_ = df.columns.tolist()
        return self

    def transform(self, X):
        df = self._base_prepare(X)

        existing_cats = [c for c in CATEGORICAL_COLS if c in df.columns]
        if existing_cats:
            df = pd.get_dummies(df, columns=existing_cats, dtype=int)

        if self.drop_raw_water_dummy and RAW_WATER_DUMMY in df.columns:
            df.drop(columns=[RAW_WATER_DUMMY], inplace=True)

        # Align to fitted columns (add missing with 0, drop extras, order columns)
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_names_]

        return df

    def get_feature_names_out(self):
        return np.array(self.feature_names_ if self.feature_names_ is not None else [])


# -----------------------------
# Main: end-to-end run
# -----------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

def main():
    # 1) Load raw data
    raw = load_raw_df(RAW_CSV)

    # 2) Build y from raw (before we drop leakage columns)
    y = make_target_column(raw)

    # 3) Train/test split with stratify
    X_train, X_test, y_train, y_test = train_test_split(
        raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # 4) Define models to evaluate
    models = {
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            class_weight="balanced",
            criterion="gini",
            max_depth=6,
            min_samples_leaf=5,
            min_samples_split=10
        ),
        "Logistic Regression": LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=3000,
            class_weight="balanced",
            solver="lbfgs"
        )
    }

    # 5) Evaluate each model
    for name, model in models.items():
        print(f"\n=== Evaluating: {name} ===")

        pipe = Pipeline([
        ("prep", PandasPreprocessor(drop_raw_water_dummy=True)),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),  
        ("clf", model)
        ])
        # Cross-validation
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        scoring = {
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "balanced_accuracy": "balanced_accuracy"
        }

        cv_results = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, return_train_score=False)

        print(f"CV accuracy: {cv_results['test_accuracy'].mean():.4f} ± {cv_results['test_accuracy'].std():.4f}")
        print(f"CV f1_macro: {cv_results['test_f1_macro'].mean():.4f} ± {cv_results['test_f1_macro'].std():.4f}")
        print(f"CV balanced_accuracy: {cv_results['test_balanced_accuracy'].mean():.4f} ± {cv_results['test_balanced_accuracy'].std():.4f}")

        # Train and evaluate on test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print("\nTest Metrics:")
        print(f"Accuracy        : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Macro F1        : {f1_score(y_test, y_pred, average='macro'): .4f}")
        print(f"Balanced Acc    : {balanced_accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

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
        
        # -----------------------------
        # SHAP Evaluation
        # -----------------------------
        print("Computing SHAP values...")

        try:
            # Get fitted model and preprocessed features
            fitted_model = pipe.named_steps["clf"]
            prep = pipe.named_steps["prep"]
            imputer = pipe.named_steps["imputer"]
            scaler = pipe.named_steps["scaler"]

            # Apply same preprocessing to X_train
            X_train_prepped = prep.transform(X_train)
            X_train_imputed = imputer.transform(X_train_prepped)
            X_train_scaled = scaler.transform(X_train_imputed)

            feature_names = prep.get_feature_names_out()
            X_df = pd.DataFrame(X_train_scaled, columns=feature_names)

            if "tree" in str(type(fitted_model)).lower():
                explainer = shap.TreeExplainer(fitted_model)
                shap_vals = explainer.shap_values(X_df)
            else:
                # Use a subset for efficiency (KernelExplainer is slow)
                background = shap.sample(X_df, 100, random_state=RANDOM_STATE)
                explainer = shap.KernelExplainer(fitted_model.predict_proba, background)
                shap_vals = explainer.shap_values(X_df[:50])  # sample to speed up
                X_df = X_df.iloc[:50]


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

        except Exception as e:
            print(f"[SHAP] Skipped SHAP summary for {name} due to error:\n{e}")


    print("\nArtifacts saved in ./figures")

if __name__ == "__main__":
    main()
