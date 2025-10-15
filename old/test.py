# simple_water_quality_models.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    classification_report, confusion_matrix
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin

# -----------------------------
# basic config
# -----------------------------
RAW_CSV = "data/original_dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

FIG_DIR = Path("figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# columns we don't want / leak
DROP_EARLY = [
    'STN code','Monitoring Location','Year',
    'Conductivity (¬µmho/cm) - Min','Conductivity (¬µmho/cm) - Max',
    'NitrateN (mg/L) - Min','NitrateN (mg/L) - Max',
    'Fecal Coliform (MPN/100ml) - Min','Fecal Coliform (MPN/100ml) - Max',
    'Fecal - Min','Fecal - Max'
]
DROP_AFTER_TARGET = [
    'Total Coliform (MPN/100ml) - Min','Total Coliform (MPN/100ml) - Max',
    'BOD (mg/L) - Min','BOD (mg/L) - Max','Dissolved - Max'
]

CATEGORICAL_COLS = ['Type Water Body', 'State Name']
RAW_WATER_DUMMY = 'Type Water Body_WATER TREATMENT PLANT (RAW WATER)'

OBJ_TO_FLOAT = ['Dissolved - Min','BOD (mg/L) - Min','Total Coliform (MPN/100ml) - Min']


# -----------------------------
# target rule (CPCB-ish)
# -----------------------------
def make_target(df: pd.DataFrame) -> pd.Series:
    tmp = df.copy()

    # replace BDL with half LOD-ish
    if 'Dissolved - Min' in tmp:
        tmp['Dissolved - Min'] = tmp['Dissolved - Min'].replace('BDL', 0.3/2)
    if 'BOD (mg/L) - Max' in tmp:
        tmp['BOD (mg/L) - Max'] = tmp['BOD (mg/L) - Max'].replace('BDL', 1.0/2)

    # columns used by rules
    cols = ['pH - Min','pH - Max','Dissolved - Min','BOD (mg/L) - Max','Total Coliform (MPN/100ml) - Max']
    for c in cols:
        if c in tmp:
            tmp[c] = pd.to_numeric(tmp[c], errors='coerce')

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

    # 2 = best (A), 1 = ok (C), 0 = else
    return np.select([class_a, class_c], [2, 1], default=0)


# -----------------------------
# simple pandas preprocessor for sklearn Pipeline
# -----------------------------
class SimplePandasPrep(BaseEstimator, TransformerMixin):
    """
    does very basic prep:
    - drop some columns
    - replace 'BDL'
    - cast object-y numbers
    - drop possible leakage
    - one-hot categoricals and remember columns
    - drop that one RAW WATER dummy if present
    """
    def __init__(self, drop_raw_dummy=True):
        self.drop_raw_dummy = drop_raw_dummy
        self.cols_ = None

    def _prep_once(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # drop early
        df.drop(columns=[c for c in DROP_EARLY if c in df.columns], errors='ignore', inplace=True)

        # BDL replacements (leave NaNs for imputer)
        if 'Dissolved - Min' in df:
            df['Dissolved - Min'] = df['Dissolved - Min'].replace('BDL', 0.3/2)
        if 'BOD (mg/L) - Min' in df:
            df['BOD (mg/L) - Min'] = df['BOD (mg/L) - Min'].replace('BDL', 1.0/2)

        # cast some object columns to numbers
        for c in OBJ_TO_FLOAT:
            if c in df:
                df[c] = pd.to_numeric(df[c], errors='coerce')

        # drop after-target (to avoid leakage)
        df.drop(columns=[c for c in DROP_AFTER_TARGET if c in df.columns], errors='ignore', inplace=True)

        # one-hot categoricals (only ones that exist)
        cats = [c for c in CATEGORICAL_COLS if c in df.columns]
        if cats:
            df = pd.get_dummies(df, columns=cats, dtype=int)

        # drop the raw water dummy if there
        if self.drop_raw_dummy and RAW_WATER_DUMMY in df.columns:
            df.drop(columns=[RAW_WATER_DUMMY], inplace=True)

        return df

    def fit(self, X, y=None):
        Xp = self._prep_once(X)
        self.cols_ = Xp.columns.tolist()
        return self

    def transform(self, X):
        Xp = self._prep_once(X)
        # align to fit columns
        for c in self.cols_:
            if c not in Xp:
                Xp[c] = 0
        # drop any extra columns and fix order
        Xp = Xp[self.cols_]
        return Xp


# -----------------------------
# training and evaluation
# -----------------------------
def main():
    # load raw
    p = Path(RAW_CSV)
    if not p.exists():
        sys.exit(f"File not found: {RAW_CSV}")
    raw = pd.read_csv(p)

    # target BEFORE dropping leakage
    y = make_target(raw)

    # split
    X_train, X_test, y_train, y_test = train_test_split(
        raw, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    # models to try
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

    # evaluate
    for name, clf in models.items():
        print(f"\n=== {name} ===")
        pipe = Pipeline([
            ("prep", SimplePandasPrep(drop_raw_dummy=True)),
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf)
        ])

        # quick CV
        skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
        accs, f1s, bals = [], [], []
        for tr_idx, va_idx in skf.split(X_train, y_train):
            Xtr, Xva = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            ytr, yva = y_train.iloc[tr_idx], y_train.iloc[va_idx]
            pipe.fit(Xtr, ytr)
            pr = pipe.predict(Xva)
            accs.append(accuracy_score(yva, pr))
            f1s.append(f1_score(yva, pr, average="macro"))
            bals.append(balanced_accuracy_score(yva, pr))

        print(f"CV accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
        print(f"CV f1_macro: {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
        print(f"CV balanced_accuracy: {np.mean(bals):.4f} ± {np.std(bals):.4f}")

        # test set
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        print("\nTest Metrics:")
        print(f"Accuracy        : {accuracy_score(y_test, y_pred):.4f}")
        print(f"Macro F1        : {f1_score(y_test, y_pred, average='macro'): .4f}")
        print(f"Balanced Acc    : {balanced_accuracy_score(y_test, y_pred):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # confusion matrix figure
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title(f"Confusion Matrix - {name}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        out = FIG_DIR / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        plt.savefig(out, dpi=300)
        plt.close()

    print("\nSaved figures in ./figures")


if __name__ == "__main__":
    main()
