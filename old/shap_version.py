import pandas as pd
import shap
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

# === Load your data ===
raw = pd.read_csv("data/original_dataset.csv")

# === Create the target column (based on CPCB rules) ===
def make_target_column(df):
    tmp = df.copy()

    # Replace BDLs
    tmp['Dissolved - Min'] = tmp['Dissolved - Min'].replace('BDL', 0.3 / 2)
    tmp['BOD (mg/L) - Max'] = tmp['BOD (mg/L) - Max'].replace('BDL', 1.0 / 2)

    cols_needed = ['pH - Min', 'pH - Max', 'Dissolved - Min',
                   'BOD (mg/L) - Max', 'Total Coliform (MPN/100ml) - Max']
    for c in cols_needed:
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

    return pd.Series(np.select([class_a, class_c], [2, 1], default=0), name="target")

# === Drop columns that cause leakage or are irrelevant ===
DROP_EARLY = [
    'STN code','Monitoring Location','Year',
    'Conductivity (µmho/cm) - Min','Conductivity (µmho/cm) - Max',
    'NitrateN (mg/L) - Min','NitrateN (mg/L) - Max',
    'Fecal Coliform (MPN/100ml) - Min','Fecal Coliform (MPN/100ml) - Max',
    'Fecal - Min','Fecal - Max'
]
DROP_AFTER_TARGET = [
    'Total Coliform (MPN/100ml) - Min','Total Coliform (MPN/100ml) - Max',
    'BOD (mg/L) - Min','BOD (mg/L) - Max','Dissolved - Max'
]

# === Prepare data ===
y = make_target_column(raw)
X = raw.drop(columns=DROP_EARLY, errors="ignore")
X = X.drop(columns=DROP_AFTER_TARGET, errors="ignore")
X = pd.get_dummies(X, drop_first=True)

feature_names = X.columns.tolist()

# === Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# === Preprocessing ===
imp = SimpleImputer(strategy='median')
scaler = StandardScaler()

X_train_imp = imp.fit_transform(X_train)
X_train_scaled = scaler.fit_transform(X_train_imp)
X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)

# === Train Models ===
dt = DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42)
dt.fit(X_train_scaled_df, y_train)

lr = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=42)
lr.fit(X_train_scaled_df, y_train)

# === SHAP for Decision Tree (Handles Multiclass/3D) ===
explainer_tree = shap.TreeExplainer(dt)
shap_tree = explainer_tree.shap_values(X_train_scaled_df)

print("Decision Tree SHAP Summary:")
print("Unique target classes:", np.unique(y_train))

# --- Handle SHAP output shape/type ---
if isinstance(shap_tree, list):
    print("SHAP returned a list (multiclass format).")
    class_idx = 2 if 2 in np.unique(y_train) else 1
    shap_vals = shap_tree[class_idx]

elif shap_tree.ndim == 3:
    print("SHAP returned a 3D array (n_samples, n_features, n_classes).")
    class_idx = 2 if 2 in np.unique(y_train) else 1
    shap_vals = shap_tree[:, :, class_idx]

else:
    print("SHAP returned a 2D array (binary/regression).")
    shap_vals = shap_tree

# --- Debug Shapes ---
print(f"shap_vals shape: {shap_vals.shape}")
print(f"X_train_scaled_df shape: {X_train_scaled_df.shape}")
print(f"Number of features: {len(X_train_scaled_df.columns)}")

# --- Plot SHAP Summary ---
shap.summary_plot(
    shap_vals,
    features=X_train_scaled_df,
    feature_names=X_train_scaled_df.columns,
    show=True
)
