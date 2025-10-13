import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn.tree as tree
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, f1_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix #plot_roc_curve
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, classification_report

def ModelDataAndEvaluate():

    # Validate input file(s)
    def ReadPreviousData(location: str):
        Filepath = Path(location)
        if Filepath.exists():
            data = pd.read_csv(location) 
            return data, True 
        else:
            data = pd.DataFrame()
            return data, False

    df, df_exists = ReadPreviousData('data/cleaned_dataset.csv')

    PreviousDataComplete = all([df_exists])

    if PreviousDataComplete == False:
        sys.exit("ERROR: Not all required data found.") 
    print("All files found")

    # Define features/target
    X = df.drop(columns='water_quality')
    y = df['water_quality']

    # Split dataset for final test
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    clf = DecisionTreeClassifier(random_state=42, class_weight='balanced')

    # Cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    f1_macro = make_scorer(f1_score, average='macro')
    cv_acc_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
    cv_f1_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring=f1_macro)
    # Print mean of cv scores
    print(f"Avg CV accuracy score: {cv_acc_scores.mean():.4f}")
    print(f"Avg CV F1 score: {cv_f1_scores.mean():.4f}")

    # Test on test set
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    test_acc_score = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    test_recall_macro = recall_score(y_test, y_pred, average='macro')
    test_precision_macro = precision_score(y_test, y_pred, average='macro')

    print(f"Avg test accuracy score:     {test_acc_score:.4f}")
    print(f"Avg test F1 (macro):         {test_f1_macro:.4f}")
    print(f"Avg test F1 (weighted):      {test_f1_weighted:.4f}")
    print(f"Avg test recall (macro):     {test_recall_macro:.4f}")
    print(f"Avg test precision (macro):  {test_precision_macro:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True)
    plt.savefig('figures/confusion_matrix.png', dpi=300)
    plt.close() 

    # -------------------------------------------------------------------
    # Decision Tree visualization
    # -------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)  # increased size for clarity
    tree.plot_tree(clf, filled=True, rounded=True, ax=ax, fontsize=6)
    plt.title('Decision Tree')
    plt.tight_layout()
    plt.savefig('figures/decision_tree.png', dpi=300)
    plt.close()

    # #_------------------------------
    # # # Multiclass proof
    # # print(clf.classes_)
    # # print(len(clf.classes_))


    # # Classification report
    # cp = classification_report(y_test, y_pred)
    # print('\nClassification report:\n', cp)
