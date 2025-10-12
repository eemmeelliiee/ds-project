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

# Validate input file(s)
def ReadPreviousData(location: str):
    Filepath = Path(location)
    if Filepath.exists():
        data = pd.read_csv(location) 
        return data, True 
    else:
        data = pd.DataFrame()

df, df_exists = ReadPreviousData('data/water_quality_cols_complete_rows_rel_cols_all_num_less_corr.csv')

PreviousDataComplete = all([df_exists])

if PreviousDataComplete == False:
    sys.exit("ERROR: Not all required data found.") 
print("All files found")

#Define features/target
X = df.drop(columns='water_quality')
y = df['water_quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

pred = clf.predict(X_test)
print(pred)

print(clf.score(X_train, y_train))
print(clf.score(X_test, y_test))

print(classification_report(y_test, pred))
cm=confusion_matrix(y_test, pred)
print(cm)
confusion_matrix = sns.heatmap(cm, annot=True)
fig = confusion_matrix.get_figure()
fig.savefig('figures/confusion_matrix')

print(accuracy_score(y_test, pred))
print(clf.score(X_test, y_test))

# implement also for these:
# recall_score()
# precision_score()

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='accuracy')
print("Cross-validation accuracy scores:", scores)
print("Mean CV accuracy:", scores.mean())

test_accuracy = clf.score(X_test, y_test)
print("Final test set accuracy:", test_accuracy)

# Evaluate accuracy and macro-F1
acc_scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
f1_macro_scores = cross_val_score(clf, X, y, cv=cv, scoring=make_scorer(f1_score, average='macro'))
print("Mean accuracy:", acc_scores.mean())
print("Mean macro F1:", f1_macro_scores.mean())
print(y.value_counts(normalize=True))

# Multiclass proof
print(clf.classes_)
print(len(clf.classes_))

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(clf,filled=True,rounded=True, ax=axes)
plt.savefig('figures/decision_tree.png')

