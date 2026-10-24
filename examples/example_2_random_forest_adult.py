"""Example 2 - Random forest on Adult: k-pseudo-probable explanations (GA and GD).

Run:  python examples/example_2_random_forest_adult.py
"""
import os
import sys

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from probexplain import TreeEnsembleExplainer  # noqa: E402

K = 4                  # maximal size of the explanations
N_INSTANCES = 5        # number of test instances to explain

# ---- Adult: the categorical columns are encoded as integer codes
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adult.csv"))
X, y = df.drop(columns="income"), df["income"]
for c in X.select_dtypes(include="object").columns:
    X[c] = pd.factorize(X[c], sort=True)[0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# ---- model
rf = RandomForestClassifier(n_estimators=20, max_depth=6, random_state=0).fit(X_train, y_train)
print(f"Random forest accuracy: {rf.score(X_test, y_test):.3f}\n")

# ---- explanations
ex = TreeEnsembleExplainer(rf, feature_names=X.columns)
for i in range(N_INSTANCES):
    x = X_test.iloc[i].values
    print(f"========== instance {i} (true class {y_test.iloc[i]})")
    for method in ["ga", "gd"]:
        print(ex.explain(x, k=K, method=method), "\n")
