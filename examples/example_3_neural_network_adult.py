"""Example 3 - Neural network on Adult: BE (BanditExplain) PAC explanations.

Run:  python examples/example_3_neural_network_adult.py
"""
import os
import sys
import warnings

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from probexplain import BlackBoxExplainer  # noqa: E402

warnings.filterwarnings("ignore")   # empty quantile bins of capital-gain / capital-loss

K = 4                  # maximal size of the explanations
N_INSTANCES = 3        # number of test instances to explain
ETA, DELTA = 0.1, 0.05  # PAC parameters of BE

# ---- Adult: the categorical columns are encoded as integer codes, decoders keep the names
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "adult.csv"))
X, y = df.drop(columns="income"), df["income"]
categorical = list(X.select_dtypes(include="object").columns)
decoders = {}
for c in categorical:
    X[c], names = pd.factorize(X[c], sort=True)
    decoders[X.columns.get_loc(c)] = list(names)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

# ---- model
nn = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0))
nn.fit(X_train, y_train)
print(f"Neural network accuracy: {nn.score(X_test, y_test):.3f}\n")

# ---- explanations
ex = BlackBoxExplainer(nn, X_train, categorical_features=categorical, cat_decoders=decoders,
                       n_bins=3, eta=ETA, delta=DELTA, random_state=0)
for i in range(N_INSTANCES):
    print(f"========== instance {i} (true class {y_test.iloc[i]})")
    print(ex.explain(X_test.iloc[i].values, k=K), "\n")
