"""Example 5 - Text classification (20 Newsgroups, CATEGORY_A vs CATEGORY_B): BE explanations.

Same approach as the MNIST example: the raw input (a document) is turned into a fixed number of
features, the model is trained on them, and BE (BlackBoxExplainer, the classic method) explains
its predictions.
  * features: presence (1) / absence (0) of the N_WORDS most discriminant words (chi2);
    each word is one categorical feature, i.e. one literal "word = present / absent";
  * model: a neural network (MLPClassifier) on these features;
  * the words of the explanation are highlighted in the document: >>word<<.

Run:  python examples/example_5_text_be.py
20 Newsgroups is downloaded by scikit-learn the first time.
"""
import os
import re
import sys
import time

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from probexplain import BlackBoxExplainer  # noqa: E402

CATEGORY_A, CATEGORY_B = "sci.med", "sci.space"
N_WORDS = 50                    # number of word features (= literals of BE)
N_INSTANCES = 4                 # number of test documents to explain
K = 5                           # maximal size of the explanations (in words)
ETA, DELTA = 0.15, 0.1          # PAC parameters of BE

# ---- documents -> presence / absence of the N_WORDS most discriminant words
remove = ("headers", "footers", "quotes")
train = fetch_20newsgroups(subset="train", categories=[CATEGORY_A, CATEGORY_B], remove=remove)
test = fetch_20newsgroups(subset="test", categories=[CATEGORY_A, CATEGORY_B], remove=remove)
names = np.array(train.target_names)
y_train, y_test = names[train.target], names[test.target]

vectorizer = CountVectorizer(binary=True, stop_words="english", min_df=5)
counts = vectorizer.fit_transform(train.data)
selector = SelectKBest(chi2, k=N_WORDS).fit(counts, y_train)
words = list(vectorizer.get_feature_names_out()[selector.get_support()])


def to_features(docs):
    return selector.transform(vectorizer.transform(docs)).toarray().astype(float)


X_train, X_test = to_features(train.data), to_features(test.data)
print(f"{CATEGORY_A} vs {CATEGORY_B}: {len(X_train)} train / {len(X_test)} test documents")
print(f"{N_WORDS} word features: {', '.join(words)}\n")

# ---- model
nn = MLPClassifier(hidden_layer_sizes=(32,), max_iter=500, random_state=0).fit(X_train, y_train)
print(f"Neural network accuracy: {nn.score(X_test, y_test):.3f}\n")

# ---- BE explanations
ex = BlackBoxExplainer(nn, X_train, categorical_features=list(range(N_WORDS)), feature_names=words,
                       cat_decoders={j: ["absent", "present"] for j in range(N_WORDS)},
                       eta=ETA, delta=DELTA, random_state=0)


def highlight(text, marked):
    """The text with the words of ``marked`` written >>word<<."""
    return re.sub(r"\w+", lambda m: f">>{m.group()}<<" if m.group().lower() in marked else m.group(),
                  text)


# short documents with at least 3 of the word features, half of each class
chosen = [i for i in range(len(X_test)) if X_test[i].sum() >= 3 and len(test.data[i]) < 600]
chosen = [i for c in names for i in [j for j in chosen if y_test[j] == c][:N_INSTANCES // 2]]
for i in chosen:
    t0 = time.perf_counter()
    e = ex.explain(X_test[i], k=K)
    elapsed = time.perf_counter() - t0
    present = {words[j] for j in e.features if X_test[i, j] == 1}
    print("=" * 78)
    print(f"document {i} | true class = {y_test[i]} | predicted class = {e.prediction}\n")
    print(highlight(test.data[i].strip(), present), "\n")
    print(e)
    print(f"  time: {elapsed:.1f} s\n")
