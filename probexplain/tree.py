"""Probable explanations for decision trees (UAI 2023).

The error of a term S (set of literals of the instance) is the probability, under the uniform
distribution on the Boolean variables, that the tree does not predict the class of x when S is
fixed: W_f(phi, S) with phi the orthogonal DNF of the other classes. We look for S of size <= k
with minimal error.

Methods
  * "gd"     greedy descent  (Approx_descent):   starts from the direct reason, removes literals
  * "ga"     greedy ascent   (Approx_ascendant): starts from the empty set, adds literals
"""
import numpy as np

from ._boolean import Binarization, BooleanTree
from ._counting import M_s, W_s
from .explanation import Explanation


def W_f(dnf, t, n):
    """Error of term t: probability that the orthogonal DNF holds when t is fixed."""
    return W_s(dnf, t, n)


def _argmin(values):
    """Index of the first minimum (like np.argmin, but safe for huge Python ints)."""
    values = list(values)
    return min(range(len(values)), key=values.__getitem__)


def D_s(Q, l):
    """Copy of Q without l."""
    O = list(Q)
    O.remove(l)
    return O


def approx_descent(phi, I_s, n, k=None):
    """Greedy descent (GD): best subset of I_s of size at most k.

    phi: orthogonal DNF of the negated decision, I_s: direct reason, n: number of variables."""
    S_k = []
    S = list(I_s)
    if k is None or k > len(I_s):
        k = len(I_s)
    for _ in range(len(I_s)):
        e = S[_argmin([M_s(phi, D_s(S, l), n=n) for l in S])]
        S_k.append(S.copy())
        S.remove(e)
    S_k.reverse()   # S_k[i] has size i + 1
    return S_k[_argmin([W_f(phi, C, n) for C in S_k[0:k]])]


def approx_ascent(phi, I_s, n, k=None):
    """Greedy ascent (GA): best subset of I_s of size at most k."""
    S, S_k = [], []
    X = list(I_s)
    if k is None or k > len(I_s):
        k = len(I_s)
    for _ in range(k):
        e = X[_argmin([M_s(phi, S + [l], n) for l in X])]
        S = S + [e]
        S_k.append(S)
        X.remove(e)
    return S_k[_argmin([W_f(phi, C, n) for C in S_k])]


class DecisionTreeExplainer:
    """Probable explanations (UAI 2023) for a fitted ``sklearn.tree.DecisionTreeClassifier``.

    >>> ex = DecisionTreeExplainer(dt, feature_names=X.columns)
    >>> print(ex.explain(x, k=4, method="gd"))
    """

    METHODS = ("gd", "ga")

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.bina = Binarization()
        self.tree = BooleanTree(model, self.bina)
        self.n = len(self.bina)

    def explain(self, x, k=None, method="gd"):
        x = np.asarray(x, dtype=float).ravel()
        target = self.tree.predict_index(x)
        phi = self.tree.dnf_not(target)
        R_d = self.tree.decision_path(x)
        n = self.n

        if method == "gd":
            S = approx_descent(phi, R_d, n, k)
        elif method == "ga":
            S = approx_ascent(phi, R_d, n, k)
        else:
            raise ValueError(f"unknown method {method!r}, expected one of {self.METHODS}")

        return Explanation(method=f"uai2023-{method}", prediction=self.model.classes_[target],
                           literals=list(S), features=self.bina.features(S),
                           conditions=self.bina.describe(S, self.feature_names),
                           error=float(W_f(phi, S, n)))
