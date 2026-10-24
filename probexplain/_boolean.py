"""Boolean (threshold) encoding of scikit-learn decision trees.

Each pair (feature, threshold) used by a split becomes a Boolean variable numbered 1..n.
Literal +v means ``x[feature] > threshold`` and -v means ``x[feature] <= threshold``
(same convention as scikit-learn: ``x <= threshold`` goes to the left child).

A tree is represented by its root-to-leaf paths; the paths leading to a set of classes form an
orthogonal DNF (the leaves are disjoint), which is what the error functions of the UAI 2023 and
k-pseudo papers count models of.
"""
from fractions import Fraction

import numpy as np


class Binarization:
    """Mapping (feature, threshold) <-> Boolean variable, shared by all the trees of an ensemble."""

    def __init__(self):
        self._var = {}          # (feature, threshold) -> variable id
        self._key = [None]      # variable id -> (feature, threshold)

    def __len__(self):
        return len(self._key) - 1

    def add(self, feature, threshold):
        key = (int(feature), float(threshold))
        if key not in self._var:
            self._var[key] = len(self._key)
            self._key.append(key)
        return self._var[key]

    def var(self, feature, threshold):
        return self._var[(int(feature), float(threshold))]

    def key(self, literal):
        """(feature, threshold, is_greater) of a literal."""
        feature, threshold = self._key[abs(literal)]
        return feature, threshold, literal > 0

    def binarize(self, x):
        """Full implicant of an instance: one literal per Boolean variable."""
        x = np.asarray(x, dtype=float)
        return [v if x[f] > t else -v for v, (f, t) in enumerate(self._key[1:], start=1)]

    def features(self, literals):
        """Sorted indices of the original features involved in a set of literals."""
        return sorted({self._key[abs(l)][0] for l in literals})

    def describe(self, literals, feature_names=None):
        """Human readable conditions; the literals on one feature are merged into an interval."""
        bounds = {}
        for l in literals:
            f, t, greater = self.key(l)
            lo, hi = bounds.get(f, (-np.inf, np.inf))
            bounds[f] = (max(lo, t), hi) if greater else (lo, min(hi, t))
        out = []
        for f in sorted(bounds):
            name = feature_names[f] if feature_names is not None else f"x[{f}]"
            lo, hi = bounds[f]
            if lo == -np.inf:
                out.append(f"{name} <= {hi:.4g}")
            elif hi == np.inf:
                out.append(f"{name} > {lo:.4g}")
            else:
                out.append(f"{lo:.4g} < {name} <= {hi:.4g}")
        return out


class BooleanTree:
    """A fitted scikit-learn tree seen through a (possibly shared) Binarization.

    Parameters
    ----------
    sk_tree : sklearn.tree.DecisionTreeClassifier (fitted)
    binarization : Binarization, variables of this tree are added to it
    class_map : array, class_map[j] = class index of the j-th column of ``tree_.value``
        (identity for a stand-alone tree; ``estimator.classes_`` for trees inside an ensemble)
    feature_map : array, feature_map[j] = index in the full instance of the j-th feature seen by
        the tree (identity, except for Bagging with feature sub-sampling)
    """

    def __init__(self, sk_tree, binarization, class_map=None, feature_map=None):
        ts = sk_tree.tree_
        self._ts = ts
        self.bina = binarization
        n_cols = ts.value.shape[2]
        self._class_map = np.arange(n_cols) if class_map is None else np.asarray(class_map).astype(int)
        self._feature_map = (np.arange(ts.n_features) if feature_map is None
                             else np.asarray(feature_map).astype(int))
        # variables are created in node index order (as my_tree.from_DecisionTreeClassifier)
        self._node_var = {}
        for i in range(ts.node_count):
            if ts.children_left[i] != -1:
                self._node_var[i] = binarization.add(self._feature_map[ts.feature[i]], ts.threshold[i])
        # root-to-leaf paths: (class index, literals)
        self.paths = []
        stack = [(0, [])]
        while stack:
            node, lits = stack.pop()
            if ts.children_left[node] == -1:
                self.paths.append((self._leaf_class(node), lits))
            else:
                v = self._node_var[node]
                stack.append((ts.children_right[node], lits + [v]))
                stack.append((ts.children_left[node], lits + [-v]))

    def _leaf_class(self, node):
        return int(self._class_map[int(np.argmax(self._ts.value[node][0]))])

    def _walk(self, x):
        x = np.asarray(x, dtype=float)
        ts, node, lits = self._ts, 0, []
        while ts.children_left[node] != -1:
            v = self._node_var[node]
            if x[self._feature_map[ts.feature[node]]] <= ts.threshold[node]:
                lits.append(-v)
                node = ts.children_left[node]
            else:
                lits.append(v)
                node = ts.children_right[node]
        return node, lits

    def predict_index(self, x):
        """Class index predicted for x."""
        return self._leaf_class(self._walk(x)[0])

    def confidence(self, x):
        """Proportion of the majority class in the leaf reached by x (exact Fraction)."""
        value = self._ts.value[self._walk(x)[0]][0]
        return Fraction(float(value.max())) / Fraction(float(value.sum()))

    def decision_path(self, x):
        """Direct reason of x: the literals of its root-to-leaf path."""
        return self._walk(x)[1]

    def dnf(self, classes):
        """Orthogonal DNF (list of terms) of the leaves whose class is in ``classes``."""
        classes = set(classes)
        return [lits for c, lits in self.paths if c in classes]

    def dnf_not(self, target):
        """Orthogonal DNF of "the tree does not predict ``target``"."""
        return [lits for c, lits in self.paths if c != target]
