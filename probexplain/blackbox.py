"""BE (BanditExplain): PAC probable explanations for any black-box classifier.

Only the ``predict`` method of the model is used. Each feature of x becomes ONE literal:
  * numeric feature     : the quantile interval (out of ``n_bins``) containing the value of x;
  * categorical feature : the value of x.
The error of a set S of literals is estimated by Monte-Carlo: the features of S are fixed to their
value in x, the others are taken from a random row of the background data, and the error is the
proportion of predictions different from h(x).

(eta, delta)-PAC guarantee: N = ceil(2/eta^2 * ln(2m/delta_step)) samples per candidate,
delta_step = delta / (d + 1).
"""
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer

from .explanation import Explanation


def remove_literal(Q, l):
    """Copy of Q without the literal l."""
    O = list(Q)
    O.remove(l)
    return O


class MixedBinarizer:
    """Numeric: ``n_bins`` intervals (quantiles). Categorical: exact value."""

    def __init__(self, numeric_cols, cat_cols, n_bins=3, strategy="quantile"):
        self.numeric_cols = list(numeric_cols)
        self.cat_cols = list(cat_cols)
        self.n_bins, self.strategy = n_bins, strategy
        self.kbins = None

    def fit(self, X_train):
        X_train = np.asarray(X_train, dtype=float)
        if self.numeric_cols:
            self.kbins = KBinsDiscretizer(n_bins=self.n_bins, encode="ordinal", strategy=self.strategy)
            self.kbins.fit(X_train[:, self.numeric_cols])
        return self

    def bin_edges_for(self, j):
        return self.kbins.bin_edges_[self.numeric_cols.index(j)]


def binarize_instance(x, binarizer):
    """Instance -> (list of literals 1..d, lit_info). One literal per feature."""
    x = np.asarray(x)
    literals, lit_info, lit_id = [], {}, 1

    if binarizer.numeric_cols:
        x_num = x[binarizer.numeric_cols].reshape(1, -1).astype(float)
        bins_idx = binarizer.kbins.transform(x_num)[0].astype(int)
        for local_idx, j in enumerate(binarizer.numeric_cols):
            b = bins_idx[local_idx]
            edges = binarizer.bin_edges_for(j)
            lit_info[lit_id] = {"type": "num", "feature": j, "bin": int(b),
                                "range": (edges[b], edges[b + 1])}
            literals.append(lit_id)
            lit_id += 1

    for j in binarizer.cat_cols:
        lit_info[lit_id] = {"type": "cat", "feature": j, "value": x[j]}
        literals.append(lit_id)
        lit_id += 1

    return literals, lit_info


def decode_explanation(S, lit_info, feature_names=None, cat_decoders=None):
    """Readable explanation. cat_decoders[j]: code -> original category."""
    out = []
    for l in S:
        info = lit_info[abs(l)]
        j = info["feature"]
        name = feature_names[j] if feature_names is not None else f"x[{j}]"
        if info["type"] == "num":
            lo, hi = info["range"]
            out.append(f"{name} in [{lo:.4g}, {hi:.4g}]")
        else:
            value = info["value"]
            if cat_decoders is not None and j in cat_decoders:
                value = cat_decoders[j][int(value)]
            out.append(f"{name} = {value}")
    return out


class _Oracle:
    """Model wrapper counting the queries (rebuilds a DataFrame if the model was fit on one)."""

    def __init__(self, h, columns=None):
        self.h, self.columns, self.n_queries = h, columns, 0

    def predict(self, Z):
        self.n_queries += len(Z)
        if self.columns is not None:
            import pandas as pd
            Z = pd.DataFrame(Z, columns=self.columns)
        return np.asarray(self.h.predict(Z))


def estimate_error(h, x, S, background, lit_info, pred, N, rng):
    """Estimate of eps_{h,x}(S) with N queries to the model h."""
    x = np.asarray(x)
    fixed = np.array([lit_info[abs(l)]["feature"] for l in S], dtype=int)
    Z = background[rng.integers(0, len(background), size=N)].copy()
    if len(fixed) > 0:
        Z[:, fixed] = x[fixed]
    return np.mean(h.predict(Z) != pred)


def best_arm_identification(h, x, candidates, background, lit_info, pred, eta, delta, rng):
    """Candidate with minimal estimated error (eta-optimal with probability >= 1 - delta)."""
    m = len(candidates)
    N = int(np.ceil((2 / eta ** 2) * np.log(2 * m / delta)))
    best_C, best_err = None, None
    for C in candidates:
        err = estimate_error(h, x, C, background, lit_info, pred, N, rng)
        if best_err is None or err < best_err:
            best_err, best_C = err, C
    return best_C, best_err


def BE_explain(h, x, binarizer, background, k, eta, delta, rng=None):
    """BanditExplain: starts from the full implicant, removes at each step the literal chosen by
    best-arm identification, then chooses the final explanation among those of size 1..k.

    Returns (explanation, estimated error, lit_info)."""
    rng = rng if rng is not None else np.random.default_rng()
    x = np.asarray(x)
    pred = h.predict(x.reshape(1, -1))[0]

    literals, lit_info = binarize_instance(x, binarizer)
    d = len(literals)
    k = min(k, d)
    delta_step = delta / (d + 1)

    S, chain = list(literals), []
    while len(S) > 0:
        candidates = [remove_literal(S, l) for l in S]
        selected, _ = best_arm_identification(h, x, candidates, background, lit_info,
                                              pred, eta, delta_step, rng)
        removed = [l for l in S if l not in selected][0]
        chain.append(list(S))
        S.remove(removed)

    chain.reverse()                   # chain[i] has size i + 1
    explanation, error = best_arm_identification(h, x, chain[:k], background, lit_info,
                                                 pred, eta, delta_step, rng)
    return explanation, error, lit_info


class BlackBoxExplainer:
    """BE explanations for any fitted classifier with a ``predict`` method.

    Parameters
    ----------
    model : fitted classifier (neural network, SVM, boosting, pipeline, ...)
    X_background : data (array or DataFrame) the free features are sampled from, e.g. X_train
    categorical_features : indices (or column names) of the categorical features, encoded as
        integer codes; all the other features are treated as numeric
    n_bins : number of quantile intervals of the numeric features
    eta, delta : PAC parameters (precision of the estimated errors, failure probability)
    cat_decoders : optional {feature index: list of category names} to print the explanations
    """

    def __init__(self, model, X_background, categorical_features=(), n_bins=3, eta=0.1,
                 delta=0.05, feature_names=None, cat_decoders=None, random_state=None):
        columns = getattr(X_background, "columns", None)
        self.feature_names = (list(feature_names) if feature_names is not None
                              else list(map(str, columns)) if columns is not None else None)
        cat = [list(columns).index(c) if columns is not None and not isinstance(c, (int, np.integer))
               else int(c) for c in categorical_features]
        self.background = np.asarray(X_background, dtype=float)
        numeric = [j for j in range(self.background.shape[1]) if j not in cat]
        self.binarizer = MixedBinarizer(numeric, cat, n_bins=n_bins).fit(self.background)
        self.oracle = _Oracle(model, None if columns is None else list(columns))
        self.model = model
        self.eta, self.delta = eta, delta
        self.cat_decoders = cat_decoders
        self.rng = np.random.default_rng(random_state)

    def explain(self, x, k, *, eta=None, delta=None):
        x = np.asarray(x, dtype=float).ravel()
        eta = self.eta if eta is None else eta
        delta = self.delta if delta is None else delta
        self.oracle.n_queries = 0
        S, error, lit_info = BE_explain(self.oracle, x, self.binarizer, self.background,
                                        k, eta, delta, self.rng)
        prediction = self.oracle.predict(x.reshape(1, -1))[0]
        return Explanation(method="be", prediction=prediction, literals=list(S),
                           features=sorted(lit_info[l]["feature"] for l in S),
                           conditions=decode_explanation(S, lit_info, self.feature_names,
                                                         self.cat_decoders),
                           error=float(error))
