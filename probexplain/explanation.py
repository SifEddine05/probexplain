"""Result object returned by every explainer."""
from dataclasses import dataclass
from typing import Any, List

import numpy as np


def predict_one(model, x):
    """model.predict on a single instance (as a DataFrame if the model was fit on one)."""
    X = np.asarray(x, dtype=float).reshape(1, -1)
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        import pandas as pd
        X = pd.DataFrame(X, columns=names)
    return model.predict(X)[0]


@dataclass
class Explanation:
    """A (probable / pseudo-probable / PAC) explanation of a prediction.

    Attributes
    ----------
    method : str        algorithm used, e.g. "uai2023-ga", "kpseudo-gd", "be"
    prediction : Any    the predicted class being explained
    literals : list     the explanation, as Boolean literals of the model's binarization
    features : list     indices of the original features involved
    conditions : list   human readable conditions (x[f] > t, intervals, categorical values...)
    error : float       error of the explanation: probability (UAI 2023), weighted pseudo-probability
                        (k-pseudo) or Monte-Carlo estimate (BE) that the prediction changes when the
                        features outside the explanation vary
    """
    method: str
    prediction: Any
    literals: List[int]
    features: List[int]
    conditions: List[str]
    error: float

    @property
    def size(self):
        return len(self.literals)

    @property
    def precision(self):
        return 1.0 - self.error

    def __str__(self):
        lines = [f"Explanation [{self.method}] of prediction {self.prediction!r}",
                 f"  size = {self.size}, error = {self.error:.6g} (precision {self.precision:.6g})"]
        lines += [f"  * {c}" for c in self.conditions] or ["  (empty explanation)"]
        return "\n".join(lines)
