"""probexplain: probable explanations of classifiers.

  * decision trees           -> UAI 2023 probable explanations       (DecisionTreeExplainer)
  * tree ensembles           -> k-pseudo-probable explanations       (TreeEnsembleExplainer)
  * any other (black box)    -> BE, BanditExplain PAC explanations   (BlackBoxExplainer)

>>> from probexplain import explain
>>> e = explain(model, x, k=4, X_background=X_train)
>>> print(e)
"""
from sklearn.tree import DecisionTreeClassifier

from .blackbox import BlackBoxExplainer
from .ensemble import TreeEnsembleExplainer
from .explanation import Explanation
from .tree import DecisionTreeExplainer

__all__ = ["explain", "make_explainer", "Explanation", "DecisionTreeExplainer",
           "TreeEnsembleExplainer", "BlackBoxExplainer"]
__version__ = "0.1.0"


def _is_tree_ensemble(model):
    estimators = getattr(model, "estimators_", None)
    return (isinstance(estimators, list) and len(estimators) > 0 and hasattr(model, "classes_")
            and all(isinstance(e, DecisionTreeClassifier) for e in estimators))


def make_explainer(model, X_background=None, **kwargs):
    """Explainer adapted to the model: UAI 2023 for a DecisionTreeClassifier, k-pseudo for a forest
    of decision trees (RandomForest, ExtraTrees, Bagging), BE for anything else.

    kwargs are passed to the explainer constructor (feature_names, categorical_features, n_bins,
    eta, delta, cat_decoders, random_state)."""
    if isinstance(model, DecisionTreeClassifier):
        return DecisionTreeExplainer(model, feature_names=kwargs.get("feature_names"))
    if _is_tree_ensemble(model):
        return TreeEnsembleExplainer(model, feature_names=kwargs.get("feature_names"))
    if X_background is None:
        raise ValueError(f"{type(model).__name__} is explained with BE (black box): "
                         "X_background (e.g. the training data) is required")
    return BlackBoxExplainer(model, X_background, **kwargs)


def explain(model, x, k, method=None, X_background=None, **kwargs):
    """Explain the prediction of ``model`` on instance ``x`` with an explanation of size <= k.

    method: decision tree "gd" (default) or "ga";
            tree ensemble "ga" (default) or "gd";
            black box: ignored (BE).
    Other keyword arguments go to the explainer constructor (see make_explainer)."""
    explainer = make_explainer(model, X_background, **kwargs)
    if isinstance(explainer, BlackBoxExplainer) or method is None:
        return explainer.explain(x, k)
    return explainer.explain(x, k, method=method)
