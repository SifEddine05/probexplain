# probexplain

**probexplain** is a Python library that explains the predictions of classifiers with **probable explanations**.

A probable explanation of the prediction of a model on an instance `x` is a small set of conditions on the features of `x`, at most `k` of them. When these conditions are fixed and the other features vary, the prediction stays the same with high probability. The **error** of an explanation is the probability that the prediction changes. Its **precision** is `1 - error`.

The library brings together the methods of three articles:

| Article | Models | Explainer |
|---|---|---|
| **[1] UAI 2023**: probable explanations of decision trees, greedy descent (GD) and greedy ascent (GA). The error is computed exactly, uniform over the Boolean variables of the tree. | `DecisionTreeClassifier` | `DecisionTreeExplainer` |
| **[2] k-pseudo-probable explanations** for weighted tree ensembles: GA (with a curvature-based stopping rule) and GD. The error is the average, over the trees, of their errors weighted by `1 - P_i` (`P_i`: confidence of tree `i` on `x`). | `RandomForestClassifier`, `ExtraTreesClassifier`, `BaggingClassifier` of trees | `TreeEnsembleExplainer` |
| **[3] BE (BanditExplain)**: explanations for any black-box model, with an (η, δ)-PAC guarantee. The error is estimated by Monte-Carlo sampling on background data, and only `predict` is used. | any model with `predict` (neural networks, CNN, SVM, boosting, pipelines...) | `BlackBoxExplainer` |

**References**

1. L. Bounia, F. Koriche. *Approximating Probabilistic Explanations via Supermodular Minimization*. Proceedings of the 39th Conference on Uncertainty in Artificial Intelligence (UAI 2023), PMLR 216.
2. *k-Pseudo-Probable Explanations for Weighted Tree Ensembles*. Authors, venue, year: **to be completed**.
3. *BE (BanditExplain)*. Title, authors, venue, year: **to be completed**.

## Project organization

```
probexplain/
├── README.md
├── pyproject.toml              # package description (dependencies: numpy, scikit-learn)
├── requirements-examples.txt   # extra dependencies, only to run the examples
├── probexplain/                # the library
│   ├── __init__.py             # explain() and make_explainer(): choose the explainer from the model
│   ├── tree.py                 # [1] UAI 2023: DecisionTreeExplainer (GD, GA)
│   ├── ensemble.py             # [2] k-pseudo: TreeEnsembleExplainer (GA, GD)
│   ├── blackbox.py             # [3] BE: BlackBoxExplainer
│   ├── explanation.py          # Explanation: the result returned by every explainer
│   ├── _boolean.py             # binarization of the trees (thresholds -> Boolean variables)
│   └── _counting.py            # exact model counting on orthogonal DNFs
└── examples/                   # executable examples
    ├── data/adult.csv
    ├── example_1_decision_tree_adult.py
    ├── example_2_random_forest_adult.py
    ├── example_3_neural_network_adult.py
    ├── example_4_cnn_mnist_heatmap.py
    └── example_5_text_be.py
```

## Installation

Requires Python >= 3.9. The library only depends on `numpy` and `scikit-learn`.

**From GitHub.** Use this to use the library in your own code:

```bash
pip install git+https://github.com/.../probexplain.git
```

**From a clone.** Use this to get the code and the examples:

```bash
git clone https://github.com/.../probexplain.git
cd probexplain
python -m venv .venv
.venv\Scripts\activate          # Windows   (Linux / macOS: source .venv/bin/activate)
pip install -e .
```

**Dependencies of the examples.** `requirements-examples.txt` (pandas, matplotlib, torch) is needed **only to run the provided examples**. The library does not need it:

```bash
pip install -r requirements-examples.txt
```

`torch` is only used by the CNN example. Its CPU build is lighter: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Usage

### `explain`: the explainer is chosen from the model

```python
from probexplain import explain

e = explain(model, x, k=4)          # model: fitted classifier, x: one instance (1-D array)
print(e)
```

`explain` picks the method from the type of the model:

| Model | Method used | `method` |
|---|---|---|
| `DecisionTreeClassifier` | UAI 2023 | `"gd"` (default) or `"ga"` |
| `RandomForestClassifier`, `ExtraTreesClassifier`, `BaggingClassifier` of trees | k-pseudo | `"ga"` (default) or `"gd"` |
| any other model | BE | ignored; `X_background` is required |

```python
e = explain(decision_tree, x, k=4, method="ga", feature_names=X.columns)
e = explain(random_forest, x, k=4, method="gd")
e = explain(neural_net, x, k=4, X_background=X_train, categorical_features=["workclass", ...],
            eta=0.1, delta=0.05, random_state=0)
```

### Choosing the explainer yourself

You can also build one of the three explainers directly. It is faster when you explain many instances, because the model is analysed only once:

```python
from probexplain import DecisionTreeExplainer, TreeEnsembleExplainer, BlackBoxExplainer

ex = DecisionTreeExplainer(dt, feature_names=X.columns)
e = ex.explain(x, k=3, method="gd")

ex = TreeEnsembleExplainer(rf, feature_names=X.columns)
e = ex.explain(x, k=4, method="ga")

ex = BlackBoxExplainer(nn, X_train, categorical_features=[...], n_bins=3, eta=0.1, delta=0.05)
e = ex.explain(x, k=4)
```

### Parameters

**Common to all explainers**

| Parameter | Meaning |
|---|---|
| `x` | the instance to explain: a 1-D array of numeric values. Categorical columns must be encoded as integer codes. |
| `k` | maximal size of the explanation (number of conditions) |
| `feature_names` | names of the features, used to print the conditions |

**`DecisionTreeExplainer(model, feature_names=None)`, then `.explain(x, k, method="gd")`**

| Parameter | Meaning |
|---|---|
| `method` | `"gd"`: greedy descent; starts from the path of `x` in the tree and removes conditions. `"ga"`: greedy ascent; starts from the empty set and adds conditions. |

**`TreeEnsembleExplainer(model, feature_names=None)`, then `.explain(x, k, method="ga")`**

| Parameter | Meaning |
|---|---|
| `method` | `"ga"` (greedy ascent) or `"gd"` (greedy descent) |

The explanation is chosen among the conditions of the direct reason: the union of the paths of the trees that vote for the prediction.

**`BlackBoxExplainer(model, X_background, categorical_features=(), n_bins=3, eta=0.1, delta=0.05, feature_names=None, cat_decoders=None, random_state=None)`, then `.explain(x, k)`**

| Parameter | Meaning |
|---|---|
| `X_background` | data the free features are sampled from, e.g. the training set |
| `categorical_features` | indices or column names of the categorical features. The other features are numeric. |
| `n_bins` | a numeric condition is the quantile interval of `x`, out of `n_bins` intervals |
| `eta`, `delta` | PAC guarantee: each error is estimated within `eta`, with probability at least `1 - delta`. A smaller `eta` gives more precise but slower explanations (about 1/eta² model queries per estimate). |
| `cat_decoders` | optional, display only: `{feature index: list of category names}`, to print the categories instead of their integer codes (see below) |
| `random_state` | seed of the sampling |

**`cat_decoders`.** The library only works with numbers, so categorical columns are encoded as integer codes before training. Without `cat_decoders`, the explanation prints these codes, e.g. `gender = 1`. With `cat_decoders`, it prints the category name, e.g. `gender = Male`.

### Result: `Explanation`

| Attribute | Content |
|---|---|
| `e.prediction` | the predicted class being explained |
| `e.conditions` | the conditions, readable (`age in [37, 90]`, `capital-gain <= 7074`, `gender = Male`...) |
| `e.features` | indices of the features used |
| `e.literals` | the explanation as Boolean literals of the model's binarization |
| `e.error`, `e.precision` | error of the explanation and `1 - error` |
| `e.size` | number of literals |

## Examples

Five executable scripts in `examples/`. Run them from the root of the project after installing `requirements-examples.txt`:

```bash
python examples/example_1_decision_tree_adult.py
python examples/example_2_random_forest_adult.py
python examples/example_3_neural_network_adult.py
python examples/example_4_cnn_mnist_heatmap.py
python examples/example_5_text_be.py
```

Their parameters (`K`, `ETA`, `DELTA`, instances to explain...) are constants at the top of each script.

| Script | Dataset | Model | Explanations | Time (CPU) |
|---|---|---|---|---|
| `example_1_decision_tree_adult.py` | Adult | decision tree (depth 8) | UAI 2023, GD and GA | a few seconds |
| `example_2_random_forest_adult.py` | Adult | random forest (20 trees, depth 6) | k-pseudo, GA and GD | a few seconds |
| `example_3_neural_network_adult.py` | Adult | neural network (MLP, 64 neurons) | BE | about 1 min |
| `example_4_cnn_mnist_heatmap.py` | MNIST, 3 vs 8 | CNN (PyTorch) | BE + heatmaps | about 2 min |
| `example_5_text_be.py` | 20 Newsgroups, sci.med vs sci.space | neural network (MLP, 32 neurons) | BE on words | about 20 s |

**Tabular data: Adult** (`examples/data/adult.csv`). The task is to predict whether income is `<=50K` or `>50K` from 14 features: age, education, occupation, marital status, hours per week, capital gain... The categorical columns are encoded as integer codes. The same data is explained by the three methods, which gives explanations like:

```
Explanation [uai2023-gd] of prediction '<=50K'
  size = 3, error = 0.25 (precision 0.75)
  * 0.5 < relationship <= 4.5
  * capital-gain <= 7074
```

**Images: MNIST 3 vs 8.** The script follows the notebook `examples/GDB_MNIST_heatmap_CNN (1).ipynb`:

- Each 28x28 image is averaged by blocks of `BLOCK_SIZE` x `BLOCK_SIZE` pixels. With `BLOCK_SIZE = 4`, this gives 49 features.
- The CNN is trained on this grid of blocks.
- BE selects at most `K` blocks.
- The heatmap shows the selected blocks in red on the image. There is one PDF per instance in `examples/results/`.

`BLOCK_SIZE = 2` (196 features) gives finer heatmaps but is much slower.

**Text: 20 Newsgroups, sci.med vs sci.space** (downloaded by scikit-learn the first time). Each of the 50 most discriminant words is a feature, present or absent, and BE explains the predictions of the network with them. The words of the explanation are marked `>>word<<` in the document:

```
Explanation [be] of prediction 'sci.space'
  size = 4, error = 0.142 (precision 0.858)
  * medical = absent
  * satellite = present
  * satellites = present
  * spacecraft = present
```
