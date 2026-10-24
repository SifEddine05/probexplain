"""Example 4 - CNN on MNIST (DIGIT_A vs DIGIT_B): BE explanations and heatmaps.

Same pipeline as the notebook GDB_MNIST_heatmap_CNN:
  * a 28x28 image is averaged by blocks of BLOCK_SIZE x BLOCK_SIZE pixels (super-pixels); each
    block is one numeric feature (4 -> 7x7 = 49 features; 2 -> 196, finer but much slower);
  * the CNN is trained on this grid of blocks, so it sees exactly what BE fixes / frees;
  * BE (BlackBoxExplainer, the classic method) explains the prediction;
  * the heatmap shows the blocks of the explanation on the original 28x28 image.

Needs PyTorch and matplotlib (pip install torch matplotlib).
Run:  python examples/example_4_cnn_mnist_heatmap.py
MNIST is downloaded by scikit-learn the first time; the figures go to examples/results/.
"""
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.datasets import fetch_openml  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch import nn  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from probexplain import BlackBoxExplainer  # noqa: E402

warnings.filterwarnings("ignore")   # empty quantile bins of the always black blocks
np.random.seed(0)
torch.manual_seed(0)

DIGIT_A, DIGIT_B = 3, 8
BLOCK_SIZE = 4                 # 28 must be divisible by BLOCK_SIZE (1, 2, 4, 7, 14)
N_BINS = 3
IMG_SHAPE = (28, 28)
INSTANCE_INDEX = [0, 1, 5, 10]  # test instances to explain
K = 7                          # maximal size of the explanations (in blocks)
ETA, DELTA = 0.15, 0.1         # PAC parameters of BE
OUTPUT_DIR = os.path.join(HERE, "results")


def pool_images(images_flat, block_size, img_shape=(28, 28)):
    """Average each image by blocks: returns (pooled features (n, rows * cols), rows, cols)."""
    H, W = img_shape
    rows, cols = H // block_size, W // block_size
    images = images_flat.reshape(-1, H, W)[:, :rows * block_size, :cols * block_size]
    pooled = images.reshape(-1, rows, block_size, cols, block_size).mean(axis=(2, 4))
    return pooled.reshape(-1, rows * cols), rows, cols


# ---- MNIST, two digits, grid of blocks
print("Loading MNIST (downloaded the first time)...")
X_all, y_all = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser="auto")
X_all, y_all = X_all.astype(np.float32), y_all.astype(int)
mask = (y_all == DIGIT_A) | (y_all == DIGIT_B)
images_pair = X_all[mask]
labels_pair = (y_all[mask] == DIGIT_B).astype(np.int64)      # 0 = DIGIT_A, 1 = DIGIT_B
X_pooled, ROWS, COLS = pool_images(images_pair, BLOCK_SIZE, IMG_SHAPE)
n_blocks = ROWS * COLS
print(f"{DIGIT_A} vs {DIGIT_B}: {len(images_pair)} images, grid {ROWS}x{COLS} = {n_blocks} features")

X_train, X_test, y_train, y_test, img_train, img_test = train_test_split(
    X_pooled, labels_pair, images_pair.reshape(-1, *IMG_SHAPE), test_size=0.3, random_state=42,
    stratify=labels_pair)


# ---- CNN on the grid of blocks
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, ceil_mode=True),
            nn.Flatten(), nn.LazyLinear(32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        return self.net(x)


class CNNWrapper:
    """predict(X_flat) -> labels, on flat vectors of blocks (the format BE works with)."""

    def __init__(self, model, rows, cols):
        self.model, self.rows, self.cols = model.eval(), rows, cols

    @torch.no_grad()
    def predict(self, X_flat):
        X_img = torch.as_tensor(np.atleast_2d(np.asarray(X_flat, dtype=np.float32)) / 255.0)
        return self.model(X_img.reshape(-1, 1, self.rows, self.cols)).argmax(1).numpy()


def train(model, X, y, epochs=10, batch_size=64):
    X = torch.as_tensor(X / 255.0, dtype=torch.float32).reshape(-1, 1, ROWS, COLS)
    y = torch.as_tensor(y)
    model(X[:1])                                  # initializes the lazy layer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for epoch in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"epoch {epoch + 1}/{epochs}: loss {loss.item():.4f}")


cnn = CNN()
train(cnn, X_train, y_train)
h = CNNWrapper(cnn, ROWS, COLS)
print(f"\nCNN accuracy ({DIGIT_A} vs {DIGIT_B}, grid {ROWS}x{COLS}): "
      f"{np.mean(h.predict(X_test) == y_test):.4f}\n")

# ---- BE explanations and heatmaps
ex = BlackBoxExplainer(h, X_train, n_bins=N_BINS, eta=ETA, delta=DELTA, random_state=0,
                       feature_names=[f"block_{r}_{c}" for r in range(ROWS) for c in range(COLS)])
os.makedirs(OUTPUT_DIR, exist_ok=True)
digit = {0: DIGIT_A, 1: DIGIT_B}
print(f"{'instance':>8} {'true':>5} {'pred':>5} {'size':>5} {'error':>7} {'time (s)':>9}")
for idx in INSTANCE_INDEX:
    t0 = time.perf_counter()
    e = ex.explain(X_test[idx], k=K)
    elapsed = time.perf_counter() - t0

    # selected blocks -> 28x28 heatmap (each block repeated BLOCK_SIZE x BLOCK_SIZE times)
    block_mask = np.zeros(n_blocks)
    block_mask[e.features] = 1
    heatmap = np.kron(block_mask.reshape(ROWS, COLS), np.ones((BLOCK_SIZE, BLOCK_SIZE)))

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_test[idx], cmap="gray")
    ax.imshow(heatmap, cmap="Reds", alpha=0.9 * heatmap)
    ax.set_title(f"True class = {digit[y_test[idx]]} | Predicted class = {digit[e.prediction]}\n"
                 f"Explanation size = {e.size} | Error ≈ {e.error:.3f}", fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, f"heatmap_{DIGIT_A}_vs_{DIGIT_B}_instance{idx}.pdf"),
                bbox_inches="tight")
    plt.close(fig)
    print(f"{idx:>8} {digit[y_test[idx]]:>5} {digit[e.prediction]:>5} {e.size:>5} "
          f"{e.error:>7.3f} {elapsed:>9.1f}")

print("\nFigures saved in", OUTPUT_DIR)
