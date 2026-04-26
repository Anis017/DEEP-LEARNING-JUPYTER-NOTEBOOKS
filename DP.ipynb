import nbformat as nbf
import os

OUT = "/home/claude/UDL_Notebooks"
os.makedirs(OUT, exist_ok=True)

def nb(cells):
    n = nbf.v4.new_notebook()
    n.cells = cells
    return n

def md(src): return nbf.v4.new_markdown_cell(src)
def code(src): return nbf.v4.new_code_cell(src)

SETUP = """import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
np.random.seed(42)
"""

notebooks = {}

# ── 1.1 Background Mathematics ──────────────────────────────────────────────
notebooks["Notebook_1_1_Background_mathematics"] = nb([
md("# Notebook 1.1 – Background Mathematics\nFoundations of linear algebra, probability, and calculus used throughout deep learning."),
code(SETUP),
md("## 1. Vectors and Matrices"),
code("""
# Vectors
v = np.array([1, 2, 3])
w = np.array([4, 5, 6])
print("Dot product:", np.dot(v, w))
print("Norm:", np.linalg.norm(v))

# Matrix multiplication
A = np.array([[1,2],[3,4]])
B = np.array([[5,6],[7,8]])
print("A @ B =\\n", A @ B)
"""),
md("## 2. Derivatives and Gradients"),
code("""
# Numerical gradient check
def f(x): return x**3 - 2*x + 1

def numerical_grad(f, x, h=1e-5):
    return (f(x+h) - f(x-h)) / (2*h)

x_vals = np.linspace(-2, 2, 300)
y_vals = f(x_vals)
grads  = np.array([numerical_grad(f, xi) for xi in x_vals])

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(x_vals, y_vals, 'b'); axes[0].set_title("f(x) = x³ - 2x + 1")
axes[1].plot(x_vals, grads, 'r'); axes[1].set_title("df/dx")
plt.tight_layout(); plt.savefig("math_derivatives.png", dpi=100)
plt.show(); print("Plot saved.")
"""),
md("## 3. Chain Rule"),
code("""
# Chain rule: d/dx [sin(x²)] = cos(x²) · 2x
x = np.linspace(-2, 2, 300)
y = np.sin(x**2)
dydx_analytical = np.cos(x**2) * 2 * x
dydx_numerical  = np.gradient(y, x)

plt.figure(figsize=(8,4))
plt.plot(x, dydx_analytical, 'b', label='Analytical')
plt.plot(x, dydx_numerical,  'r--', label='Numerical')
plt.legend(); plt.title("Chain Rule: d/dx sin(x²)")
plt.tight_layout(); plt.savefig("chain_rule.png", dpi=100)
plt.show()
"""),
md("## 4. Probability Basics"),
code("""
from scipy.stats import norm, multivariate_normal

x = np.linspace(-4, 4, 400)
for mu, sigma in [(0,1),(0,2),(1,1)]:
    plt.plot(x, norm.pdf(x, mu, sigma), label=f'μ={mu}, σ={sigma}')
plt.legend(); plt.title("Gaussian PDFs")
plt.tight_layout(); plt.savefig("gaussians.png", dpi=100)
plt.show()
"""),
md("## 5. Bayes' Theorem"),
code("""
# P(A|B) = P(B|A) * P(A) / P(B)
P_A   = 0.01   # prior: disease prevalence
P_B_A = 0.95   # sensitivity
P_B   = P_B_A * P_A + 0.10 * (1 - P_A)   # total probability
posterior = P_B_A * P_A / P_B
print(f"P(disease | positive test) = {posterior:.4f}")
"""),
])

# ── 2.1 Supervised Learning ──────────────────────────────────────────────────
notebooks["Notebook_2_1_Supervised_learning"] = nb([
md("# Notebook 2.1 – Supervised Learning\nCore concepts: inputs, outputs, loss functions, and model fitting."),
code(SETUP),
md("## 1. Generate Toy Data"),
code("""
np.random.seed(0)
X = np.random.uniform(-3, 3, 50)
y = 0.5 * X**2 - X + 2 + np.random.randn(50) * 0.8

plt.figure(figsize=(7,4))
plt.scatter(X, y, c='steelblue', alpha=0.7, label='Data')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Toy Regression Dataset')
plt.legend(); plt.tight_layout(); plt.savefig("supervised_data.png", dpi=100)
plt.show()
"""),
md("## 2. Linear Regression by Least Squares"),
code("""
# Design matrix
Phi = np.c_[np.ones_like(X), X, X**2]
# Closed-form solution: w = (Φ^T Φ)^{-1} Φ^T y
w = np.linalg.lstsq(Phi, y, rcond=None)[0]
print("Weights:", w)

x_plot = np.linspace(-3, 3, 300)
Phi_plot = np.c_[np.ones(300), x_plot, x_plot**2]
y_hat = Phi_plot @ w

plt.figure(figsize=(7,4))
plt.scatter(X, y, c='steelblue', alpha=0.7, label='Data')
plt.plot(x_plot, y_hat, 'r', lw=2, label='Fit (degree-2)')
plt.legend(); plt.tight_layout(); plt.savefig("supervised_fit.png", dpi=100)
plt.show()
"""),
md("## 3. Generalisation & Train/Test Split"),
code("""
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=1)

for deg in [1, 2, 5, 10]:
    Phi_tr = np.column_stack([X_tr**d for d in range(deg+1)])
    Phi_te = np.column_stack([X_te**d for d in range(deg+1)])
    w_d = np.linalg.lstsq(Phi_tr, y_tr, rcond=None)[0]
    tr_mse = mean_squared_error(y_tr, Phi_tr @ w_d)
    te_mse = mean_squared_error(y_te, Phi_te @ w_d)
    print(f"Degree {deg:2d} | Train MSE: {tr_mse:.3f} | Test MSE: {te_mse:.3f}")
"""),
])

# ── 3.1 Shallow Networks I ───────────────────────────────────────────────────
notebooks["Notebook_3_1_Shallow_networks_I"] = nb([
md("# Notebook 3.1 – Shallow Networks I\nA single hidden-layer network as a composition of linear maps and activations."),
code(SETUP),
md("## 1. Single Neuron"),
code("""
def relu(z): return np.maximum(0, z)

x = np.linspace(-3, 3, 300)
# Neuron: h = ReLU(w*x + b)
w, b = 1.5, -0.5
h = relu(w * x + b)

plt.figure(figsize=(7,4))
plt.plot(x, h, 'b', lw=2, label=f'ReLU({w}x + {b})')
plt.axhline(0, color='k', lw=0.5); plt.legend()
plt.title("Single ReLU Neuron"); plt.tight_layout()
plt.savefig("single_neuron.png", dpi=100); plt.show()
"""),
md("## 2. Shallow Network (1 hidden layer, 4 neurons)"),
code("""
def shallow_net(x, W1, b1, w2, b2):
    H = relu(np.outer(x, W1) + b1)   # (N, D)
    return H @ w2 + b2

np.random.seed(3)
D = 6
W1 = np.random.randn(D)
b1 = np.random.randn(D)
w2 = np.random.randn(D)
b2 = np.random.randn()

x = np.linspace(-3, 3, 500)
y = shallow_net(x, W1, b1, w2, b2)

plt.figure(figsize=(8,4))
plt.plot(x, y, 'b', lw=2)
plt.title(f"Shallow network output ({D} hidden neurons)")
plt.tight_layout(); plt.savefig("shallow_net.png", dpi=100); plt.show()
"""),
md("## 3. Increasing Width"),
code("""
fig, axes = plt.subplots(2, 3, figsize=(12,6))
for ax, D in zip(axes.flat, [1, 2, 4, 8, 16, 32]):
    W1 = np.random.randn(D); b1 = np.random.randn(D)
    w2 = np.random.randn(D); b2 = np.random.randn()
    y  = shallow_net(x, W1, b1, w2, b2)
    ax.plot(x, y, 'b', lw=1.5); ax.set_title(f"D = {D}")
plt.suptitle("Effect of Width on Shallow Network"); plt.tight_layout()
plt.savefig("shallow_width.png", dpi=100); plt.show()
"""),
])

# ── 3.2 Shallow Networks II ──────────────────────────────────────────────────
notebooks["Notebook_3_2_Shallow_networks_II"] = nb([
md("# Notebook 3.2 – Shallow Networks II\nFitting data with shallow networks; visualising the hidden-layer geometry."),
code(SETUP),
md("## 1. Toy 1-D Regression with a Shallow Net"),
code("""
import torch
import torch.nn as nn
import torch.optim as optim

torch.manual_seed(0)
X_np = np.random.uniform(-3, 3, 60).astype(np.float32)
y_np = np.sin(X_np) + 0.2 * np.random.randn(60).astype(np.float32)

X_t = torch.tensor(X_np).unsqueeze(1)
y_t = torch.tensor(y_np).unsqueeze(1)

class ShallowNet(nn.Module):
    def __init__(self, D=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, D), nn.ReLU(),
            nn.Linear(D, 1)
        )
    def forward(self, x): return self.net(x)

model = ShallowNet(D=32)
opt   = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

losses = []
for epoch in range(1000):
    pred = model(X_t)
    loss = loss_fn(pred, y_t)
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

print(f"Final loss: {losses[-1]:.4f}")

x_plot = torch.linspace(-3, 3, 300).unsqueeze(1)
with torch.no_grad():
    y_plot = model(x_plot).squeeze().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.scatter(X_np, y_np, s=15, c='steelblue', alpha=0.7)
ax1.plot(x_plot.squeeze(), y_plot, 'r', lw=2)
ax1.set_title("Shallow Net Fit")
ax2.plot(losses); ax2.set_yscale('log'); ax2.set_title("Training Loss")
plt.tight_layout(); plt.savefig("shallow_fit.png", dpi=100); plt.show()
"""),
md("## 2. Hidden Activations"),
code("""
# Extract hidden activations for a selection of neurons
with torch.no_grad():
    W1 = model.net[0].weight.numpy().squeeze()   # (D,)
    b1 = model.net[0].bias.numpy()               # (D,)
    W2 = model.net[2].weight.numpy().squeeze()   # (D,)

x_vals = np.linspace(-3, 3, 300)
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, ax in enumerate(axes.flat):
    h = np.maximum(0, W1[i] * x_vals + b1[i])
    ax.plot(x_vals, h * W2[i], label=f'neuron {i}')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_title(f"Neuron {i} contribution")
plt.tight_layout(); plt.savefig("hidden_activations.png", dpi=100); plt.show()
"""),
])

# ── 3.3 Shallow Network Regions ──────────────────────────────────────────────
notebooks["Notebook_3_3_Shallow_network_regions"] = nb([
md("# Notebook 3.3 – Shallow Network Regions\nLinear regions produced by ReLU shallow networks in 2-D input space."),
code(SETUP),
md("## 1. Counting Linear Regions"),
code("""
import torch, torch.nn as nn

def count_regions_1d(model, x_min=-5, x_max=5, n=5000):
    x = torch.linspace(x_min, x_max, n).unsqueeze(1)
    with torch.no_grad():
        y = model(x).squeeze().numpy()
    # Count sign changes in second derivative (breakpoints)
    dy  = np.diff(y)
    ddy = np.diff(dy)
    return int(np.sum(np.abs(ddy) > 1e-6)) + 1

torch.manual_seed(7)
for D in [2, 4, 8, 16, 32]:
    m = nn.Sequential(nn.Linear(1, D), nn.ReLU(), nn.Linear(D, 1))
    r = count_regions_1d(m)
    print(f"D = {D:2d}  |  regions ≈ {r}")
"""),
md("## 2. Visualising 2-D Decision Regions"),
code("""
import torch.nn as nn, torch

torch.manual_seed(1)
D = 8
model2d = nn.Sequential(nn.Linear(2, D), nn.ReLU(), nn.Linear(D, 1))

xx = np.linspace(-3, 3, 200); yy = np.linspace(-3, 3, 200)
XX, YY = np.meshgrid(xx, yy)
grid = torch.tensor(np.c_[XX.ravel(), YY.ravel()], dtype=torch.float32)

with torch.no_grad():
    Z = model2d(grid).squeeze().numpy().reshape(200, 200)

plt.figure(figsize=(6,6))
plt.contourf(XX, YY, Z, levels=50, cmap='RdBu')
plt.colorbar(); plt.title("2-D Shallow Net Output (Random Weights)")
plt.tight_layout(); plt.savefig("2d_regions.png", dpi=100); plt.show()
"""),
])

# ── 3.4 Activation Functions ─────────────────────────────────────────────────
notebooks["Notebook_3_4_Activation_functions"] = nb([
md("# Notebook 3.4 – Activation Functions\nComparison of ReLU, Leaky ReLU, ELU, GELU, Sigmoid, Tanh."),
code(SETUP),
code("""
x = np.linspace(-4, 4, 400)

activations = {
    'ReLU':        lambda x: np.maximum(0, x),
    'Leaky ReLU':  lambda x: np.where(x > 0, x, 0.1*x),
    'ELU':         lambda x: np.where(x > 0, x, np.exp(x)-1),
    'GELU':        lambda x: x * 0.5 * (1 + np.tanh(np.sqrt(2/np.pi)*(x + 0.044715*x**3))),
    'Sigmoid':     lambda x: 1 / (1 + np.exp(-x)),
    'Tanh':        lambda x: np.tanh(x),
    'Swish':       lambda x: x / (1 + np.exp(-x)),
    'Softplus':    lambda x: np.log1p(np.exp(x)),
}

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (name, fn) in zip(axes.flat, activations.items()):
    y = fn(x)
    ax.plot(x, y, 'b', lw=2)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_title(name); ax.set_xlim(-4, 4)
plt.suptitle("Activation Functions", fontsize=13)
plt.tight_layout(); plt.savefig("activations.png", dpi=100); plt.show()
"""),
md("## Derivatives"),
code("""
fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (name, fn) in zip(axes.flat, activations.items()):
    y  = fn(x)
    dy = np.gradient(y, x)
    ax.plot(x, dy, 'r', lw=2)
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5)
    ax.set_title(f"d/dx {name}"); ax.set_xlim(-4, 4)
plt.suptitle("Activation Function Derivatives", fontsize=13)
plt.tight_layout(); plt.savefig("activation_derivatives.png", dpi=100); plt.show()
"""),
])

# ── 4.1 Composing Networks ───────────────────────────────────────────────────
notebooks["Notebook_4_1_Composing_networks"] = nb([
md("# Notebook 4.1 – Composing Networks\nBuilding deep networks by composing simple functions."),
code(SETUP),
code("""
import torch, torch.nn as nn

def make_net(depth, width, activation=nn.ReLU):
    layers = [nn.Linear(1, width), activation()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), activation()]
    layers.append(nn.Linear(width, 1))
    return nn.Sequential(*layers)

torch.manual_seed(42)
x_plot = torch.linspace(-3, 3, 300).unsqueeze(1)

fig, axes = plt.subplots(2, 3, figsize=(13, 7))
configs = [(1,8),(2,8),(4,8),(1,32),(2,32),(4,32)]
for ax, (depth, width) in zip(axes.flat, configs):
    net = make_net(depth, width)
    with torch.no_grad():
        y = net(x_plot).squeeze().numpy()
    ax.plot(x_plot.squeeze(), y, 'b', lw=1.5)
    ax.set_title(f"Depth={depth}, Width={width}")
plt.suptitle("Composed Networks (Random Weights)"); plt.tight_layout()
plt.savefig("composing_networks.png", dpi=100); plt.show()
"""),
])

# ── 4.2 Clipping Functions ───────────────────────────────────────────────────
notebooks["Notebook_4_2_Clipping_functions"] = nb([
md("# Notebook 4.2 – Clipping Functions\nHow ReLU networks create clipped/folded linear functions."),
code(SETUP),
code("""
x = np.linspace(-3, 3, 500)

def relu(z): return np.maximum(0, z)
def clip(z, lo, hi): return np.clip(z, lo, hi)

# A two-neuron decomposition of a tent function
h1 = relu(x + 1)
h2 = relu(-x + 1)
tent = h1 + h2 - 1  # triangular

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].plot(x, h1, label='ReLU(x+1)'); axes[0].legend(); axes[0].set_title("Neuron 1")
axes[1].plot(x, h2, label='ReLU(-x+1)', c='r'); axes[1].legend(); axes[1].set_title("Neuron 2")
axes[2].plot(x, tent, label='Tent = h1+h2-1', c='g'); axes[2].legend(); axes[2].set_title("Tent function")
plt.suptitle("Clipping & Tent Functions"); plt.tight_layout()
plt.savefig("clipping.png", dpi=100); plt.show()
"""),
md("## Stacking to Approximate Any Function"),
code("""
# Approximate sin(x) with stacked tent functions
def tent_approx(x, n_tents=10):
    knots = np.linspace(-3, 3, n_tents + 2)[1:-1]
    width = knots[1] - knots[0]
    out = np.zeros_like(x)
    for k in knots:
        h = np.maximum(0, 1 - np.abs(x - k) / width)
        out += np.sin(k) * h
    return out

x = np.linspace(-3, 3, 500)
y_true = np.sin(x)

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, n in zip(axes, [5, 10, 20]):
    y_approx = tent_approx(x, n)
    ax.plot(x, y_true, 'k', label='sin(x)')
    ax.plot(x, y_approx, 'r--', label=f'{n} tents')
    ax.legend(); ax.set_title(f"{n} tent functions")
plt.suptitle("Piecewise Linear Approximation"); plt.tight_layout()
plt.savefig("tent_approx.png", dpi=100); plt.show()
"""),
])

# ── 4.3 Deep Networks ────────────────────────────────────────────────────────
notebooks["Notebook_4_3_Deep_networks"] = nb([
md("# Notebook 4.3 – Deep Networks\nDeep vs shallow: expressive power and the folding argument."),
code(SETUP),
code("""
import torch, torch.nn as nn

torch.manual_seed(5)
x_t = torch.linspace(-3, 3, 500).unsqueeze(1)

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
for ax, (d, w) in zip(axes.flat, [(1,4),(2,4),(4,4),(8,4),(1,32),(2,32),(4,32),(8,32)]):
    layers = [nn.Linear(1, w), nn.ReLU()]
    for _ in range(d - 1): layers += [nn.Linear(w, w), nn.ReLU()]
    layers.append(nn.Linear(w, 1))
    net = nn.Sequential(*layers)
    with torch.no_grad():
        y = net(x_t).squeeze().numpy()
    ax.plot(x_t.squeeze(), y, lw=1.5)
    ax.set_title(f"D={d}, W={w}")
plt.suptitle("Deep Network Outputs (Random Weights)")
plt.tight_layout(); plt.savefig("deep_nets.png", dpi=100); plt.show()
"""),
md("## Parameter Count vs Depth"),
code("""
def param_count(depth, width, input_dim=1, output_dim=1):
    if depth == 1:
        return (input_dim + 1)*width + (width + 1)*output_dim
    return ((input_dim + 1)*width +
            (depth - 1)*(width + 1)*width +
            (width + 1)*output_dim)

print(f"{'Depth':>6} {'Width':>6} {'Params':>10}")
for d in [1, 2, 4, 8]:
    for w in [8, 16, 32]:
        print(f"{d:>6} {w:>6} {param_count(d,w):>10,}")
"""),
])

# ── 5.1 Least Squares Loss ───────────────────────────────────────────────────
notebooks["Notebook_5_1_Least_squares_loss"] = nb([
md("# Notebook 5.1 – Least Squares Loss\nMSE loss, its landscape, and connection to maximum likelihood under Gaussian noise."),
code(SETUP),
code("""
np.random.seed(1)
X = np.random.uniform(-2, 2, 30)
y = 2*X + 1 + np.random.randn(30) * 0.8

def mse(w0, w1):
    pred = w0 + w1 * X
    return np.mean((pred - y)**2)

w0_grid = np.linspace(-2, 5, 100)
w1_grid = np.linspace(-1, 5, 100)
W0, W1  = np.meshgrid(w0_grid, w1_grid)
Z = np.vectorize(mse)(W0, W1)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
cp = axes[0].contourf(W0, W1, Z, levels=50, cmap='viridis')
plt.colorbar(cp, ax=axes[0]); axes[0].set_xlabel('w0'); axes[0].set_ylabel('w1')
axes[0].set_title("MSE Loss Landscape")

# Optimal solution
w1_opt = np.cov(X, y)[0,1] / np.var(X)
w0_opt = np.mean(y) - w1_opt * np.mean(X)
axes[0].scatter([w0_opt], [w1_opt], c='red', s=100, zorder=5, label='Optimum')
axes[0].legend()

axes[1].scatter(X, y, c='steelblue', alpha=0.7)
axes[1].plot(np.sort(X), w0_opt + w1_opt * np.sort(X), 'r', lw=2)
axes[1].set_title("Best Fit Line")

plt.tight_layout(); plt.savefig("least_squares.png", dpi=100); plt.show()
print(f"Optimal w0={w0_opt:.3f}, w1={w1_opt:.3f}")
"""),
])

# ── 5.2 Binary Cross-Entropy ─────────────────────────────────────────────────
notebooks["Notebook_5_2_Binary_cross-entropy_loss"] = nb([
md("# Notebook 5.2 – Binary Cross-Entropy Loss\nLoss for binary classification; logistic regression."),
code(SETUP),
code("""
sigmoid = lambda z: 1 / (1 + np.exp(-z))

# BCE loss surface
def bce(y_true, y_pred_prob):
    eps = 1e-8
    return -np.mean(y_true * np.log(y_pred_prob + eps) +
                    (1 - y_true) * np.log(1 - y_pred_prob + eps))

# Toy 1-D binary classification
np.random.seed(3)
N = 80
X = np.r_[np.random.randn(N//2) - 2, np.random.randn(N//2) + 2]
y = np.r_[np.zeros(N//2), np.ones(N//2)]

def logistic_fit(X, y, lr=0.1, epochs=500):
    w, b = 0.0, 0.0
    losses = []
    for _ in range(epochs):
        z    = w * X + b
        prob = sigmoid(z)
        loss = bce(y, prob)
        losses.append(loss)
        # Gradients
        err  = prob - y
        w   -= lr * np.mean(err * X)
        b   -= lr * np.mean(err)
    return w, b, losses

w_fit, b_fit, losses = logistic_fit(X, y)
print(f"w={w_fit:.3f}, b={b_fit:.3f}")

x_p = np.linspace(-6, 6, 300)
prob_p = sigmoid(w_fit * x_p + b_fit)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.scatter(X, y, c=['b' if yi==0 else 'r' for yi in y], alpha=0.6)
ax1.plot(x_p, prob_p, 'k', lw=2, label='Logistic fit')
ax1.axhline(0.5, ls='--', c='grey'); ax1.legend(); ax1.set_title("Logistic Regression")
ax2.plot(losses); ax2.set_title("BCE Loss"); ax2.set_xlabel("Epoch")
plt.tight_layout(); plt.savefig("bce.png", dpi=100); plt.show()
"""),
])

# ── 5.3 Multiclass Cross-Entropy ─────────────────────────────────────────────
notebooks["Notebook_5_3_Multiclass_cross-entropy_loss"] = nb([
md("# Notebook 5.3 – Multiclass Cross-Entropy Loss\nSoftmax + cross-entropy for K-class classification."),
code(SETUP),
code("""
def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)

def cross_entropy(y_prob, y_true):
    N = len(y_true)
    return -np.mean(np.log(y_prob[np.arange(N), y_true] + 1e-8))

# Toy 3-class dataset
from sklearn.datasets import make_blobs
X_data, y_data = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=0)

# Train simple softmax classifier with SGD
K = 3
W = np.zeros((2, K)); b = np.zeros(K)
lr, epochs = 0.05, 300
losses = []

for ep in range(epochs):
    logits = X_data @ W + b
    probs  = softmax(logits)
    loss   = cross_entropy(probs, y_data)
    losses.append(loss)
    # Gradient
    dL_dz  = probs.copy()
    dL_dz[np.arange(len(y_data)), y_data] -= 1
    dL_dz /= len(y_data)
    W -= lr * X_data.T @ dL_dz
    b -= lr * dL_dz.sum(axis=0)

# Decision boundary
xx, yy = np.meshgrid(np.linspace(-6,6,200), np.linspace(-6,6,200))
grid_logits = np.c_[xx.ravel(), yy.ravel()] @ W + b
preds = np.argmax(grid_logits, axis=1).reshape(200,200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.contourf(xx, yy, preds, alpha=0.3, cmap='tab10')
ax1.scatter(*X_data.T, c=y_data, cmap='tab10', edgecolors='k', s=30)
ax1.set_title("Softmax Classifier Decision Regions")
ax2.plot(losses); ax2.set_title("Cross-Entropy Loss")
plt.tight_layout(); plt.savefig("multiclass_ce.png", dpi=100); plt.show()
print(f"Final loss: {losses[-1]:.4f}")
"""),
])

# ── 6.1 Line Search ──────────────────────────────────────────────────────────
notebooks["Notebook_6_1_Line_search"] = nb([
md("# Notebook 6.1 – Line Search\nFinding step size along gradient direction."),
code(SETUP),
code("""
def rosenbrock(x, a=1, b=100):
    return (a - x[0])**2 + b*(x[1] - x[0]**2)**2

def grad_rosenbrock(x, a=1, b=100):
    return np.array([
        -2*(a - x[0]) - 4*b*x[0]*(x[1] - x[0]**2),
         2*b*(x[1] - x[0]**2)
    ])

# Backtracking line search
def backtrack(f, grad_f, x, d, alpha=1.0, rho=0.5, c=1e-4):
    f0 = f(x); g0 = grad_f(x)
    while f(x + alpha*d) > f0 + c*alpha*np.dot(g0, d):
        alpha *= rho
    return alpha

# Gradient descent with line search
x = np.array([-1.5, 1.5])
path = [x.copy()]
for _ in range(200):
    g = grad_rosenbrock(x)
    d = -g
    alpha = backtrack(rosenbrock, grad_rosenbrock, x, d)
    x = x + alpha * d
    path.append(x.copy())
    if np.linalg.norm(g) < 1e-6: break

path = np.array(path)
print(f"Solution: {path[-1]}, f={rosenbrock(path[-1]):.6f}")

xx = np.linspace(-2, 2, 200); yy = np.linspace(-1, 3, 200)
XX, YY = np.meshgrid(xx, yy)
ZZ = rosenbrock(np.array([XX, YY]))

plt.figure(figsize=(8,6))
plt.contour(XX, YY, np.log(ZZ+1), levels=30, cmap='viridis')
plt.plot(*path.T, 'r.-', ms=3, lw=1, label='GD + line search')
plt.scatter(*path[-1], c='r', s=100, zorder=5)
plt.legend(); plt.title("Line Search on Rosenbrock")
plt.tight_layout(); plt.savefig("line_search.png", dpi=100); plt.show()
"""),
])

# ── 6.2 Gradient Descent ────────────────────────────────────────────────────
notebooks["Notebook_6_2_Gradient_descent"] = nb([
md("# Notebook 6.2 – Gradient Descent\nFull-batch gradient descent with various learning rates."),
code(SETUP),
code("""
# Fit a simple model: y = w*x, MSE loss
np.random.seed(2)
X = np.random.randn(100); y = 3.0 * X + np.random.randn(100) * 0.5

def loss(w): return np.mean((w*X - y)**2)
def grad(w): return 2 * np.mean((w*X - y) * X)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

w_range = np.linspace(0, 6, 300)
ax1.plot(w_range, [loss(w) for w in w_range], 'k')
ax1.set_xlabel('w'); ax1.set_ylabel('Loss'); ax1.set_title("Loss Landscape")

for lr in [0.001, 0.01, 0.1, 0.5]:
    w  = 0.0
    ws = [w]
    for _ in range(200):
        w -= lr * grad(w)
        ws.append(w)
    ax2.plot(ws, label=f'lr={lr}')

ax2.axhline(3.0, ls='--', c='k', label='True w=3')
ax2.set_xlabel('Iteration'); ax2.set_title("Convergence for Different LRs")
ax2.legend(); plt.tight_layout()
plt.savefig("gradient_descent.png", dpi=100); plt.show()
"""),
])

# ── 6.3 SGD ─────────────────────────────────────────────────────────────────
notebooks["Notebook_6_3_Stochastic_gradient_descent"] = nb([
md("# Notebook 6.3 – Stochastic Gradient Descent\nMini-batch SGD and the noise it introduces."),
code(SETUP),
code("""
np.random.seed(0)
N = 500
X = np.random.randn(N); y = 2.0 * X + np.random.randn(N) * 1.0

def sgd(X, y, batch_size, lr, epochs=20):
    w = 0.0; losses = []
    for ep in range(epochs):
        idx = np.random.permutation(N)
        ep_loss = []
        for i in range(0, N, batch_size):
            xi = X[idx[i:i+batch_size]]; yi = y[idx[i:i+batch_size]]
            g  = 2 * np.mean((w*xi - yi) * xi)
            w -= lr * g
            ep_loss.append(np.mean((w*X - y)**2))
        losses.append(np.mean(ep_loss))
    return w, losses

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, bs in zip(axes, [1, 32, N]):
    w_fit, losses = sgd(X, y, bs, lr=0.01)
    ax.plot(losses)
    ax.set_title(f"Batch size={bs}, w≈{w_fit:.2f}")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
plt.suptitle("SGD: Effect of Batch Size"); plt.tight_layout()
plt.savefig("sgd.png", dpi=100); plt.show()
"""),
])

# ── 6.4 Momentum ─────────────────────────────────────────────────────────────
notebooks["Notebook_6_4_Momentum"] = nb([
md("# Notebook 6.4 – Momentum\nGradient descent with classical and Nesterov momentum."),
code(SETUP),
code("""
def rosenbrock(w): return (1-w[0])**2 + 100*(w[1]-w[0]**2)**2
def grad_rb(w):
    return np.array([-2*(1-w[0]) - 400*w[0]*(w[1]-w[0]**2),
                      200*(w[1]-w[0]**2)])

def gd_momentum(lr=0.001, beta=0.9, n=2000):
    w = np.array([-1.0, 1.0]); v = np.zeros(2); path=[w.copy()]
    for _ in range(n):
        v = beta*v - lr*grad_rb(w)
        w = w + v
        path.append(w.copy())
    return np.array(path)

def gd_nag(lr=0.001, beta=0.9, n=2000):
    w = np.array([-1.0, 1.0]); v = np.zeros(2); path=[w.copy()]
    for _ in range(n):
        w_look = w + beta*v
        v = beta*v - lr*grad_rb(w_look)
        w = w + v
        path.append(w.copy())
    return np.array(path)

path_sgd = gd_momentum(beta=0.0)
path_mom = gd_momentum(beta=0.9)
path_nag = gd_nag(beta=0.9)

xx = np.linspace(-2, 2, 200); yy = np.linspace(-1, 3, 200)
XX, YY = np.meshgrid(xx, yy)
ZZ = (1-XX)**2 + 100*(YY-XX**2)**2

plt.figure(figsize=(8,6))
plt.contour(XX, YY, np.log(ZZ+1), levels=30, cmap='Greys')
for path, label, c in [(path_sgd,'GD','r'),(path_mom,'Momentum','b'),(path_nag,'NAG','g')]:
    plt.plot(*path.T, c=c, lw=1, label=label)
plt.legend(); plt.title("Momentum on Rosenbrock")
plt.tight_layout(); plt.savefig("momentum.png", dpi=100); plt.show()
"""),
])

# ── 6.5 Adam ─────────────────────────────────────────────────────────────────
notebooks["Notebook_6_5_Adam"] = nb([
md("# Notebook 6.5 – Adam\nAdaptive Moment Estimation optimizer."),
code(SETUP),
code("""
def adam(grad_fn, x0, lr=0.01, b1=0.9, b2=0.999, eps=1e-8, n=2000):
    x = x0.copy(); m = np.zeros_like(x); v = np.zeros_like(x)
    path = [x.copy()]
    for t in range(1, n+1):
        g = grad_fn(x)
        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g**2
        m_hat = m / (1 - b1**t)
        v_hat = v / (1 - b2**t)
        x -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(x.copy())
    return np.array(path)

def grad_rb(w):
    return np.array([-2*(1-w[0]) - 400*w[0]*(w[1]-w[0]**2),
                      200*(w[1]-w[0]**2)])

x0 = np.array([-1.0, 1.0])
path = adam(grad_rb, x0, lr=0.01)

xx = np.linspace(-2, 2, 200); yy = np.linspace(-1, 3, 200)
XX, YY = np.meshgrid(xx, yy)
ZZ = (1-XX)**2 + 100*(YY-XX**2)**2

plt.figure(figsize=(8,6))
plt.contour(XX, YY, np.log(ZZ+1), levels=30, cmap='viridis')
plt.plot(*path.T, 'r.-', ms=2, lw=1)
plt.scatter(*path[-1], c='r', s=100, zorder=5, label=f'Final: {path[-1]}')
plt.legend(); plt.title("Adam on Rosenbrock")
plt.tight_layout(); plt.savefig("adam.png", dpi=100); plt.show()
print(f"Converged to: {path[-1]}")
"""),
])

# ── 7.1 Backpropagation in Toy Model ────────────────────────────────────────
notebooks["Notebook_7_1_Backpropagation_in_toy_model"] = nb([
md("# Notebook 7.1 – Backpropagation in a Toy Model\nManual forward & backward pass for a 2-layer network."),
code(SETUP),
code("""
# Toy network: f(x) = w2 * ReLU(w1*x + b1) + b2
# We compute gradients manually and verify with numerical gradients

def forward(x, w1, b1, w2, b2):
    z1 = w1 * x + b1
    h1 = max(0, z1)          # ReLU
    out = w2 * h1 + b2
    return out, z1, h1

def loss(pred, y): return 0.5 * (pred - y)**2

def backward(x, y, w1, b1, w2, b2):
    out, z1, h1 = forward(x, w1, b1, w2, b2)
    L = loss(out, y)
    # dL/d_out
    dL_dout = out - y
    # dL/d_w2, dL/d_b2
    dL_dw2 = dL_dout * h1
    dL_db2 = dL_dout
    # dL/d_h1
    dL_dh1 = dL_dout * w2
    # dL/d_z1 (ReLU derivative)
    dL_dz1 = dL_dh1 * (1 if z1 > 0 else 0)
    # dL/d_w1, dL/d_b1
    dL_dw1 = dL_dz1 * x
    dL_db1 = dL_dz1
    return L, dL_dw1, dL_db1, dL_dw2, dL_db2

x_in = 1.5; y_true = 2.0
w1, b1, w2, b2 = 0.8, -0.3, 1.2, 0.1

L, gw1, gb1, gw2, gb2 = backward(x_in, y_true, w1, b1, w2, b2)
print(f"Loss: {L:.4f}")
print(f"Analytic gradients — dw1={gw1:.4f}, db1={gb1:.4f}, dw2={gw2:.4f}, db2={gb2:.4f}")

# Numerical check
eps = 1e-5
def num_grad(param_name):
    params = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}
    params_plus = {**params}; params_plus[param_name] += eps
    params_minus= {**params}; params_minus[param_name] -= eps
    fwd_p,_,_ = forward(x_in, **params_plus)
    fwd_m,_,_ = forward(x_in, **params_minus)
    lp = loss(fwd_p, y_true); lm = loss(fwd_m, y_true)
    return (lp - lm) / (2*eps)

for p in ['w1','b1','w2','b2']:
    print(f"Numerical d{p} = {num_grad(p):.4f}")
"""),
])

# ── 7.2 Backpropagation ──────────────────────────────────────────────────────
notebooks["Notebook_7_2_Backpropagation"] = nb([
md("# Notebook 7.2 – Backpropagation\nGeneral backprop via computational graph."),
code(SETUP),
code("""
import torch, torch.nn as nn

torch.manual_seed(0)
N, D_in, H, D_out = 64, 10, 20, 3

X = torch.randn(N, D_in)
y = torch.randint(0, D_out, (N,))

model = nn.Sequential(
    nn.Linear(D_in, H), nn.ReLU(),
    nn.Linear(H, H),    nn.ReLU(),
    nn.Linear(H, D_out)
)
loss_fn = nn.CrossEntropyLoss()
opt     = torch.optim.SGD(model.parameters(), lr=0.01)

losses = []
for t in range(300):
    pred = model(X)
    loss = loss_fn(pred, y)
    opt.zero_grad()
    loss.backward()          # ← backprop
    opt.step()
    losses.append(loss.item())

plt.figure(figsize=(7,4))
plt.plot(losses); plt.xlabel('Iteration'); plt.ylabel('Cross-Entropy')
plt.title('Backprop Training'); plt.tight_layout()
plt.savefig("backprop.png", dpi=100); plt.show()
print(f"Final loss: {losses[-1]:.4f}")
"""),
md("## Gradient Norms per Layer"),
code("""
for name, p in model.named_parameters():
    if p.grad is not None:
        print(f"{name:25s} grad norm = {p.grad.norm().item():.5f}")
"""),
])

# ── 7.3 Initialization ───────────────────────────────────────────────────────
notebooks["Notebook_7_3_Initialization"] = nb([
md("# Notebook 7.3 – Initialization\nEffect of weight initialisation on gradient flow."),
code(SETUP),
code("""
import torch, torch.nn as nn

def build_net(depth, width, init='random', gain=1.0):
    layers = []
    for i in range(depth):
        in_f  = 1 if i == 0 else width
        out_f = 1 if i == depth-1 else width
        lin   = nn.Linear(in_f, out_f, bias=False)
        if init == 'zeros':
            nn.init.zeros_(lin.weight)
        elif init == 'large':
            nn.init.normal_(lin.weight, std=5.0)
        elif init == 'xavier':
            nn.init.xavier_normal_(lin.weight, gain=gain)
        elif init == 'he':
            nn.init.kaiming_normal_(lin.weight)
        else:  # random small
            nn.init.normal_(lin.weight, std=gain)
        layers += [lin, nn.ReLU()]
    return nn.Sequential(*layers)

torch.manual_seed(7)
depth, width = 20, 64
x = torch.randn(1, 1)

for init, gain in [('random', 0.01), ('random', 1.0), ('xavier', 1.0), ('he', 1.0)]:
    net = build_net(depth, width, init, gain)
    # Forward
    out = net(x)
    # Backward
    out.sum().backward()
    grad_norms = [p.grad.norm().item() for p in net.parameters() if p.grad is not None]
    print(f"Init={init:8s} gain={gain:.2f} | output={out.item():.3e} | "
          f"min_grad={min(grad_norms):.2e} max_grad={max(grad_norms):.2e}")
"""),
md("## Activation Statistics by Layer"),
code("""
import torch, torch.nn as nn

def activation_stats(init='he', depth=20, width=128):
    layers_list = []
    for i in range(depth):
        in_f  = 1 if i == 0 else width
        out_f = 1 if i == depth-1 else width
        lin   = nn.Linear(in_f, out_f, bias=True)
        if init == 'he':
            nn.init.kaiming_normal_(lin.weight)
        else:
            nn.init.normal_(lin.weight, std=1.0)
        layers_list.append(lin)
    
    x = torch.randn(1000, 1)
    stds = []
    for lin in layers_list:
        x = torch.relu(lin(x))
        stds.append(x.std().item())
    return stds

stds_rand = activation_stats('normal')
stds_he   = activation_stats('he')

plt.figure(figsize=(8,4))
plt.plot(stds_rand, 'r', label='Normal init (std=1)')
plt.plot(stds_he,   'b', label='He init')
plt.xlabel('Layer'); plt.ylabel('Activation Std')
plt.legend(); plt.title('Activation Std by Layer Depth')
plt.yscale('log'); plt.tight_layout()
plt.savefig("initialization.png", dpi=100); plt.show()
"""),
])

# ── 8.1 MNIST-1D Performance ─────────────────────────────────────────────────
notebooks["Notebook_8_1_MNIST-1D_performance"] = nb([
md("# Notebook 8.1 – MNIST-1D Performance\nTraining and evaluating networks on a 1-D analogue of MNIST."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# --- Synthetic MNIST-1D-like dataset ---
torch.manual_seed(0); np.random.seed(0)
N_train, N_test, D, K = 4000, 1000, 40, 10

def make_mnist1d(N, D=40, K=10, noise=0.1):
    templates = [np.random.randn(D) for _ in range(K)]
    X, y = [], []
    for _ in range(N):
        k  = np.random.randint(K)
        xi = templates[k] + noise * np.random.randn(D)
        X.append(xi); y.append(k)
    return np.array(X, np.float32), np.array(y, np.int64)

X_tr, y_tr = make_mnist1d(N_train)
X_te, y_te = make_mnist1d(N_test)

tr_loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                       batch_size=64, shuffle=True)

model = nn.Sequential(
    nn.Linear(D, 128), nn.ReLU(),
    nn.Linear(128, 64), nn.ReLU(),
    nn.Linear(64, K)
)
opt = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

tr_accs, te_accs = [], []
for epoch in range(30):
    model.train()
    for xb, yb in tr_loader:
        opt.zero_grad()
        loss_fn(model(xb), yb).backward()
        opt.step()
    # Eval
    model.eval()
    with torch.no_grad():
        tr_acc = (model(torch.tensor(X_tr)).argmax(1) == torch.tensor(y_tr)).float().mean().item()
        te_acc = (model(torch.tensor(X_te)).argmax(1) == torch.tensor(y_te)).float().mean().item()
    tr_accs.append(tr_acc); te_accs.append(te_acc)
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1:3d} | Train Acc: {tr_acc:.3f} | Test Acc: {te_acc:.3f}")

plt.figure(figsize=(7,4))
plt.plot(tr_accs, label='Train'); plt.plot(te_accs, label='Test')
plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
plt.title('MNIST-1D Performance'); plt.tight_layout()
plt.savefig("mnist1d.png", dpi=100); plt.show()
"""),
])

# ── 8.2 Bias-Variance Trade-off ──────────────────────────────────────────────
notebooks["Notebook_8_2_Bias-variance_trade-off"] = nb([
md("# Notebook 8.2 – Bias-Variance Trade-off\nUnderstanding overfitting and underfitting."),
code(SETUP),
code("""
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

np.random.seed(5)
f_true = lambda x: np.sin(2*np.pi*x)
N_train = 15

x_test = np.linspace(0, 1, 200)
y_test  = f_true(x_test)

bias2, variance, mse_list = [], [], []
degrees = [1, 2, 3, 5, 7, 10, 15]
n_trials = 50

for deg in degrees:
    preds_test = []
    for _ in range(n_trials):
        x_tr = np.random.uniform(0, 1, N_train)
        y_tr = f_true(x_tr) + 0.3*np.random.randn(N_train)
        m = make_pipeline(PolynomialFeatures(deg), Ridge(alpha=1e-9))
        m.fit(x_tr.reshape(-1,1), y_tr)
        preds_test.append(m.predict(x_test.reshape(-1,1)))
    preds_test = np.array(preds_test)
    mean_pred  = preds_test.mean(0)
    b2 = np.mean((mean_pred - y_test)**2)
    var= np.mean(preds_test.var(0))
    bias2.append(b2); variance.append(var); mse_list.append(b2 + var)

plt.figure(figsize=(8,5))
plt.plot(degrees, bias2,    'b', label='Bias²')
plt.plot(degrees, variance, 'r', label='Variance')
plt.plot(degrees, mse_list, 'k', label='MSE = Bias²+Var')
plt.xlabel('Polynomial Degree'); plt.legend()
plt.title('Bias-Variance Trade-off'); plt.tight_layout()
plt.savefig("bias_variance.png", dpi=100); plt.show()
"""),
])

# ── 8.3 Double Descent ───────────────────────────────────────────────────────
notebooks["Notebook_8_3_Double_descent"] = nb([
md("# Notebook 8.3 – Double Descent\nThe double descent risk curve in modern over-parameterised models."),
code(SETUP),
code("""
np.random.seed(0)
N = 50  # training points

def make_data(N, D_feat=100):
    X = np.random.randn(N, D_feat)
    w_true = np.random.randn(D_feat) * 0.1
    y = X @ w_true + 0.5 * np.random.randn(N)
    return X, y

X_all, y_all = make_data(1000, 200)
X_tr = X_all[:N]; y_tr = y_all[:N]
X_te = X_all[N:]; y_te = y_all[N:]

train_mse, test_mse = [], []
param_counts = list(range(2, 200, 4))

for D in param_counts:
    Xtr = X_tr[:, :D]; Xte = X_te[:, :D]
    # Minimum norm solution
    if D <= N:
        w = np.linalg.lstsq(Xtr, y_tr, rcond=None)[0]
    else:
        # Pseudo-inverse (interpolating solution)
        w = Xtr.T @ np.linalg.lstsq(Xtr @ Xtr.T, y_tr, rcond=None)[0]
    train_mse.append(np.mean((Xtr @ w - y_tr)**2))
    test_mse.append( np.mean((Xte @ w - y_te)**2))

plt.figure(figsize=(9,5))
plt.plot(param_counts, train_mse, 'b', label='Train MSE')
plt.plot(param_counts, test_mse,  'r', label='Test MSE')
plt.axvline(N, ls='--', c='k', label=f'N={N} (interpolation threshold)')
plt.yscale('log'); plt.ylim(1e-3, 1e3)
plt.xlabel('# Parameters'); plt.ylabel('MSE'); plt.legend()
plt.title('Double Descent'); plt.tight_layout()
plt.savefig("double_descent.png", dpi=100); plt.show()
"""),
])

# ── 8.4 High-Dimensional Spaces ──────────────────────────────────────────────
notebooks["Notebook_8_4_High-dimensional_spaces"] = nb([
md("# Notebook 8.4 – High-Dimensional Spaces\nConcentration of measure, curse of dimensionality."),
code(SETUP),
code("""
# Volume of unit ball
from math import gamma, pi

dims = np.arange(1, 51)
vol  = [pi**(d/2) / gamma(d/2 + 1) for d in dims]

plt.figure(figsize=(8,4))
plt.plot(dims, vol, 'b-o', ms=3)
plt.xlabel('Dimension'); plt.ylabel('Volume of Unit Ball')
plt.title('Volume of d-dimensional Unit Ball (radius=1)')
plt.tight_layout(); plt.savefig("unit_ball_volume.png", dpi=100); plt.show()
"""),
code("""
# Concentration of norm of random Gaussian vectors
n_samples = 10000
fig, axes = plt.subplots(1, 4, figsize=(14, 3))
for ax, d in zip(axes, [2, 10, 100, 1000]):
    norms = np.linalg.norm(np.random.randn(n_samples, d), axis=1)
    ax.hist(norms, bins=50, density=True, color='steelblue')
    ax.axvline(np.sqrt(d), c='r', label=f'√d={np.sqrt(d):.1f}')
    ax.set_title(f"d={d}"); ax.legend(fontsize=7)
plt.suptitle("Norm concentration of Gaussian vectors")
plt.tight_layout(); plt.savefig("norm_concentration.png", dpi=100); plt.show()
"""),
code("""
# Nearest-neighbour distances in high dimensions
def nn_distance_ratio(d, n=200):
    pts = np.random.randn(n, d)
    dists = []
    for i in range(n):
        diff = pts - pts[i]
        d_to_others = np.sqrt((diff**2).sum(1))
        d_to_others[i] = np.inf
        dists.append(d_to_others.min())
    dists = np.array(dists)
    return dists.max() / dists.min()

dims = [1, 2, 5, 10, 20, 50, 100]
ratios = [nn_distance_ratio(d) for d in dims]
plt.figure(figsize=(7,4))
plt.plot(dims, ratios, 'ro-')
plt.xlabel('Dimension'); plt.ylabel('max_nn_dist / min_nn_dist')
plt.title('Curse of Dimensionality: NN Distance Ratio')
plt.tight_layout(); plt.savefig("curse_dim.png", dpi=100); plt.show()
"""),
])

# ── 9.1 L2 Regularization ────────────────────────────────────────────────────
notebooks["Notebook_9_1_L2_regularization"] = nb([
md("# Notebook 9.1 – L2 Regularization\nWeight decay, ridge regression, and the effect on learned weights."),
code(SETUP),
code("""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

np.random.seed(3)
f_true = lambda x: np.sin(2*np.pi*x)
N = 20
x_tr = np.random.uniform(0,1,N); y_tr = f_true(x_tr) + 0.2*np.random.randn(N)
x_te = np.linspace(0,1,200);    y_te = f_true(x_te)

deg = 12
PF  = PolynomialFeatures(deg)
Xtr = PF.fit_transform(x_tr.reshape(-1,1))
Xte = PF.transform(x_te.reshape(-1,1))

fig, axes = plt.subplots(2, 3, figsize=(13,8))
lambdas = [0, 1e-6, 1e-4, 1e-2, 0.1, 1.0]
for ax, lam in zip(axes.flat, lambdas):
    m = Ridge(alpha=lam, fit_intercept=False).fit(Xtr, y_tr)
    ax.scatter(x_tr, y_tr, s=20, c='k', alpha=0.7)
    ax.plot(x_te, y_te, 'g', lw=2, label='True')
    ax.plot(x_te, m.predict(Xte), 'r', lw=2, label='Fit')
    ax.set_title(f'λ={lam}'); ax.set_ylim(-2.5, 2.5); ax.legend(fontsize=7)
plt.suptitle('L2 Regularization: Polynomial Fit'); plt.tight_layout()
plt.savefig("l2_reg.png", dpi=100); plt.show()
"""),
])

# ── 9.2 Implicit Regularization ──────────────────────────────────────────────
notebooks["Notebook_9_2_Implicit_regularization"] = nb([
md("# Notebook 9.2 – Implicit Regularization\nHow SGD implicitly regularizes by converging to minimum-norm solutions."),
code(SETUP),
code("""
import torch, torch.nn as nn

torch.manual_seed(0)

# Under-determined linear system: more params than data
N, D = 10, 100
X = torch.randn(N, D); y = torch.randn(N, 1)

def train_linear(X, y, lr=0.01, steps=5000):
    W = nn.Parameter(torch.zeros(D, 1))
    opt = torch.optim.SGD([W], lr=lr)
    losses = []
    for _ in range(steps):
        loss = ((X @ W - y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return W.data, losses

W_sgd, losses = train_linear(X, y)

# Minimum-norm (pseudo-inverse) solution
W_pinv = X.T @ torch.linalg.lstsq(X @ X.T, y).solution

print(f"SGD  solution norm: {W_sgd.norm():.4f}")
print(f"Pinv solution norm: {W_pinv.norm():.4f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.plot(losses); ax1.set_title("SGD Loss"); ax1.set_yscale('log')
ax2.scatter(W_sgd.numpy().squeeze(), W_pinv.numpy().squeeze(), s=5, alpha=0.5)
ax2.set_xlabel("SGD weights"); ax2.set_ylabel("Pinv weights")
ax2.plot([-0.2,0.2],[-0.2,0.2],'r--')
ax2.set_title("SGD vs Pseudo-Inverse Weights")
plt.tight_layout(); plt.savefig("implicit_reg.png", dpi=100); plt.show()
"""),
])

# ── 9.3 Ensembling ───────────────────────────────────────────────────────────
notebooks["Notebook_9_3_Ensembling"] = nb([
md("# Notebook 9.3 – Ensembling\nBagging, averaging predictions, and variance reduction."),
code(SETUP),
code("""
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

def train_ensemble(X, y, n_models=20, max_depth=3):
    models = []
    for i in range(n_models):
        idx = np.random.choice(len(X), len(X), replace=True)
        m = DecisionTreeClassifier(max_depth=max_depth, random_state=i)
        m.fit(X[idx], y[idx])
        models.append(m)
    return models

models = train_ensemble(X, y)

xx, yy = np.meshgrid(np.linspace(-3,3,200), np.linspace(-2,2,200))
grid = np.c_[xx.ravel(), yy.ravel()]

# Single model
Z_single = models[0].predict_proba(grid)[:,1].reshape(200,200)
# Ensemble
Z_ens = np.mean([m.predict_proba(grid)[:,1] for m in models], axis=0).reshape(200,200)

fig, axes = plt.subplots(1, 2, figsize=(11,5))
for ax, Z, title in zip(axes, [Z_single, Z_ens], ['Single Tree', f'Ensemble ({len(models)} trees)']):
    ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.7)
    ax.scatter(*X.T, c=y, cmap='RdBu', edgecolors='k', s=25)
    ax.set_title(title)
plt.suptitle("Ensembling: Variance Reduction")
plt.tight_layout(); plt.savefig("ensembling.png", dpi=100); plt.show()
"""),
])

# ── 9.4 Bayesian Approach ────────────────────────────────────────────────────
notebooks["Notebook_9_4_Bayesian_approach"] = nb([
md("# Notebook 9.4 – Bayesian Approach\nPosterior predictive distribution and uncertainty quantification."),
code(SETUP),
code("""
# Bayesian linear regression (conjugate Gaussian)
np.random.seed(1)
f_true = lambda x: np.sin(2*np.pi*x)
N = 10
x_tr = np.random.uniform(0,1,N); y_tr = f_true(x_tr) + 0.2*np.random.randn(N)
x_star = np.linspace(0,1,200)

# Feature map: polynomial of degree 5
def phi(x, deg=5):
    return np.stack([x**d for d in range(deg+1)], axis=1)

Phi_tr = phi(x_tr); Phi_star = phi(x_star)
sigma2 = 0.04  # noise variance
alpha  = 1.0   # prior precision

# Posterior
A   = alpha * np.eye(6) + Phi_tr.T @ Phi_tr / sigma2
A_inv = np.linalg.inv(A)
mu_post = A_inv @ Phi_tr.T @ y_tr / sigma2

# Predictive mean and std
mu_pred  = Phi_star @ mu_post
var_pred  = sigma2 + np.einsum('ij,jk,ik->i', Phi_star, A_inv, Phi_star)
std_pred  = np.sqrt(var_pred)

plt.figure(figsize=(8,5))
plt.fill_between(x_star, mu_pred-2*std_pred, mu_pred+2*std_pred,
                 alpha=0.3, color='b', label='±2σ')
plt.plot(x_star, mu_pred, 'b', lw=2, label='Posterior mean')
plt.plot(x_star, f_true(x_star), 'k--', label='True')
plt.scatter(x_tr, y_tr, c='k', s=50, zorder=5, label='Data')
plt.legend(); plt.title("Bayesian Polynomial Regression")
plt.tight_layout(); plt.savefig("bayesian.png", dpi=100); plt.show()
"""),
])

# ── 9.5 Augmentation ─────────────────────────────────────────────────────────
notebooks["Notebook_9_5_Augmentation"] = nb([
md("# Notebook 9.5 – Augmentation\nData augmentation strategies for improving generalisation."),
code(SETUP),
code("""
import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.patches as patches

# Create a synthetic 28x28 image (digit-like blob)
def make_sample_img():
    img = np.zeros((28,28), dtype=np.float32)
    for cy, cx in [(10,8),(14,14),(18,20)]:
        for i in range(28):
            for j in range(28):
                img[i,j] += np.exp(-((i-cy)**2+(j-cx)**2)/8)
    img = (img / img.max() * 255).astype(np.uint8)
    return Image.fromarray(img)

img = make_sample_img()

augmentations = {
    'Original':         T.ToTensor(),
    'Horizontal Flip':  T.Compose([T.RandomHorizontalFlip(p=1), T.ToTensor()]),
    'Rotation ±30°':    T.Compose([T.RandomRotation(30), T.ToTensor()]),
    'Random Crop':      T.Compose([T.RandomCrop(22, padding=4), T.Resize(28), T.ToTensor()]),
    'Gaussian Blur':    T.Compose([T.GaussianBlur(3, sigma=(0.1,2.0)), T.ToTensor()]),
    'Color Jitter':     T.Compose([T.Grayscale(3), T.ColorJitter(0.5,0.5,0.5,0.1),
                                   T.Grayscale(1), T.ToTensor()]),
}

fig, axes = plt.subplots(2, 3, figsize=(10,7))
for ax, (name, tfm) in zip(axes.flat, augmentations.items()):
    t = tfm(img).squeeze().numpy()
    ax.imshow(t, cmap='gray'); ax.set_title(name); ax.axis('off')
plt.suptitle("Data Augmentation Examples"); plt.tight_layout()
plt.savefig("augmentation.png", dpi=100); plt.show()
"""),
])

# ── 10.1 1-D Convolution ─────────────────────────────────────────────────────
notebooks["Notebook_10_1_1D_convolution"] = nb([
md("# Notebook 10.1 – 1-D Convolution\nConvolution as parameter sharing; filters and feature maps."),
code(SETUP),
code("""
from scipy.signal import convolve

# Signal
t = np.linspace(0, 4*np.pi, 400)
signal = np.sin(t) + 0.5*np.sin(3*t) + 0.2*np.random.randn(400)

# Filters
filters = {
    'Smoothing (box)':    np.ones(21)/21,
    'Derivative (diff)':  np.array([1, 0, -1])/2,
    'Edge (Laplacian)':   np.array([-1, 2, -1]),
    'Gaussian':           np.exp(-np.linspace(-3,3,21)**2 / 2),
}
filters['Gaussian'] /= filters['Gaussian'].sum()

fig, axes = plt.subplots(len(filters)+1, 1, figsize=(10, 12))
axes[0].plot(t, signal, 'k'); axes[0].set_title("Original Signal")
for ax, (name, filt) in zip(axes[1:], filters.items()):
    out = convolve(signal, filt, mode='same')
    ax.plot(t, out, 'b', lw=1.5); ax.set_title(f"After '{name}' filter")
plt.tight_layout(); plt.savefig("conv1d.png", dpi=100); plt.show()
"""),
])

# ── 10.2 Convolution for MNIST-1D ────────────────────────────────────────────
notebooks["Notebook_10_2_Convolution_for_MNIST-1D"] = nb([
md("# Notebook 10.2 – Convolution for MNIST-1D\nApplying 1-D convolution to classify MNIST-1D signals."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0); np.random.seed(0)
D, K, N = 40, 10, 5000

templates = [np.random.randn(D) * 2 for _ in range(K)]

def make_data(N):
    X, y = [], []
    for _ in range(N):
        k  = np.random.randint(K)
        xi = np.roll(templates[k], np.random.randint(5)) + 0.5*np.random.randn(D)
        X.append(xi); y.append(k)
    return np.array(X, np.float32), np.array(y, np.int64)

X_tr, y_tr = make_data(4000); X_te, y_te = make_data(1000)

class ConvNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),
            nn.Linear(32, K)
        )
    def forward(self, x): return self.net(x.unsqueeze(1))

model = ConvNet1D()
opt   = optim.Adam(model.parameters(), 1e-3)
loader = DataLoader(TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr)),
                    batch_size=64, shuffle=True)

for ep in range(20):
    for xb, yb in loader:
        nn.CrossEntropyLoss()(model(xb), yb).backward()
        opt.step(); opt.zero_grad()
    if (ep+1) % 5 == 0:
        acc = (model(torch.tensor(X_te)).argmax(1) == torch.tensor(y_te)).float().mean()
        print(f"Epoch {ep+1:2d} | Test acc: {acc:.3f}")
"""),
])

# ── 10.3 2-D Convolution ─────────────────────────────────────────────────────
notebooks["Notebook_10_3_2D_convolution"] = nb([
md("# Notebook 10.3 – 2-D Convolution\nImage filters, edge detection, and learnable kernels."),
code(SETUP),
code("""
from scipy.ndimage import convolve as nd_conv

# Synthetic image
img = np.zeros((64,64))
img[20:44, 20:44] = 1.0
img[30:34, 10:54] = 0.5
img += 0.1 * np.random.randn(64,64)

kernels = {
    'Identity':   np.array([[0,0,0],[0,1,0],[0,0,0]]),
    'Blur':       np.ones((5,5))/25,
    'Sharpen':    np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]),
    'Sobel-X':    np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),
    'Sobel-Y':    np.array([[-1,-2,-1],[0,0,0],[1,2,1]]),
    'Laplacian':  np.array([[0,1,0],[1,-4,1],[0,1,0]]),
}

fig, axes = plt.subplots(2, 3, figsize=(12,8))
for ax, (name, k) in zip(axes.flat, kernels.items()):
    out = nd_conv(img, k)
    ax.imshow(out, cmap='gray'); ax.set_title(name); ax.axis('off')
plt.suptitle("2-D Convolution with Different Kernels")
plt.tight_layout(); plt.savefig("conv2d.png", dpi=100); plt.show()
"""),
])

# ── 10.4 Downsampling & Upsampling ──────────────────────────────────────────
notebooks["Notebook_10_4_Downsampling_&_upsampling"] = nb([
md("# Notebook 10.4 – Downsampling & Upsampling\nPooling, strided convolution, transposed convolution."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.nn.functional as F

# Synthetic 1×64×64 image
torch.manual_seed(0)
img = torch.zeros(1,1,64,64)
img[0,0,20:44,20:44] = 1.0

print("Original:", img.shape)

# Downsampling
avg_pool  = F.avg_pool2d(img, kernel_size=2, stride=2)
max_pool  = F.max_pool2d(img, kernel_size=2, stride=2)
strided   = nn.Conv2d(1,1,3,stride=2,padding=1)(img)
print("After 2× downsample (avg pool):", avg_pool.shape)

# Upsampling
nearest   = F.interpolate(avg_pool, scale_factor=2, mode='nearest')
bilinear  = F.interpolate(avg_pool, scale_factor=2, mode='bilinear', align_corners=False)
trans_conv= nn.ConvTranspose2d(1,1,2,stride=2)(avg_pool)
print("After 2× upsample (bilinear):", bilinear.shape)

fig, axes = plt.subplots(2, 3, figsize=(12,8))
imgs = [(img.squeeze(),'Original 64×64'),
        (avg_pool.squeeze(),'Avg Pool 32×32'),
        (max_pool.squeeze(),'Max Pool 32×32'),
        (nearest.squeeze(),'Nearest Up 64×64'),
        (bilinear.squeeze(),'Bilinear Up 64×64'),
        (trans_conv.detach().squeeze(),'TransConv 64×64')]
for ax, (im, title) in zip(axes.flat, imgs):
    ax.imshow(im.detach().numpy(), cmap='gray')
    ax.set_title(title); ax.axis('off')
plt.suptitle("Downsampling & Upsampling"); plt.tight_layout()
plt.savefig("downsample_upsample.png", dpi=100); plt.show()
"""),
])

# ── 10.5 Convolution for MNIST ───────────────────────────────────────────────
notebooks["Notebook_10_5_Convolution_for_MNIST"] = nb([
md("# Notebook 10.5 – Convolution for MNIST\nLeNet-style ConvNet trained on MNIST (or Fashion-MNIST)."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))])

try:
    train_ds = datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test_ds  = datasets.MNIST('./data', train=False, download=True, transform=transform)
    dname = "MNIST"
except Exception:
    # Fallback synthetic
    print("Using synthetic data (MNIST unavailable)")
    N, C, H, W = 1000, 1, 28, 28
    X_s = torch.randn(N, C, H, W)
    y_s = torch.randint(0, 10, (N,))
    train_ds = torch.utils.data.TensorDataset(X_s[:800], y_s[:800])
    test_ds  = torch.utils.data.TensorDataset(X_s[800:], y_s[800:])
    dname = "Synthetic"

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=256)

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1,6,5,padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6,16,5),          nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*5*5, 120), nn.ReLU(),
            nn.Linear(120, 84),     nn.ReLU(),
            nn.Linear(84, 10))
    def forward(self, x): return self.classifier(self.features(x))

device = 'cpu'
model  = LeNet().to(device)
opt    = optim.Adam(model.parameters(), 1e-3)
loss_fn= nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(); loss_fn(model(xb), yb).backward(); opt.step()
    model.eval()
    correct = sum((model(xb.to(device)).argmax(1) == yb.to(device)).sum().item()
                  for xb, yb in test_loader)
    print(f"Epoch {epoch+1} | Test Acc: {correct/len(test_ds):.4f}")
"""),
])

# ── 11.1 Shattered Gradients ─────────────────────────────────────────────────
notebooks["Notebook_11_1_Shattered_gradients"] = nb([
md("# Notebook 11.1 – Shattered Gradients\nGradient correlation degrades exponentially with depth (without BatchNorm/ResNets)."),
code(SETUP),
code("""
import torch, torch.nn as nn

torch.manual_seed(0)

def gradient_correlation(depth, width=64, n=200):
    layers = []
    for _ in range(depth):
        layers += [nn.Linear(width, width), nn.Tanh()]
    net = nn.Sequential(nn.Linear(1, width), nn.Tanh(), *layers, nn.Linear(width, 1))
    
    corrs = []
    for _ in range(20):
        x1 = torch.randn(1); x2 = x1 + 0.01*torch.randn(1)
        g1, g2 = [], []
        for xi, gi in [(x1, g1), (x2, g2)]:
            net.zero_grad()
            out = net(xi.unsqueeze(0))
            out.backward()
            params = [p.grad.view(-1) for p in net.parameters() if p.grad is not None]
            gi.append(torch.cat(params))
        g1, g2 = g1[0], g2[0]
        corr = torch.dot(g1, g2) / (g1.norm() * g2.norm() + 1e-10)
        corrs.append(corr.item())
    return np.mean(corrs)

depths = [1, 2, 4, 8, 12, 16, 20]
corrs  = [gradient_correlation(d) for d in depths]

plt.figure(figsize=(8,4))
plt.plot(depths, corrs, 'bo-')
plt.axhline(0, ls='--', c='k')
plt.xlabel('Depth'); plt.ylabel('Gradient Correlation')
plt.title('Shattered Gradients: Correlation vs Depth (Tanh, no BatchNorm)')
plt.tight_layout(); plt.savefig("shattered_grads.png", dpi=100); plt.show()
for d, c in zip(depths, corrs): print(f"Depth {d:2d}: corr = {c:.4f}")
"""),
])

# ── 11.2 Residual Networks ───────────────────────────────────────────────────
notebooks["Notebook_11_2_Residual_networks"] = nb([
md("# Notebook 11.2 – Residual Networks\nSkip connections and how they help gradient flow."),
code(SETUP),
code("""
import torch, torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(),
            nn.Linear(dim, dim))
    def forward(self, x): return torch.relu(x + self.block(x))

class PlainNet(nn.Module):
    def __init__(self, depth, dim=64):
        super().__init__()
        layers = [nn.Linear(1, dim), nn.ReLU()]
        for _ in range(depth): layers += [nn.Linear(dim,dim), nn.ReLU()]
        layers.append(nn.Linear(dim, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class ResNet(nn.Module):
    def __init__(self, depth, dim=64):
        super().__init__()
        self.inp  = nn.Linear(1, dim)
        self.res  = nn.Sequential(*[ResBlock(dim) for _ in range(depth)])
        self.out  = nn.Linear(dim, 1)
    def forward(self, x): return self.out(self.res(torch.relu(self.inp(x))))

torch.manual_seed(1)
X = torch.randn(200, 1); y = torch.sin(X)

def train_and_eval(model, epochs=300):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    losses = []
    for _ in range(epochs):
        loss = ((model(X) - y)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
for depth, c in [(5,'b'),(10,'g'),(20,'r')]:
    l_plain = train_and_eval(PlainNet(depth))
    l_res   = train_and_eval(ResNet(depth))
    ax1.plot(l_plain, c=c, ls='-',  label=f'Plain d={depth}')
    ax2.plot(l_res,   c=c, ls='--', label=f'ResNet d={depth}')
ax1.set_title("Plain Net Loss"); ax1.legend(); ax1.set_yscale('log')
ax2.set_title("ResNet Loss");    ax2.legend(); ax2.set_yscale('log')
plt.suptitle("Residual vs Plain Networks"); plt.tight_layout()
plt.savefig("resnets.png", dpi=100); plt.show()
"""),
])

# ── 11.3 Batch Normalization ─────────────────────────────────────────────────
notebooks["Notebook_11_3_Batch_normalization"] = nb([
md("# Notebook 11.3 – Batch Normalization\nBatchNorm mechanics, effect on training stability."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)
N, D_in, D_out = 1000, 20, 5

X = torch.randn(N, D_in) * 10   # large-scale input
y = torch.randint(0, D_out, (N,))

def build_model(use_bn, depth=6, width=64):
    layers = [nn.Linear(D_in, width), nn.ReLU()]
    if use_bn: layers.insert(1, nn.BatchNorm1d(width))
    for _ in range(depth-1):
        layers += [nn.Linear(width, width), nn.ReLU()]
        if use_bn: layers.append(nn.BatchNorm1d(width))
    layers.append(nn.Linear(width, D_out))
    return nn.Sequential(*layers)

loss_fn = nn.CrossEntropyLoss()
idx = torch.randperm(N)
Xtr, ytr = X[idx[:800]], y[idx[:800]]
Xte, yte = X[idx[800:]], y[idx[800:]]

fig, axes = plt.subplots(1, 2, figsize=(12,4))
for use_bn, label, c in [(False,'Without BN','r'),(True,'With BN','b')]:
    model = build_model(use_bn)
    opt   = optim.Adam(model.parameters(), 1e-3)
    losses = []
    for ep in range(100):
        model.train()
        loss = loss_fn(model(Xtr), ytr)
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    axes[0].plot(losses, c=c, label=label)
    model.eval()
    with torch.no_grad():
        acc = (model(Xte).argmax(1) == yte).float().mean()
    print(f"{label}: Final acc = {acc:.3f}")

axes[0].set_title("Training Loss"); axes[0].legend()

# Visualise activation distributions with/without BN
for use_bn, ax, label in [(False, axes[1], 'No BN'), (True, axes[1], 'BN')]:
    model = build_model(use_bn)
    with torch.no_grad():
        # Get activations at layer 3
        out = X[:100]
        for i, layer in enumerate(model):
            out = layer(out)
            if i == 5: break
    ax.hist(out.numpy().ravel(), bins=50, alpha=0.5, label=label)
axes[1].legend(); axes[1].set_title("Layer Activation Distribution")
plt.suptitle("Batch Normalization"); plt.tight_layout()
plt.savefig("batchnorm.png", dpi=100); plt.show()
"""),
])

# ── 12.1 Self-Attention ──────────────────────────────────────────────────────
notebooks["Notebook_12_1_Self-attention"] = nb([
md("# Notebook 12.1 – Self-Attention\nDot-product self-attention mechanism from scratch."),
code(SETUP),
code("""
import torch, torch.nn.functional as F

def self_attention(X, W_q, W_k, W_v):
    \"\"\"X: (T, D), Ws: (D, D_k or D_v)\"\"\"
    Q = X @ W_q; K = X @ W_k; V = X @ W_v
    d_k = Q.shape[-1]
    scores = Q @ K.T / d_k**0.5          # (T, T)
    attn   = F.softmax(scores, dim=-1)   # (T, T)
    out    = attn @ V                    # (T, D_v)
    return out, attn

torch.manual_seed(0)
T, D, D_k = 6, 8, 4
X   = torch.randn(T, D)
W_q = torch.randn(D, D_k); W_k = torch.randn(D, D_k); W_v = torch.randn(D, D_k)

out, attn = self_attention(X, W_q, W_k, W_v)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
ax1.imshow(attn.detach().numpy(), cmap='Blues')
ax1.set_title("Attention Weights"); ax1.set_xlabel("Key position"); ax1.set_ylabel("Query position")
plt.colorbar(ax1.images[0], ax=ax1)

ax2.imshow(out.detach().numpy(), cmap='viridis', aspect='auto')
ax2.set_title("Output (T × D_v)"); ax2.set_xlabel("D_v"); ax2.set_ylabel("Token")
plt.tight_layout(); plt.savefig("self_attention.png", dpi=100); plt.show()
"""),
])

# ── 12.2 Multi-Head Self-Attention ───────────────────────────────────────────
notebooks["Notebook_12_2_Multi-head_self-attention"] = nb([
md("# Notebook 12.2 – Multi-Head Self-Attention\nRunning H parallel attention heads and concatenating."),
code(SETUP),
code("""
import torch, torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, D, H):
        super().__init__()
        assert D % H == 0
        self.H = H; self.d_k = D // H
        self.W_qkv = nn.Linear(D, 3*D, bias=False)
        self.W_o   = nn.Linear(D, D, bias=False)
    
    def forward(self, x):
        B, T, D = x.shape
        qkv = self.W_qkv(x).reshape(B, T, 3, self.H, self.d_k).permute(2,0,3,1,4)
        Q, K, V = qkv[0], qkv[1], qkv[2]   # (B, H, T, d_k)
        attn = torch.softmax(Q @ K.transpose(-2,-1) / self.d_k**0.5, dim=-1)
        out  = (attn @ V).transpose(1,2).reshape(B, T, D)
        return self.W_o(out), attn

torch.manual_seed(42)
B, T, D, H = 2, 10, 16, 4
x   = torch.randn(B, T, D)
mha = MultiHeadAttention(D, H)
out, attn = mha(x)
print(f"Input:  {x.shape}")
print(f"Output: {out.shape}")
print(f"Attn:   {attn.shape}  (B, H, T, T)")

# Visualise all heads for first batch item
fig, axes = plt.subplots(1, H, figsize=(14, 3))
for h, ax in enumerate(axes):
    ax.imshow(attn[0, h].detach().numpy(), cmap='Blues')
    ax.set_title(f"Head {h+1}"); ax.set_xlabel("Key"); ax.set_ylabel("Query")
plt.suptitle("Multi-Head Attention Weights"); plt.tight_layout()
plt.savefig("multihead_attn.png", dpi=100); plt.show()
"""),
])

# ── 12.3 Tokenization ────────────────────────────────────────────────────────
notebooks["Notebook_12_3_Tokenization"] = nb([
md("# Notebook 12.3 – Tokenization\nByte-pair encoding (BPE) and positional embeddings."),
code(SETUP),
code("""
# Simple character-level tokeniser
text = "the quick brown fox jumps over the lazy dog"
chars = sorted(set(text))
c2i   = {c:i for i,c in enumerate(chars)}
i2c   = {i:c for c,i in c2i.items()}
tokens = [c2i[c] for c in text]
print("Vocabulary:", chars)
print("Tokens:",     tokens[:20], "...")

# Sinusoidal positional encoding
def positional_encoding(max_len, d_model):
    PE = np.zeros((max_len, d_model))
    pos = np.arange(max_len).reshape(-1,1)
    div = np.exp(np.arange(0, d_model, 2) * -(np.log(10000) / d_model))
    PE[:, 0::2] = np.sin(pos * div)
    PE[:, 1::2] = np.cos(pos * div)
    return PE

PE = positional_encoding(50, 64)
plt.figure(figsize=(10,4))
plt.imshow(PE.T, aspect='auto', cmap='RdBu')
plt.colorbar(); plt.xlabel("Position"); plt.ylabel("Dimension")
plt.title("Sinusoidal Positional Encoding"); plt.tight_layout()
plt.savefig("positional_encoding.png", dpi=100); plt.show()
"""),
md("## Simple BPE Merge Step"),
code("""
from collections import Counter

def get_pairs(vocab):
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[(symbols[i], symbols[i+1])] += freq
    return pairs

def merge_pair(pair, vocab):
    new_vocab = {}
    bigram = ' '.join(pair)
    replacement = ''.join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab

words = text.split()
vocab = Counter(' '.join(w) for w in words)
print("Initial vocab:", dict(list(vocab.items())[:5]))

for _ in range(5):
    pairs = get_pairs(vocab)
    if not pairs: break
    best = max(pairs, key=pairs.get)
    vocab = merge_pair(best, vocab)
    print(f"  Merged: {best} → {''.join(best)}")
"""),
])

# ── 12.4 Decoding Strategies ─────────────────────────────────────────────────
notebooks["Notebook_12_4_Decoding_strategies"] = nb([
md("# Notebook 12.4 – Decoding Strategies\nGreedy, beam search, top-k, top-p (nucleus) sampling."),
code(SETUP),
code("""
import torch, torch.nn.functional as F

# Simulated logit distributions
torch.manual_seed(0)

def greedy_decode(logits): return logits.argmax().item()

def top_k_sample(logits, k=5):
    top_vals, top_idx = torch.topk(logits, k)
    probs = F.softmax(top_vals, dim=-1)
    return top_idx[torch.multinomial(probs, 1)].item()

def nucleus_sample(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = probs.sort(descending=True)
    cum_probs = sorted_probs.cumsum(0)
    # Remove tokens once cumulative prob exceeds p
    mask = cum_probs - sorted_probs > p
    sorted_probs[mask] = 0
    sorted_probs /= sorted_probs.sum()
    return sorted_idx[torch.multinomial(sorted_probs, 1)].item()

def beam_search(prob_fn, start, vocab_size, beam_width=3, max_len=10):
    beams = [(0.0, [start])]
    for _ in range(max_len):
        candidates = []
        for score, seq in beams:
            logits = prob_fn(seq)
            probs  = F.softmax(logits, dim=-1)
            top_p, top_i = probs.topk(beam_width)
            for p, i in zip(top_p, top_i):
                candidates.append((score - p.log().item(), seq + [i.item()]))
        beams = sorted(candidates)[:beam_width]
    return beams[0][1]

# Demo
logits = torch.randn(20)
print("Greedy:", greedy_decode(logits))
print("Top-k (k=5):", top_k_sample(logits, k=5))
print("Nucleus (p=0.9):", nucleus_sample(logits, p=0.9))

# Visualise probability distributions
probs = F.softmax(logits, dim=-1).numpy()
sorted_idx = np.argsort(probs)[::-1]

fig, axes = plt.subplots(1, 3, figsize=(13,4))
# Top-k mask
probs_topk = probs.copy(); probs_topk[np.argsort(probs_topk)[:-5]] = 0
probs_topk /= probs_topk.sum()
# Nucleus mask
sorted_p = np.sort(probs)[::-1]; cum = np.cumsum(sorted_p)
threshold_idx = np.where(cum >= 0.9)[0][0] + 1
mask_nuc = np.argsort(probs)[::-1][threshold_idx:]
probs_nuc = probs.copy(); probs_nuc[mask_nuc] = 0
probs_nuc /= probs_nuc.sum()

for ax, (p, title) in zip(axes, [(probs,'Original'),(probs_topk,'Top-k (k=5)'),(probs_nuc,'Nucleus (p=0.9)')]):
    ax.bar(range(20), p[sorted_idx])
    ax.set_title(title); ax.set_xlabel("Token (ranked)"); ax.set_ylabel("Prob")
plt.suptitle("Decoding Strategies"); plt.tight_layout()
plt.savefig("decoding.png", dpi=100); plt.show()
"""),
])

# ── 13.1 Encoding Graphs ─────────────────────────────────────────────────────
notebooks["Notebook_13_1_Encoding_graphs"] = nb([
md("# Notebook 13.1 – Encoding Graphs\nAdjacency matrices, node features, and graph representations."),
code(SETUP),
code("""
import networkx as nx

# Karate Club graph
G = nx.karate_club_graph()
A = nx.to_numpy_array(G)       # Adjacency matrix
D = np.diag(A.sum(1))          # Degree matrix
L = D - A                      # Laplacian

print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
print(f"Adjacency matrix shape: {A.shape}")

fig, axes = plt.subplots(1, 3, figsize=(14,4))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, ax=axes[0], node_size=80, node_color=list(nx.get_node_attributes(G,'club').values()),
        cmap=plt.cm.Set1, with_labels=False)
axes[0].set_title("Karate Club Graph")

im1 = axes[1].imshow(A, cmap='Blues'); axes[1].set_title("Adjacency Matrix")
plt.colorbar(im1, ax=axes[1], fraction=0.04)

# Normalised Laplacian eigenspectrum
eigvals = np.linalg.eigvalsh(L)
axes[2].plot(eigvals, 'b.'); axes[2].set_title("Laplacian Eigenspectrum")
axes[2].set_xlabel("Index"); axes[2].set_ylabel("Eigenvalue")

plt.suptitle("Graph Encoding"); plt.tight_layout()
plt.savefig("graph_encoding.png", dpi=100); plt.show()
"""),
])

# ── 13.2 Graph Classification ────────────────────────────────────────────────
notebooks["Notebook_13_2_Graph_classification"] = nb([
md("# Notebook 13.2 – Graph Classification\nGraph-level readout and simple GNN classification."),
code(SETUP),
code("""
import torch, torch.nn as nn

def simple_gcn_layer(X, A_hat):
    \"\"\"X: (N,F), A_hat: normalized adjacency\"\"\"
    return torch.relu(A_hat @ X)

def readout(H): return H.mean(0)   # Global average pooling

# Synthetic graphs: 2 classes
def make_graph(n=10, p=0.4, label=0, seed=None):
    if seed: np.random.seed(seed)
    A = (np.random.rand(n,n) < p).astype(float)
    A = (A + A.T) > 0; np.fill_diagonal(A, 0)
    A = A.astype(float)
    X = np.random.randn(n, 8) + label * 2
    # Normalized A_hat
    D_inv_sqrt = np.diag(1/(A.sum(1)+1e-6)**0.5)
    A_hat = D_inv_sqrt @ (A + np.eye(n)) @ D_inv_sqrt
    return torch.tensor(X, dtype=torch.float32), torch.tensor(A_hat, dtype=torch.float32)

graphs = [(make_graph(label=0, seed=i), 0) for i in range(20)] + \
         [(make_graph(label=1, seed=i+100), 1) for i in range(20)]

# 2-layer GCN + classifier
class GCNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(8, 16)
        self.lin2 = nn.Linear(16, 2)
    def forward(self, X, A_hat):
        H = torch.relu(A_hat @ self.lin1(X))
        g = H.mean(0)
        return self.lin2(g)

model = GCNClassifier()
opt   = torch.optim.Adam(model.parameters(), 1e-3)

for ep in range(50):
    total_loss = 0
    for (X,A), y in graphs:
        logit = model(X, A)
        loss  = nn.CrossEntropyLoss()(logit.unsqueeze(0), torch.tensor([y]))
        opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item()
    if (ep+1) % 10 == 0:
        correct = sum(model(X,A).argmax().item() == y for (X,A),y in graphs)
        print(f"Epoch {ep+1:3d} | Acc: {correct}/{len(graphs)}")
"""),
])

# ── 13.3 Neighborhood Sampling ───────────────────────────────────────────────
notebooks["Notebook_13_3_Neighborhood_sampling"] = nb([
md("# Notebook 13.3 – Neighborhood Sampling\nMinibatch training via neighborhood sampling (GraphSAGE style)."),
code(SETUP),
code("""
import networkx as nx, numpy as np

G = nx.karate_club_graph()
N = G.number_of_nodes()

def sample_neighbors(G, node, k):
    nbrs = list(G.neighbors(node))
    if len(nbrs) <= k: return nbrs
    return list(np.random.choice(nbrs, k, replace=False))

def sage_aggregate(G, node_feats, node, k=5, depth=2):
    if depth == 0: return node_feats[node]
    nbrs = sample_neighbors(G, node, k)
    nbr_feats = [sage_aggregate(G, node_feats, n, k, depth-1) for n in nbrs]
    if not nbr_feats: return node_feats[node]
    return np.mean([node_feats[node]] + nbr_feats, axis=0)

np.random.seed(0)
node_feats = np.random.randn(N, 4)

# Aggregate for a few nodes
print("GraphSAGE-style aggregation:")
for node in [0, 1, 5, 10]:
    agg = sage_aggregate(G, node_feats, node, k=3, depth=2)
    nbrs = list(G.neighbors(node))
    print(f"  Node {node:2d} | degree={len(nbrs):2d} | agg norm={np.linalg.norm(agg):.3f}")

# Visualise 2-hop subgraph
fig, axes = plt.subplots(1, 2, figsize=(11,5))
pos = nx.spring_layout(G, seed=0)
nx.draw(G, pos, ax=axes[0], node_size=60, with_labels=True, font_size=6)
axes[0].set_title("Full Graph")

target = 0
subgraph_nodes = {target} | set(G.neighbors(target))
for n in list(G.neighbors(target)):
    subgraph_nodes |= set(G.neighbors(n))
SG = G.subgraph(subgraph_nodes)
colors = ['red' if n == target else 'orange' if n in G.neighbors(target) else 'steelblue' for n in SG.nodes()]
nx.draw(SG, {n:pos[n] for n in SG.nodes()}, ax=axes[1], node_color=colors, with_labels=True, node_size=150, font_size=7)
axes[1].set_title("2-hop Neighbourhood of Node 0")
plt.tight_layout(); plt.savefig("neighborhood_sampling.png", dpi=100); plt.show()
"""),
])

# ── 13.4 Graph Attention ─────────────────────────────────────────────────────
notebooks["Notebook_13_4_Graph_attention"] = nb([
md("# Notebook 13.4 – Graph Attention\nGraph Attention Networks (GAT): attention over neighbours."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.nn.functional as F
import networkx as nx

class GATLayer(nn.Module):
    def __init__(self, in_feat, out_feat, heads=1):
        super().__init__()
        self.W  = nn.Linear(in_feat, out_feat * heads, bias=False)
        self.a  = nn.Parameter(torch.randn(2 * out_feat, 1) * 0.1)
        self.H  = heads; self.out = out_feat
    
    def forward(self, X, edge_index):
        N = X.shape[0]
        Wh = self.W(X).view(N, self.H, self.out)
        
        src, dst = edge_index
        Wh_src = Wh[src].squeeze(1); Wh_dst = Wh[dst].squeeze(1)
        e_ij   = F.leaky_relu((torch.cat([Wh_src, Wh_dst], dim=-1) @ self.a).squeeze(-1))
        
        # Softmax over incoming edges per node
        alpha = torch.zeros(N, N); alpha[dst, src] = e_ij
        alpha = F.softmax(alpha.masked_fill(alpha == 0, -1e9), dim=-1)
        alpha = alpha.masked_fill(alpha != alpha, 0)
        
        out = alpha @ Wh.squeeze(1)
        return F.elu(out)

G   = nx.karate_club_graph()
ei  = torch.tensor(list(G.edges())).T   # (2, E)
ei  = torch.cat([ei, ei.flip(0)], dim=1)
X   = torch.randn(G.number_of_nodes(), 8)
gat = GATLayer(8, 4)
out = gat(X, ei)
print("GAT output shape:", out.shape)

# Visualise learned attention for node 0
pos = nx.spring_layout(G, seed=0)
fig, ax = plt.subplots(figsize=(7,7))
nx.draw(G, pos, ax=ax, node_size=100, with_labels=True, font_size=7, node_color='lightblue')
ax.set_title("Karate Club — GAT applied (attention not displayed due to edge index format)")
plt.tight_layout(); plt.savefig("graph_attention.png", dpi=100); plt.show()
"""),
])

# ── 15.1 GAN Toy Example ─────────────────────────────────────────────────────
notebooks["Notebook_15_1_GAN_toy_example"] = nb([
md("# Notebook 15.1 – GAN Toy Example\nGenerative Adversarial Network on a 1-D Gaussian mixture."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)

# Real data: mixture of two Gaussians
def sample_real(n=256):
    half = n // 2
    return torch.cat([torch.randn(half,1)*0.5 - 2,
                       torch.randn(half,1)*0.5 + 2])

# Generator: noise → data
G = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,32), nn.Tanh(), nn.Linear(32,1))
# Discriminator: data → P(real)
D = nn.Sequential(nn.Linear(1,32), nn.LeakyReLU(0.2), nn.Linear(32,32), nn.LeakyReLU(0.2), nn.Linear(32,1), nn.Sigmoid())

opt_G = optim.Adam(G.parameters(), 1e-3); opt_D = optim.Adam(D.parameters(), 1e-3)
bce   = nn.BCELoss()

g_losses, d_losses = [], []
for step in range(2000):
    # --- Train D ---
    real = sample_real(128); noise = torch.randn(128, 1)
    fake = G(noise).detach()
    loss_D = bce(D(real), torch.ones(128,1)) + bce(D(fake), torch.zeros(128,1))
    opt_D.zero_grad(); loss_D.backward(); opt_D.step()
    # --- Train G ---
    fake = G(torch.randn(128,1))
    loss_G = bce(D(fake), torch.ones(128,1))
    opt_G.zero_grad(); loss_G.backward(); opt_G.step()
    if step % 50 == 0:
        g_losses.append(loss_G.item()); d_losses.append(loss_D.item())

# Plot
with torch.no_grad():
    generated = G(torch.randn(2000, 1)).squeeze().numpy()
real_np = sample_real(2000).squeeze().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.hist(real_np,      bins=60, alpha=0.5, density=True, label='Real')
ax1.hist(generated,    bins=60, alpha=0.5, density=True, label='Generated')
ax1.legend(); ax1.set_title("Real vs Generated Distribution")
ax2.plot(g_losses, label='G loss'); ax2.plot(d_losses, label='D loss'); ax2.legend()
ax2.set_title("GAN Losses"); plt.tight_layout()
plt.savefig("gan_toy.png", dpi=100); plt.show()
"""),
])

# ── 15.2 Wasserstein Distance ────────────────────────────────────────────────
notebooks["Notebook_15_2_Wasserstein_distance"] = nb([
md("# Notebook 15.2 – Wasserstein Distance\nEarth mover's distance and WGAN training."),
code(SETUP),
code("""
from scipy.stats import wasserstein_distance

# Compare Wasserstein vs JS divergence as distributions shift
mu_list = np.linspace(-5, 5, 50)
ks_divs, w_dists = [], []

p = np.random.randn(2000)  # reference
for mu in mu_list:
    q = np.random.randn(2000) + mu
    wd = wasserstein_distance(p, q)
    # KL approximation via histograms
    bins = np.linspace(-8, 8, 100)
    hp,_ = np.histogram(p, bins, density=True); hp += 1e-10
    hq,_ = np.histogram(q, bins, density=True); hq += 1e-10
    m = (hp + hq) / 2
    js = 0.5 * np.sum(hp * np.log(hp/m) + hq * np.log(hq/m)) * (bins[1]-bins[0])
    w_dists.append(wd); ks_divs.append(js)

fig, axes = plt.subplots(1, 2, figsize=(12,4))
axes[0].plot(mu_list, w_dists, 'b'); axes[0].set_title("Wasserstein Distance vs μ shift")
axes[1].plot(mu_list, ks_divs, 'r'); axes[1].set_title("JS Divergence vs μ shift")
for ax in axes: ax.set_xlabel("μ (mean shift)")
plt.suptitle("Wasserstein vs JS Divergence"); plt.tight_layout()
plt.savefig("wasserstein.png", dpi=100); plt.show()
"""),
])

# ── 16.1 1-D Normalizing Flows ───────────────────────────────────────────────
notebooks["Notebook_16_1_1D_normalizing_flows"] = nb([
md("# Notebook 16.1 – 1-D Normalizing Flows\nMapping a simple distribution to a complex one via invertible transforms."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)

# Target: mixture of Gaussians
def sample_target(n=1000):
    idx = torch.randint(0, 3, (n,))
    means = torch.tensor([-3.0, 0.0, 3.0])
    x = torch.randn(n) * 0.5 + means[idx]
    return x.unsqueeze(1)

# Affine coupling layer (1-D toy)
class AffineCoupling(nn.Module):
    def __init__(self):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,1))
        self.t = nn.Sequential(nn.Linear(1,16),nn.Tanh(),nn.Linear(16,1))
    def forward(self, x):
        s = self.s(x); t = self.t(x)
        z = x * torch.exp(s) + t
        log_det = s.sum(-1)
        return z, log_det
    def inverse(self, z):
        s = self.s(z)
        return (z - self.t(z)) * torch.exp(-s)

flow = AffineCoupling()
opt  = optim.Adam(flow.parameters(), 1e-3)

# Train by maximising log-likelihood
losses = []
for _ in range(1000):
    x = sample_target(256)
    z, log_det = flow(x)
    log_pz = -0.5 * (z**2).sum(-1) - 0.5*np.log(2*np.pi)
    loss   = -(log_pz + log_det).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())

# Sample from flow
with torch.no_grad():
    z_sample = torch.randn(2000, 1)
    x_sample  = flow.inverse(z_sample).squeeze().numpy()
    x_real    = sample_target(2000).squeeze().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.plot(losses); ax1.set_title("Training Loss (NLL)")
ax2.hist(x_real,   bins=60, density=True, alpha=0.5, label='Target')
ax2.hist(x_sample, bins=60, density=True, alpha=0.5, label='Flow samples')
ax2.legend(); ax2.set_title("1-D Normalizing Flow")
plt.tight_layout(); plt.savefig("nf_1d.png", dpi=100); plt.show()
"""),
])

# ── 16.2 Autoregressive Flows ────────────────────────────────────────────────
notebooks["Notebook_16_2_Autoregressive_flows"] = nb([
md("# Notebook 16.2 – Autoregressive Flows\nMasked Autoregressive Flow (MAF) concept."),
code(SETUP),
code("""
import torch, torch.nn as nn

# Minimal masked autoregressive model (MADE-style, 2-D for illustration)
class MADE2D(nn.Module):
    \"\"\"Autoregressive model for 2-D x = (x1, x2).
       p(x) = p(x1) * p(x2 | x1)
    \"\"\"
    def __init__(self):
        super().__init__()
        # x1 parameters (unconditional)
        self.mu1    = nn.Parameter(torch.zeros(1))
        self.log_s1 = nn.Parameter(torch.zeros(1))
        # x2 | x1 network
        self.net = nn.Sequential(nn.Linear(1,32), nn.Tanh(), nn.Linear(32,2))  # mu, log_s
    
    def log_prob(self, x):
        x1, x2 = x[:,0:1], x[:,1:2]
        # p(x1)
        lp1 = (-0.5*((x1 - self.mu1) / self.log_s1.exp())**2
                - self.log_s1 - 0.5*np.log(2*np.pi)).squeeze(-1)
        # p(x2|x1)
        out = self.net(x1)
        mu2, logs2 = out[:,0:1], out[:,1:2]
        lp2 = (-0.5*((x2 - mu2) / logs2.exp())**2
                - logs2 - 0.5*np.log(2*np.pi)).squeeze(-1)
        return lp1 + lp2
    
    def sample(self, n):
        x1 = torch.randn(n,1) * self.log_s1.exp() + self.mu1
        out = self.net(x1.detach())
        mu2, logs2 = out[:,0:1], out[:,1:2]
        x2 = torch.randn(n,1) * logs2.exp() + mu2
        return torch.cat([x1, x2], dim=1)

# Target: banana distribution
def sample_banana(n=2000):
    x1 = torch.randn(n)
    x2 = x1**2 + torch.randn(n)*0.5
    return torch.stack([x1, x2], dim=1)

model = MADE2D()
opt   = torch.optim.Adam(model.parameters(), 1e-3)

for _ in range(2000):
    x = sample_banana(256)
    loss = -model.log_prob(x).mean()
    opt.zero_grad(); loss.backward(); opt.step()

with torch.no_grad():
    real    = sample_banana(2000).numpy()
    sampled = model.sample(2000).numpy()

fig, axes = plt.subplots(1, 2, figsize=(11,5))
axes[0].scatter(*real.T,    s=2, alpha=0.3); axes[0].set_title("Target (Banana)")
axes[1].scatter(*sampled.T, s=2, alpha=0.3, c='r'); axes[1].set_title("MAF Samples")
plt.suptitle("Autoregressive Flow"); plt.tight_layout()
plt.savefig("autoregressive_flow.png", dpi=100); plt.show()
"""),
])

# ── 16.3 Contraction Mappings ────────────────────────────────────────────────
notebooks["Notebook_16_3_Contraction_mappings"] = nb([
md("# Notebook 16.3 – Contraction Mappings\nFixed-point theory and its role in normalizing flows."),
code(SETUP),
code("""
# Contraction mapping: |f(x) - f(y)| <= L * |x - y|, L < 1
# Fixed point: f(x*) = x*

def iterate(f, x0, n=50):
    path = [x0]
    for _ in range(n):
        x0 = f(x0)
        path.append(x0)
    return np.array(path)

f1 = lambda x: 0.5 * np.cos(x)   # contraction (L ≈ 0.5)
f2 = lambda x: np.cos(x)          # borderline (L ≈ 1 at fixed point)

x0_vals = [-2, -1, 0, 1, 2]
fig, axes = plt.subplots(1, 2, figsize=(12,5))
x_line = np.linspace(-3, 3, 300)
for ax, f, title in zip(axes, [f1, f2], ['L≈0.5 (contraction)', 'L≈1 (borderline)']):
    ax.plot(x_line, f(x_line), 'b', lw=2, label='f(x)')
    ax.plot(x_line, x_line,    'k--', label='y=x')
    for x0 in x0_vals:
        path = iterate(f, x0, n=30)
        ax.plot(path, np.array([f(p) for p in path]), 'r.-', ms=3, alpha=0.5)
        ax.scatter(path[-1], path[-1], c='g', s=50)
    ax.set_title(title); ax.legend()
plt.suptitle("Contraction Mappings & Fixed Points"); plt.tight_layout()
plt.savefig("contraction.png", dpi=100); plt.show()
"""),
])

# ── 17.1 Latent Variable Models ──────────────────────────────────────────────
notebooks["Notebook_17_1_Latent_variable_models"] = nb([
md("# Notebook 17.1 – Latent Variable Models\nGaussian Mixture Model via EM algorithm."),
code(SETUP),
code("""
from scipy.stats import multivariate_normal

np.random.seed(2)
# True GMM: 3 components in 2-D
K_true = 3
mus_true  = np.array([[-3,0],[0,3],[3,0]], dtype=float)
covs_true = [np.eye(2)*0.5 for _ in range(K_true)]
pis_true  = np.array([0.3, 0.4, 0.3])

# Sample data
N = 400
z = np.random.choice(K_true, N, p=pis_true)
X_gmm = np.vstack([np.random.multivariate_normal(mus_true[k], covs_true[k]) for k in z])

# EM algorithm
K = 3
mu  = X_gmm[np.random.choice(N, K, replace=False)]
cov = [np.eye(2) for _ in range(K)]
pi  = np.ones(K) / K

def e_step(X, mu, cov, pi):
    N = len(X)
    R = np.zeros((N, K))
    for k in range(K):
        R[:,k] = pi[k] * multivariate_normal.pdf(X, mu[k], cov[k])
    R /= R.sum(1, keepdims=True)
    return R

def m_step(X, R):
    N = len(X)
    Nk = R.sum(0)
    mu  = (R.T @ X) / Nk.reshape(-1,1)
    cov = [((R[:,k:k+1] * (X - mu[k])).T @ (X - mu[k])) / Nk[k] for k in range(K)]
    pi  = Nk / N
    return mu, cov, pi

for it in range(30):
    R      = e_step(X_gmm, mu, cov, pi)
    mu, cov, pi = m_step(X_gmm, R)

labels = R.argmax(1)
plt.figure(figsize=(7,6))
plt.scatter(*X_gmm.T, c=labels, cmap='tab10', s=20, alpha=0.7)
plt.scatter(*mu.T, c='k', marker='X', s=200, label='EM centres')
plt.legend(); plt.title("GMM fitted via EM")
plt.tight_layout(); plt.savefig("gmm_em.png", dpi=100); plt.show()
"""),
])

# ── 17.2 Reparameterization Trick ────────────────────────────────────────────
notebooks["Notebook_17_2_Reparameterization_trick"] = nb([
md("# Notebook 17.2 – Reparameterization Trick\nAllowing gradients to flow through stochastic sampling."),
code(SETUP),
code("""
import torch

torch.manual_seed(0)

mu    = torch.tensor(2.0, requires_grad=True)
log_s = torch.tensor(0.0, requires_grad=True)

# Without reparameterisation: gradient doesn't flow
# With reparameterisation: z = mu + sigma * eps,  eps ~ N(0,1)

def estimate_grad_reparam(mu, log_s, n=1000):
    eps = torch.randn(n)
    z   = mu + log_s.exp() * eps
    f_z = z**2                # E[z²] = mu² + sigma²
    loss = f_z.mean()
    loss.backward()
    return mu.grad.item(), log_s.grad.item()

def estimate_grad_score(mu, log_s, n=1000):
    \"\"\"Score function / REINFORCE estimator\"\"\"
    with torch.no_grad():
        eps = torch.randn(n)
        z   = mu + log_s.exp() * eps
        f_z = z**2
        score_mu  = (f_z * (z - mu) / log_s.exp()**2).mean()
        score_s   = (f_z * ((z - mu)**2 / log_s.exp()**3 - 1/log_s.exp())).mean()
    return score_mu.item(), score_s.item()

# Reparameterisation
gmu_rep, gs_rep = estimate_grad_reparam(mu, log_s)
print(f"Reparam  | dL/dmu={gmu_rep:.4f}  dL/dlog_s={gs_rep:.4f}")

# Analytical: E[z²] = mu² + exp(2*log_s), dL/dmu = 2*mu, dL/dlog_s = 2*exp(2*log_s)
print(f"Analytic | dL/dmu={2*mu.item():.4f}  dL/dlog_s={2*log_s.exp().item()**2:.4f}")

# Visualise
N_samp = [10, 50, 200, 1000, 5000]
rep_errs, score_errs = [], []
for n in N_samp:
    mu2 = torch.tensor(2.0, requires_grad=True); ls2 = torch.tensor(0.0, requires_grad=True)
    gr, _ = estimate_grad_reparam(mu2, ls2, n=n)
    rep_errs.append(abs(gr - 4.0))
    gs, _ = estimate_grad_score(torch.tensor(2.0), torch.tensor(0.0), n=n)
    score_errs.append(abs(gs - 4.0))
    if mu2.grad: mu2.grad.zero_()

plt.figure(figsize=(7,4))
plt.loglog(N_samp, rep_errs,   'b-o', label='Reparameterisation')
plt.loglog(N_samp, score_errs, 'r-o', label='Score function')
plt.xlabel('N samples'); plt.ylabel('|grad error|')
plt.legend(); plt.title("Gradient Estimator Variance")
plt.tight_layout(); plt.savefig("reparam_trick.png", dpi=100); plt.show()
"""),
])

# ── 17.3 Importance Sampling ─────────────────────────────────────────────────
notebooks["Notebook_17_3_Importance_sampling"] = nb([
md("# Notebook 17.3 – Importance Sampling\nEstimating expectations under a different proposal distribution."),
code(SETUP),
code("""
from scipy.stats import norm

# Estimate E_{p}[f(x)] where p = N(0,1) and f(x) = x^4

# True value: E[x^4] for N(0,1) = 3 (fourth moment)
true_val = 3.0

# Proposal q = N(mu_q, sigma_q)
def importance_sample(n, mu_q=2.0, sigma_q=1.0):
    x     = np.random.randn(n) * sigma_q + mu_q        # sample from q
    log_w = norm.logpdf(x, 0, 1) - norm.logpdf(x, mu_q, sigma_q)  # log importance weights
    w     = np.exp(log_w - log_w.max()); w /= w.sum()   # normalise
    f_x   = x**4
    return np.sum(w * f_x), np.var(w)

n_vals = [10, 50, 200, 1000, 5000, 20000]
estimates, variances = zip(*[importance_sample(n) for n in n_vals])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.semilogx(n_vals, estimates, 'b-o')
ax1.axhline(true_val, c='r', ls='--', label=f'True = {true_val}')
ax1.legend(); ax1.set_xlabel("N"); ax1.set_title("IS Estimate of E[x⁴]")
ax2.loglog(n_vals, variances, 'g-o')
ax2.set_xlabel("N"); ax2.set_title("Weight Variance")
plt.suptitle("Importance Sampling"); plt.tight_layout()
plt.savefig("importance_sampling.png", dpi=100); plt.show()
"""),
])

# ── 18.1 Diffusion Encoder ───────────────────────────────────────────────────
notebooks["Notebook_18_1_Diffusion_encoder"] = nb([
md("# Notebook 18.1 – Diffusion Encoder\nForward diffusion process: gradually adding noise to data."),
code(SETUP),
code("""
# Forward diffusion (DDPM): q(x_t | x_0)
# x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * eps

T = 1000
beta_min, beta_max = 1e-4, 0.02
betas = np.linspace(beta_min, beta_max, T)
alphas = 1 - betas
alpha_bar = np.cumprod(alphas)

t_vals = [0, 100, 250, 500, 750, 999]

# Synthetic 2-D data: Swiss roll-like circles
theta = np.linspace(0, 4*np.pi, 500)
x0 = np.column_stack([np.cos(theta)*theta/10, np.sin(theta)*theta/10])

fig, axes = plt.subplots(1, len(t_vals), figsize=(16, 3))
for ax, t in zip(axes, t_vals):
    sqrt_ab = np.sqrt(alpha_bar[t])
    sqrt_1ab= np.sqrt(1 - alpha_bar[t])
    xt = sqrt_ab * x0 + sqrt_1ab * np.random.randn(*x0.shape)
    ax.scatter(*xt.T, s=4, alpha=0.7)
    ax.set_title(f"t={t}")
    ax.set_xlim(-3,3); ax.set_ylim(-3,3); ax.axis('equal')
plt.suptitle("Forward Diffusion Process"); plt.tight_layout()
plt.savefig("diffusion_encoder.png", dpi=100); plt.show()

plt.figure(figsize=(8,3))
plt.subplot(1,2,1); plt.plot(alpha_bar); plt.title("ᾱ_t (signal retention)")
plt.subplot(1,2,2); plt.plot(1-alpha_bar, 'r'); plt.title("1-ᾱ_t (noise level)")
plt.tight_layout(); plt.savefig("noise_schedule.png", dpi=100); plt.show()
"""),
])

# ── 18.2 1-D Diffusion Model ─────────────────────────────────────────────────
notebooks["Notebook_18_2_1D_diffusion_model"] = nb([
md("# Notebook 18.2 – 1-D Diffusion Model\nTraining a simple denoising network for 1-D data."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)
T = 200
betas      = torch.linspace(1e-4, 0.02, T)
alphas     = 1 - betas
alpha_bar  = torch.cumprod(alphas, 0)

# Target: bimodal 1-D distribution
def sample_data(n): return torch.cat([torch.randn(n//2)*0.5 - 2, torch.randn(n//2)*0.5 + 2]).unsqueeze(1)

# Score/noise prediction network
class DenoiseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
            nn.Linear(64, 1))
    def forward(self, x, t_emb): return self.net(torch.cat([x, t_emb], -1))

model = DenoiseMLP()
opt   = optim.Adam(model.parameters(), 1e-3)

losses = []
for step in range(3000):
    x0 = sample_data(128)
    t  = torch.randint(0, T, (128,))
    ab = alpha_bar[t].unsqueeze(1)
    eps = torch.randn_like(x0)
    xt  = ab.sqrt() * x0 + (1-ab).sqrt() * eps
    t_emb = (t.float() / T).unsqueeze(1)
    eps_pred = model(xt, t_emb)
    loss = ((eps - eps_pred)**2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 100 == 0: losses.append(loss.item())

# Sampling (DDPM reverse)
@torch.no_grad()
def ddpm_sample(n=500):
    x = torch.randn(n, 1)
    for t in reversed(range(T)):
        t_emb = torch.full((n,1), t/T)
        eps_pred = model(x, t_emb)
        alpha_t = alphas[t]; ab_t = alpha_bar[t]
        mu = (x - (1-alpha_t)/((1-ab_t).sqrt()) * eps_pred) / alpha_t.sqrt()
        if t > 0:
            sigma = betas[t].sqrt()
            x = mu + sigma * torch.randn_like(x)
        else:
            x = mu
    return x.squeeze().numpy()

samples = ddpm_sample()
real    = sample_data(500).squeeze().numpy()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,4))
ax1.plot(losses); ax1.set_title("Training Loss")
ax2.hist(real,    bins=50, alpha=0.5, density=True, label='Real')
ax2.hist(samples, bins=50, alpha=0.5, density=True, label='Sampled')
ax2.legend(); ax2.set_title("1-D Diffusion Model")
plt.tight_layout(); plt.savefig("diffusion_1d.png", dpi=100); plt.show()
"""),
])

# ── 18.3 Reparameterized Model ───────────────────────────────────────────────
notebooks["Notebook_18_3_Reparameterized_model"] = nb([
md("# Notebook 18.3 – Reparameterized Diffusion Model\nx0-prediction vs ε-prediction parameterisation."),
code(SETUP),
code("""
import torch, torch.nn as nn

T = 200
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alpha_bar = torch.cumprod(alphas, 0)

# x_0 prediction: model predicts x_0 directly
# Relationship: eps = (x_t - sqrt(ab)*x0_pred) / sqrt(1-ab)

class X0Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self, xt, t_emb): return self.net(torch.cat([xt, t_emb], -1))

class EpsNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
    def forward(self, xt, t_emb): return self.net(torch.cat([xt, t_emb], -1))

torch.manual_seed(1)

def sample_data(n):
    return torch.cat([torch.randn(n//2)*0.5-2, torch.randn(n//2)*0.5+2]).unsqueeze(1)

def train(model, predict='eps', steps=2000):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    losses = []
    for _ in range(steps):
        x0 = sample_data(128)
        t  = torch.randint(0, T, (128,))
        ab = alpha_bar[t].unsqueeze(1)
        eps = torch.randn_like(x0)
        xt  = ab.sqrt() * x0 + (1-ab).sqrt() * eps
        t_emb = (t.float()/T).unsqueeze(1)
        pred = model(xt, t_emb)
        if predict == 'x0':
            loss = ((x0 - pred)**2).mean()
        else:
            loss = ((eps - pred)**2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return losses

model_eps = EpsNet(); model_x0 = X0Net()
losses_eps = train(model_eps, 'eps')
losses_x0  = train(model_x0,  'x0')

plt.figure(figsize=(8,4))
plt.plot(losses_eps, label='ε-prediction')
plt.plot(losses_x0,  label='x₀-prediction')
plt.legend(); plt.title("Reparameterized Diffusion: Training Loss Comparison")
plt.tight_layout(); plt.savefig("reparam_diffusion.png", dpi=100); plt.show()
"""),
])

# ── 18.4 Families of Diffusion Models ───────────────────────────────────────
notebooks["Notebook_18_4_Families_of_diffusion_models"] = nb([
md("# Notebook 18.4 – Families of Diffusion Models\nDDPM vs DDIM vs Score-Based; noise schedules."),
code(SETUP),
code("""
import torch

T = 1000

def linear_schedule(T):
    return torch.linspace(1e-4, 0.02, T)

def cosine_schedule(T):
    t = torch.arange(T+1) / T
    alpha_bar = torch.cos((t + 0.008)/(1.008) * np.pi/2)**2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(0, 0.999)

def quadratic_schedule(T):
    return torch.linspace(1e-4**0.5, 0.02**0.5, T)**2

schedules = {
    'Linear':    linear_schedule(T),
    'Cosine':    cosine_schedule(T),
    'Quadratic': quadratic_schedule(T),
}

fig, axes = plt.subplots(1, 3, figsize=(14,4))
t_range = np.arange(T)
for name, betas in schedules.items():
    alphas = 1 - betas
    ab = torch.cumprod(alphas, 0).numpy()
    axes[0].plot(t_range, betas.numpy(), label=name)
    axes[1].plot(t_range, ab, label=name)
    snr = ab / (1 - ab + 1e-8)
    axes[2].semilogy(t_range, snr, label=name)
axes[0].set_title("β_t (noise schedule)"); axes[0].legend()
axes[1].set_title("ᾱ_t (signal retention)"); axes[1].legend()
axes[2].set_title("SNR = ᾱ_t / (1-ᾱ_t)"); axes[2].legend()
plt.suptitle("Families of Diffusion Noise Schedules")
plt.tight_layout(); plt.savefig("diffusion_families.png", dpi=100); plt.show()
"""),
])

# ── 19.1 Markov Decision Processes ───────────────────────────────────────────
notebooks["Notebook_19_1_Markov_decision_processes"] = nb([
md("# Notebook 19.1 – Markov Decision Processes\nStates, actions, rewards, and the Bellman equation."),
code(SETUP),
code("""
# Simple GridWorld MDP
import numpy as np

# 4x4 grid, states 0..15, goal at state 15
S = 16; A = 4  # up, down, left, right
gamma = 0.9

def next_state(s, a):
    row, col = divmod(s, 4)
    if a == 0: row = max(0, row-1)
    elif a == 1: row = min(3, row+1)
    elif a == 2: col = max(0, col-1)
    elif a == 3: col = min(3, col+1)
    return row*4 + col

def reward(s): return 1.0 if s == 15 else -0.01

# Value iteration
V = np.zeros(S)
for _ in range(200):
    V_new = np.zeros(S)
    for s in range(S):
        if s == 15: V_new[s] = 1.0; continue
        qs = [reward(s) + gamma * V[next_state(s, a)] for a in range(A)]
        V_new[s] = max(qs)
    V = V_new

print("Value function (4x4 grid):")
print(V.reshape(4,4).round(3))

# Policy extraction
pi = [np.argmax([reward(s) + gamma*V[next_state(s,a)] for a in range(A)]) for s in range(S)]
arrows = ['↑','↓','←','→']
print("\nPolicy:")
print(np.array([arrows[p] for p in pi]).reshape(4,4))

plt.figure(figsize=(6,5))
plt.imshow(V.reshape(4,4), cmap='YlOrRd')
plt.colorbar(); plt.title("GridWorld Value Function")
for i in range(4):
    for j in range(4):
        plt.text(j, i, f"{V[i*4+j]:.2f}", ha='center', va='center', fontsize=9)
plt.tight_layout(); plt.savefig("mdp_values.png", dpi=100); plt.show()
"""),
])

# ── 19.2 Dynamic Programming ─────────────────────────────────────────────────
notebooks["Notebook_19_2_Dynamic_programming"] = nb([
md("# Notebook 19.2 – Dynamic Programming\nPolicy iteration and value iteration."),
code(SETUP),
code("""
import numpy as np

S = 16; A = 4; gamma = 0.9

def next_state(s, a):
    row, col = divmod(s, 4)
    if a == 0: row = max(0, row-1)
    elif a == 1: row = min(3, row+1)
    elif a == 2: col = max(0, col-1)
    elif a == 3: col = min(3, col+1)
    return row*4 + col

def R(s): return 1.0 if s == 15 else -0.04

# Policy Iteration
def policy_eval(pi, V, tol=1e-6):
    for _ in range(1000):
        V_new = np.array([R(s) + gamma * V[next_state(s, pi[s])] for s in range(S)])
        if np.max(np.abs(V_new - V)) < tol: break
        V = V_new
    return V

def policy_improve(V):
    return [np.argmax([R(s) + gamma * V[next_state(s,a)] for a in range(A)]) for s in range(S)]

pi = [0]*S  # random init
V  = np.zeros(S)
policy_history = []
for it in range(50):
    V  = policy_eval(pi, V)
    pi_new = policy_improve(V)
    policy_history.append(pi[:])
    if pi_new == pi: print(f"Converged at iteration {it+1}"); break
    pi = pi_new

arrows = ['↑','↓','←','→']
print("Optimal Policy:")
print(np.array([arrows[p] for p in pi]).reshape(4,4))
print("\nOptimal Values:")
print(V.reshape(4,4).round(3))

# Value iteration vs Policy iteration convergence
VI_vals = []; V_vi = np.zeros(S)
for it in range(100):
    V_new = np.zeros(S)
    for s in range(S):
        if s==15: V_new[s]=1.0; continue
        V_new[s] = max(R(s)+gamma*V_vi[next_state(s,a)] for a in range(A))
    VI_vals.append(np.max(np.abs(V_new - V_vi)))
    V_vi = V_new
    if VI_vals[-1] < 1e-8: break

plt.figure(figsize=(7,4))
plt.semilogy(VI_vals)
plt.xlabel('Iteration'); plt.ylabel('Max Bellman Error')
plt.title('Value Iteration Convergence'); plt.tight_layout()
plt.savefig("dynamic_programming.png", dpi=100); plt.show()
"""),
])

# ── 19.3 Monte-Carlo Methods ──────────────────────────────────────────────────
notebooks["Notebook_19_3_Monte-Carlo_methods"] = nb([
md("# Notebook 19.3 – Monte-Carlo Methods\nMC policy evaluation by averaging episode returns."),
code(SETUP),
code("""
import numpy as np

S = 16; A = 4; gamma = 0.9
np.random.seed(0)

def next_state(s, a):
    row, col = divmod(s, 4)
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    r = max(0,min(3,row+moves[a][0])); c = max(0,min(3,col+moves[a][1]))
    return r*4+c

def R(s): return 1.0 if s==15 else -0.04

# Random policy
def random_policy(s): return np.random.randint(A)

def run_episode(pi, max_steps=100):
    s = np.random.randint(S); traj = []
    for _ in range(max_steps):
        if s == 15: traj.append((s, None, 1.0)); break
        a = pi(s); r = R(s); traj.append((s,a,r)); s = next_state(s,a)
    return traj

# MC first-visit evaluation
returns = {s:[] for s in range(S)}
V = np.zeros(S)

for ep in range(10000):
    traj = run_episode(random_policy)
    G = 0; visited = set()
    for t in reversed(range(len(traj))):
        s, a, r = traj[t]; G = r + gamma*G
        if s not in visited:
            visited.add(s); returns[s].append(G); V[s] = np.mean(returns[s])

print("MC Value Estimates:")
print(V.reshape(4,4).round(3))

plt.figure(figsize=(5,4))
plt.imshow(V.reshape(4,4), cmap='YlOrRd')
plt.colorbar(); plt.title("Monte-Carlo Value Estimates (random policy)")
for i in range(4):
    for j in range(4):
        plt.text(j,i,f"{V[i*4+j]:.2f}", ha='center', va='center', fontsize=9)
plt.tight_layout(); plt.savefig("monte_carlo.png", dpi=100); plt.show()
"""),
])

# ── 19.4 Temporal Difference Methods ────────────────────────────────────────
notebooks["Notebook_19_4_Temporal_difference_methods"] = nb([
md("# Notebook 19.4 – Temporal Difference Methods\nTD(0), SARSA, Q-learning."),
code(SETUP),
code("""
import numpy as np

S = 16; A = 4; gamma = 0.9; alpha = 0.1; eps = 0.1
np.random.seed(0)

def next_state(s, a):
    row, col = divmod(s, 4)
    moves = [(-1,0),(1,0),(0,-1),(0,1)]
    r = max(0,min(3,row+moves[a][0])); c = max(0,min(3,col+moves[a][1]))
    return r*4+c

def R(s): return 1.0 if s==15 else -0.04

def eps_greedy(Q, s, eps=0.1):
    if np.random.rand() < eps: return np.random.randint(A)
    return Q[s].argmax()

# Q-learning
Q = np.zeros((S, A))
ep_rewards = []
for ep in range(5000):
    s = np.random.randint(S); total_r = 0
    for _ in range(200):
        a = eps_greedy(Q, s)
        s2 = next_state(s, a); r = R(s)
        Q[s,a] += alpha * (r + gamma * Q[s2].max() - Q[s,a])
        total_r += r; s = s2
        if s == 15: break
    ep_rewards.append(total_r)

# SARSA
Q_s = np.zeros((S, A))
ep_rewards_s = []
for ep in range(5000):
    s = np.random.randint(S); a = eps_greedy(Q_s, s); total_r = 0
    for _ in range(200):
        s2 = next_state(s, a); r = R(s); a2 = eps_greedy(Q_s, s2)
        Q_s[s,a] += alpha * (r + gamma * Q_s[s2,a2] - Q_s[s,a])
        total_r += r; s, a = s2, a2
        if s == 15: break
    ep_rewards_s.append(total_r)

def smooth(arr, w=200): return np.convolve(arr, np.ones(w)/w, 'valid')

plt.figure(figsize=(8,4))
plt.plot(smooth(ep_rewards),   label='Q-Learning')
plt.plot(smooth(ep_rewards_s), label='SARSA')
plt.xlabel('Episode'); plt.ylabel('Total Reward (smoothed)')
plt.legend(); plt.title("TD Methods on GridWorld")
plt.tight_layout(); plt.savefig("td_methods.png", dpi=100); plt.show()
"""),
])

# ── 19.5 Control Variates ────────────────────────────────────────────────────
notebooks["Notebook_19_5_Control_variates"] = nb([
md("# Notebook 19.5 – Control Variates\nVariance reduction in policy gradient estimation."),
code(SETUP),
code("""
import numpy as np

np.random.seed(0)

# Estimating E[f(x)] where x ~ Bernoulli(p), f(x) = x^2
# Policy gradient: grad_p E[f(x)] = E[f(x) * score(x;p)]

p = 0.4  # true parameter

def f(x): return x**2
def score(x, p): return (x/p - (1-x)/(1-p))   # d/dp log Bernoulli(x;p)

n_samples = 100
n_trials  = 500

# Without baseline
grads_raw = []
for _ in range(n_trials):
    x = (np.random.rand(n_samples) < p).astype(float)
    g = np.mean(f(x) * score(x, p))
    grads_raw.append(g)

# With baseline b = E[f(x)] ≈ empirical mean
grads_cv = []
for _ in range(n_trials):
    x = (np.random.rand(n_samples) < p).astype(float)
    b = np.mean(f(x))  # baseline
    g = np.mean((f(x) - b) * score(x, p))
    grads_cv.append(g)

true_grad = p   # d/dp E[x^2] = d/dp p = 1 ? No: E[X^2] = p, d/dp=1
# Actually E[x^2] = p for Bernoulli, so true gradient = 1
true_grad = 1.0

print(f"Raw estimator: mean={np.mean(grads_raw):.3f}, std={np.std(grads_raw):.3f}")
print(f"CV  estimator: mean={np.mean(grads_cv):.3f},  std={np.std(grads_cv):.3f}")
print(f"True gradient: {true_grad}")

fig, (ax1,ax2) = plt.subplots(1, 2, figsize=(11,4))
ax1.hist(grads_raw, bins=40, alpha=0.7, label=f'Raw (σ={np.std(grads_raw):.2f})')
ax1.hist(grads_cv,  bins=40, alpha=0.7, label=f'CV  (σ={np.std(grads_cv):.2f})')
ax1.axvline(true_grad, c='k', ls='--', label='True'); ax1.legend()
ax1.set_title("Gradient Estimates Distribution")
ax2.plot(np.cumsum(grads_raw)/np.arange(1,n_trials+1), label='Raw')
ax2.plot(np.cumsum(grads_cv) /np.arange(1,n_trials+1), label='CV')
ax2.axhline(true_grad, c='k', ls='--'); ax2.legend(); ax2.set_title("Running Mean")
plt.suptitle("Control Variates for Variance Reduction"); plt.tight_layout()
plt.savefig("control_variates.png", dpi=100); plt.show()
"""),
])

# ── 20.1 Random Data ─────────────────────────────────────────────────────────
notebooks["Notebook_20_1_Random_data"] = nb([
md("# Notebook 20.1 – Random Data\nNeural networks can memorise random labels — understanding capacity."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0); np.random.seed(0)
N, D = 200, 20
X = torch.randn(N, D)

results = {}
for label_type in ['true', 'random']:
    if label_type == 'true':
        y = (X[:,0] > 0).long()          # real pattern
    else:
        y = torch.randint(0, 2, (N,))    # random labels

    model = nn.Sequential(nn.Linear(D,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU(),nn.Linear(64,2))
    opt   = optim.Adam(model.parameters(), 1e-3)
    accs  = []
    for ep in range(200):
        out  = model(X)
        loss = nn.CrossEntropyLoss()(out, y)
        opt.zero_grad(); loss.backward(); opt.step()
        acc  = (out.argmax(1) == y).float().mean().item()
        accs.append(acc)
    results[label_type] = accs

plt.figure(figsize=(7,4))
for label_type, accs in results.items():
    plt.plot(accs, label=f'{label_type} labels')
plt.xlabel('Epoch'); plt.ylabel('Train Accuracy'); plt.legend()
plt.title("Neural Nets Can Memorise Random Labels")
plt.tight_layout(); plt.savefig("random_data.png", dpi=100); plt.show()
"""),
])

# ── 20.2 Full-Batch Gradient Descent ────────────────────────────────────────
notebooks["Notebook_20_2_Full-batch_gradient_descent"] = nb([
md("# Notebook 20.2 – Full-Batch Gradient Descent\nSharp vs flat minima; loss landscape exploration."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)
N, D = 500, 10
X = torch.randn(N, D); y = (X[:,0] > 0).long()

def train(batch_size, lr=1e-3, epochs=300):
    model = nn.Sequential(nn.Linear(D,64),nn.ReLU(),nn.Linear(64,2))
    opt   = optim.SGD(model.parameters(), lr=lr)
    losses= []
    for ep in range(epochs):
        perm = torch.randperm(N)
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            loss = nn.CrossEntropyLoss()(model(X[idx]), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            losses.append(nn.CrossEntropyLoss()(model(X), y).item())
    return losses, model

plt.figure(figsize=(8,4))
for bs in [N, 128, 32, 8]:
    losses, _ = train(bs)
    plt.plot(losses, label=f'batch={bs}')
plt.xlabel('Epoch'); plt.ylabel('Full-batch Loss'); plt.legend()
plt.title("Full-Batch vs Mini-Batch GD Loss Curves")
plt.tight_layout(); plt.savefig("fullbatch_gd.png", dpi=100); plt.show()
"""),
])

# ── 20.3 Lottery Tickets ─────────────────────────────────────────────────────
notebooks["Notebook_20_3_Lottery_tickets"] = nb([
md("# Notebook 20.3 – Lottery Tickets\nThe Lottery Ticket Hypothesis: sparse subnetworks that train well."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim
import copy

torch.manual_seed(0)
N, D, H, K = 500, 10, 64, 2
X = torch.randn(N, D); y = (X[:,0] + X[:,1] > 0).long()

def build_model():
    return nn.Sequential(nn.Linear(D,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,K))

def train_model(model, mask=None, epochs=200, lr=1e-3):
    opt = optim.Adam(model.parameters(), lr)
    losses = []
    for ep in range(epochs):
        out  = model(X)
        loss = nn.CrossEntropyLoss()(out, y)
        opt.zero_grad(); loss.backward()
        if mask:
            for name, param in model.named_parameters():
                if name in mask and param.grad is not None:
                    param.grad.data *= mask[name]
        opt.step()
        losses.append(loss.item())
    acc = (model(X).argmax(1)==y).float().mean().item()
    return losses, acc

def magnitude_prune(model, sparsity=0.7):
    mask = {}
    for name, param in model.named_parameters():
        if 'weight' in name:
            flat = param.data.abs().view(-1)
            threshold = flat.kthvalue(int(sparsity * len(flat)))[0]
            mask[name] = (param.data.abs() >= threshold).float()
    return mask

# Full model
model_full = build_model()
init_state = copy.deepcopy(model_full.state_dict())
losses_full, acc_full = train_model(model_full)
print(f"Full model acc: {acc_full:.3f}")

# Lottery ticket: prune + re-init to original weights
model_full2 = build_model(); model_full2.load_state_dict(copy.deepcopy(init_state))
_, _ = train_model(model_full2)   # train to get magnitudes
mask = magnitude_prune(model_full2, sparsity=0.8)

# Re-initialise with mask
model_lt = build_model(); model_lt.load_state_dict(copy.deepcopy(init_state))
losses_lt, acc_lt = train_model(model_lt, mask=mask)
print(f"Lottery ticket (80% sparse) acc: {acc_lt:.3f}")

plt.figure(figsize=(7,4))
plt.plot(losses_full, label='Full Network')
plt.plot(losses_lt,   label='Lottery Ticket (80% sparse)')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
plt.title("Lottery Ticket Hypothesis"); plt.tight_layout()
plt.savefig("lottery_tickets.png", dpi=100); plt.show()
"""),
])

# ── 20.4 Adversarial Attacks ─────────────────────────────────────────────────
notebooks["Notebook_20_4_Adversarial_attacks"] = nb([
md("# Notebook 20.4 – Adversarial Attacks\nFGSM and PGD adversarial examples."),
code(SETUP),
code("""
import torch, torch.nn as nn, torch.optim as optim

torch.manual_seed(0)
D, H = 20, 64
X_np = np.random.randn(300, D).astype(np.float32)
y_np = (X_np[:,0] + 0.5*X_np[:,1] > 0).astype(np.int64)
X_t  = torch.tensor(X_np); y_t = torch.tensor(y_np)

model = nn.Sequential(nn.Linear(D,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,2))
opt   = optim.Adam(model.parameters(), 1e-3)
for _ in range(500):
    loss = nn.CrossEntropyLoss()(model(X_t), y_t)
    opt.zero_grad(); loss.backward(); opt.step()

def fgsm(model, X, y, eps=0.1):
    X_adv = X.clone().requires_grad_(True)
    loss  = nn.CrossEntropyLoss()(model(X_adv), y)
    loss.backward()
    return (X_adv + eps * X_adv.grad.sign()).detach()

def pgd(model, X, y, eps=0.1, alpha=0.01, iters=40):
    X_adv = X.clone()
    for _ in range(iters):
        X_adv = X_adv.clone().requires_grad_(True)
        loss  = nn.CrossEntropyLoss()(model(X_adv), y)
        loss.backward()
        X_adv = (X_adv + alpha * X_adv.grad.sign()).detach()
        X_adv = torch.max(torch.min(X_adv, X+eps), X-eps)
    return X_adv

model.eval()
X_fgsm = fgsm(model, X_t, y_t, eps=0.3)
X_pgd  = pgd(model,  X_t, y_t, eps=0.3)

def acc(X, y): return (model(X).argmax(1)==y).float().mean().item()
print(f"Clean acc: {acc(X_t,y_t):.3f}")
print(f"FGSM  acc: {acc(X_fgsm,y_t):.3f}")
print(f"PGD   acc: {acc(X_pgd, y_t):.3f}")

# Perturbation magnitude
for name, X_adv in [('FGSM',X_fgsm),('PGD',X_pgd)]:
    pert = (X_adv - X_t).abs().numpy()
    print(f"{name} max perturbation: {pert.max():.3f}, mean: {pert.mean():.4f}")
"""),
])

# ── 21.1 Bias Mitigation ─────────────────────────────────────────────────────
notebooks["Notebook_21_1_Bias_mitigation"] = nb([
md("# Notebook 21.1 – Bias Mitigation\nFairness metrics and debiasing strategies."),
code(SETUP),
code("""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

np.random.seed(0)
N = 1000
# Protected attribute: group A (0) and group B (1)
group = np.random.randint(0, 2, N)
# Feature correlated with group
X_feat = np.column_stack([
    np.random.randn(N) + group,        # biased feature
    np.random.randn(N)                  # unbiased feature
])
# True label: depends only on X_feat[:,1]
y_true = (X_feat[:,1] > 0).astype(int)

# Standard model (uses biased feature)
clf = LogisticRegression().fit(X_feat, y_true)
y_pred = clf.predict(X_feat)

# Fairness metrics
def demographic_parity(y_pred, group):
    return abs(y_pred[group==0].mean() - y_pred[group==1].mean())

def equal_opportunity(y_true, y_pred, group):
    tpr = lambda g: y_pred[(group==g)&(y_true==1)].mean()
    return abs(tpr(0) - tpr(1))

print("=== Standard Model ===")
print(f"Accuracy:            {accuracy_score(y_true, y_pred):.3f}")
print(f"Demographic Parity:  {demographic_parity(y_pred, group):.3f}")
print(f"Equal Opportunity:   {equal_opportunity(y_true, y_pred, group):.3f}")

# Fairness-aware: remove biased feature
clf_fair = LogisticRegression().fit(X_feat[:,1:], y_true)
y_pred_fair = clf_fair.predict(X_feat[:,1:])
print("\\n=== Debiased Model (feature removal) ===")
print(f"Accuracy:            {accuracy_score(y_true, y_pred_fair):.3f}")
print(f"Demographic Parity:  {demographic_parity(y_pred_fair, group):.3f}")
print(f"Equal Opportunity:   {equal_opportunity(y_true, y_pred_fair, group):.3f}")
"""),
md("## Visualise Fairness-Accuracy Trade-off"),
code("""
from sklearn.linear_model import LogisticRegression

dp_vals, acc_vals = [], []
for lam in np.linspace(0, 1, 20):
    # Mix of biased and fair features
    X_mix = np.column_stack([X_feat[:,0]*lam, X_feat[:,1]])
    clf_m = LogisticRegression().fit(X_mix, y_true)
    yp    = clf_m.predict(X_mix)
    dp_vals.append(demographic_parity(yp, group))
    acc_vals.append(accuracy_score(y_true, yp))

plt.figure(figsize=(7,4))
plt.plot(dp_vals, acc_vals, 'bo-')
plt.xlabel("Demographic Parity Violation"); plt.ylabel("Accuracy")
plt.title("Fairness vs Accuracy Trade-off"); plt.tight_layout()
plt.savefig("bias_mitigation.png", dpi=100); plt.show()
"""),
])

# ── 21.2 Explainability ──────────────────────────────────────────────────────
notebooks["Notebook_21_2_Explainability"] = nb([
md("# Notebook 21.2 – Explainability\nSHAP values, integrated gradients, and saliency maps."),
code(SETUP),
code("""
import torch, torch.nn as nn
import numpy as np

torch.manual_seed(0)
D, H = 10, 32
X_np = np.random.randn(200, D).astype(np.float32)
y_np = (X_np[:,0]*2 + X_np[:,2] - X_np[:,4] > 0).astype(np.int64)

X_t = torch.tensor(X_np); y_t = torch.tensor(y_np)
model = nn.Sequential(nn.Linear(D,H),nn.ReLU(),nn.Linear(H,H),nn.ReLU(),nn.Linear(H,2))
opt   = torch.optim.Adam(model.parameters(), 1e-3)
for _ in range(500):
    loss = nn.CrossEntropyLoss()(model(X_t), y_t)
    opt.zero_grad(); loss.backward(); opt.step()

# ── Gradient-based saliency
def saliency(model, x, target_class=1):
    x_in = x.unsqueeze(0).clone().requires_grad_(True)
    out  = model(x_in)[0, target_class]
    out.backward()
    return x_in.grad.squeeze().abs().detach().numpy()

sal = np.array([saliency(model, X_t[i]) for i in range(len(X_t))])
mean_sal = sal.mean(0)

# ── Integrated Gradients
def integrated_gradients(model, x, target_class=1, steps=50):
    baseline = torch.zeros_like(x)
    alphas   = torch.linspace(0, 1, steps)
    grads    = []
    for alpha in alphas:
        xi = (baseline + alpha * (x - baseline)).clone().requires_grad_(True)
        out = model(xi.unsqueeze(0))[0, target_class]
        out.backward()
        grads.append(xi.grad.squeeze().detach())
    ig = (x - baseline) * torch.stack(grads).mean(0)
    return ig.numpy()

ig = np.array([integrated_gradients(model, X_t[i]) for i in range(min(50, len(X_t)))])
mean_ig = ig.mean(0)

# ── SHAP-style permutation importance
def permutation_importance(model, X, y, n_perm=20):
    base_acc = (model(X).argmax(1)==y).float().mean().item()
    importances = []
    for feat in range(X.shape[1]):
        accs = []
        for _ in range(n_perm):
            X_perm = X.clone()
            X_perm[:,feat] = X_perm[torch.randperm(len(X)),feat]
            acc = (model(X_perm).argmax(1)==y).float().mean().item()
            accs.append(acc)
        importances.append(base_acc - np.mean(accs))
    return np.array(importances)

perm_imp = permutation_importance(model, X_t, y_t)

fig, axes = plt.subplots(1, 3, figsize=(14,4))
feature_names = [f"x{i}" for i in range(D)]
for ax, vals, title in zip(axes, [mean_sal, mean_ig, perm_imp],
                            ['Gradient Saliency', 'Integrated Gradients', 'Permutation Importance']):
    ax.barh(feature_names, vals)
    ax.set_title(title); ax.set_xlabel("Importance")
plt.suptitle("Explainability Methods"); plt.tight_layout()
plt.savefig("explainability.png", dpi=100); plt.show()
print("True important features: x0, x2, x4")
"""),
])

# ── Write all notebooks ───────────────────────────────────────────────────────
for fname, notebook in notebooks.items():
    path = os.path.join(OUT, f"{fname}.ipynb")
    with open(path, 'w') as f:
        nbf.write(notebook, f)

print(f"\n  Written {len(notebooks)} notebooks to {OUT}")
print("\nList:")
for i, name in enumerate(sorted(notebooks.keys()), 1):
    print(f"  {i:2d}. {name}.ipynb")
