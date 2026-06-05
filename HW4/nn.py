import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import argparse
import warnings

#aktivacijske funkcije in softmax

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500 ,500)))

def odvod_sigmoid(z):
    s = sigmoid(z)
    return s * (1.0 - s)

def relu(z):
    return np.maximum(z, 0)

def odvod_relu(z):
    return (z > 0).astype(float)

def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z/(np.sum(exp_z, axis=1, keepdims=True))

ACTIVATIONS = {
    "sigmoid": (sigmoid, odvod_sigmoid),
    "relu": (relu, odvod_relu)
}

#Custom neural network base

class BaseANN:
    def __init__(self, layers=None, hidden_activations=None, lambda_=0.0,
                 learning_rate=0.1, epochs=5000, batch_size=32,
                 momentum=0.9, seed=None):
        self.layers = layers if layers is not None else [16]
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.seed = seed
 
        if hidden_activations is None:
            self.hidden_activations = ["sigmoid"] * len(self.layers)
        else:
            assert len(hidden_activations) == len(self.layers)
            self.hidden_activations = hidden_activations
 
    def init_params(self, n_in, n_out):
        rng = np.random.default_rng(self.seed)
        layer_sizes = [n_in] + list(self.layers) + [n_out]
        self.W, self.b = [], []

        for i in range(len(layer_sizes) - 1):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            lim = np.sqrt(6 / (fan_in + fan_out))
            self.W.append(rng.uniform(-lim, lim, (fan_in, fan_out)))
            self.b.append(np.zeros((1, fan_out)))
 
    def forward(self, X):
        activations = [X]
        zs = []
        a = X
        n_layers = len(self.W)

        for l, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w + b
            zs.append(z)
            if l < n_layers - 1:
                a = ACTIVATIONS[self.hidden_activations[l]][0](z)
            else:
                a = self.output_activation(z)
            activations.append(a)

        return zs, activations
 
    def backprop(self, X, y_targ):
        N = X.shape[0]
        zs, activations = self.forward(X)
        grad_W = [None] * len(self.W)
        grad_b = [None] * len(self.b)
        delta = activations[-1] - y_targ

        for i in reversed(range(len(self.W))):
            a_prev = activations[i]
            grad_W[i] = (a_prev.T @ delta) /N
            grad_b[i] = np.sum(delta, axis=0, keepdims=True) /N
            if self.lambda_ > 0:
                grad_W[i] += self.lambda_ * self.W[i] /N
            if i > 0:
                deriv = ACTIVATIONS[self.hidden_activations[i - 1]][1]
                delta = (delta @ self.W[i].T) * deriv(zs[i - 1])

        return grad_W, grad_b
 
    def fit(self, x, y_targ, track_loss=False):
        x = np.array(x, dtype=float)
        N = x.shape[0]
        y_formatted = self.format_target(y_targ, N)
        self.init_params(x.shape[1], y_formatted.shape[1])
        vel_W = [np.zeros_like(w) for w in self.W]
        vel_b = [np.zeros_like(b) for b in self.b]
        rng = np.random.default_rng(self.seed)
        loss_history = []

        for epoch in range(self.epochs):
            perm = rng.permutation(N)
            X_s, y_s = x[perm], y_formatted[perm]
            for start in range(0, N, self.batch_size):
                X_b = X_s[start:start + self.batch_size]
                y_b = y_s[start:start + self.batch_size]
                gW, gb = self.backprop(X_b, y_b)
                for i in range(len(self.W)):
                    vel_W[i] = self.momentum * vel_W[i] - self.learning_rate * gW[i]
                    vel_b[i] = self.momentum * vel_b[i] - self.learning_rate * gb[i]
                    self.W[i] += vel_W[i]
                    self.b[i] += vel_b[i]
            if track_loss:
                loss_history.append(self.cost(x, y_formatted))

        self.loss_history_ = loss_history
        return self
 
    def cost(self, X, Y_onehot):
        raise NotImplementedError

    def weights(self):
        return [np.vstack([bias, w]) for w, bias in zip(self.W, self.b)]

    def output_activation(self, z):
        raise NotImplementedError

    def format_target(self, y, N):
        raise NotImplementedError    

#klasifikacija

class ANNClassification(BaseANN):
    def output_activation(self, z):
        return softmax(z)
    
    def format_target(self, y, N):
        y = np.array(y)
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        label_to_idx = {c: i for i, c in enumerate(self.classes)}
        Y_onehot = np.zeros((N, n_classes))
        for i, label in enumerate(y):
            Y_onehot[i, label_to_idx[label]] = 1.0
        return Y_onehot
    
    def predict(self, X):
        X = np.array(X, dtype=float)
        _, activations = self.forward(X)
        probs = activations[-1]
        return probs
    def predict_class(self, X):
        X = np.array(X, dtype=float)
        _, activations = self.forward(X)
        probs = activations[-1]
        idx = np.argmax(probs, axis=1)
        return self.classes[idx]
    
    def numerical_grad(self, X, Y_onehot, eps= 1e-5):

        num_grad_W = [np.zeros_like(W) for W in self.W]
        num_grad_b = [np.zeros_like(b) for b in self.b]

        for i in range(len(self.W)):
            it = np.nditer(self.W[i], flags=["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = self.W[i][idx]
                self.W[i][idx] = orig + eps
                c_plus = self.cost(X, Y_onehot)
                self.W[i][idx] = orig - eps
                c_minus = self.cost(X, Y_onehot)
                self.W[i][idx] = orig
                num_grad_W[i][idx] = (c_plus - c_minus)/(2*eps)
                it.iternext()

            it = np.nditer(self.b[i], flags = ["multi_index"])
            while not it.finished:
                idx = it.multi_index
                orig = self.b[i][idx]
                self.b[i][idx] = orig + eps
                c_plus = self.cost(X, Y_onehot)
                self.b[i][idx] = orig - eps
                c_minus = self.cost(X, Y_onehot)
                self.b[i][idx] = orig 
                num_grad_b[i][idx] = (c_plus - c_minus)/(2*eps)
                it.iternext()


        return num_grad_W, num_grad_b 
    
    def cost(self, X, Y_onehot):
        _, activations = self.forward(X)
        probabilities = activations[-1]
        N = X.shape[0]
        log_probs = np.log(np.clip(probabilities, 1e-15, 1))
        c = - np.sum(Y_onehot * log_probs) / N 
        return c

#regresija    
class ANNRegression(BaseANN):
    def output_activation(self, z):
        return z

    def format_target(self, y, N):
        y = np.array(y, dtype = float)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return y

    def predict(self, X):
        X = np.array(X, dtype=float)
        _, activations = self.forward(X)
        res = activations[-1]
        return res.ravel() if res.shape[1] == 1 else res  


class ANNClassificationTorch(nn.Module):
        def __init__(self, n_features, n_classes, layers=None, seed=None):
            super().__init__()
            if layers is None:
                layers = [16]
            if seed is not None:
                torch.manual_seed(seed)
            self.network = nn.Sequential()
            sizes = [n_features] + layers + [n_classes]
            for i in range(len(sizes) - 1):
                linear = nn.Linear(sizes[i], sizes[i + 1])
                lim = np.sqrt(6.0 / (sizes[i] + sizes[i + 1]))
                nn.init.uniform_(linear.weight, -lim, lim)
                nn.init.zeros_(linear.bias)
                self.network.add_module(f"fc_{i}", linear)
                if i < len(sizes) - 2:
                    self.network.add_module(f"sigmoid_{i}", nn.Sigmoid())
 
        def forward(self, x):
            return self.network(x)
 
class PytorchTrainer:
    def __init__(self, model, learning_rate=0.1, epochs=1000,
                    batch_size=32, momentum=0.9):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.momentum = momentum
        self.optimizer = optim.SGD(model.parameters(),
                                    lr=learning_rate, momentum=momentum)
        self.criterion = nn.CrossEntropyLoss()

    def fit(self, X, y, track_loss=False):
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(Xt, yt)
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True)
        loss_history = []
        for _ in range(self.epochs):
            for bx, by in loader:
                self.optimizer.zero_grad()
                out  = self.model(bx)
                loss = self.criterion(out, by)
                loss.backward()
                self.optimizer.step()
            if track_loss:
                self.model.eval()
                with torch.no_grad():
                    full_loss = self.criterion(self.model(Xt), yt).item()
                loss_history.append(full_loss)
                self.model.train()
        self.loss_history_ = loss_history

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            out = self.model(Xt)
            return torch.argmax(out, dim=1).numpy()

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32)
            out = self.model(Xt)
            return torch.softmax(out, dim=1).numpy()

def gradient_check(model, X, Y_onehot, eps=1e-5):
    an_W, an_b = model.backprop(X, Y_onehot)
    nu_W, nu_b = model.numerical_grad(X, Y_onehot, eps)
    max_err = 0.0
    for i in range(len(model.W)):
        for ag, ng in [(an_W[i], nu_W[i]), (an_b[i], nu_b[i])]:
            diff  = np.abs(ag - ng)
            denom = np.maximum(np.abs(ag) + np.abs(ng), 1e-10)
            max_err = max(max_err, (diff / denom).max())
    return max_err


def load_tab(path):
    data = []
    with open(path) as f:
        f.readline()                      
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                data.append(parts)
    data = np.array(data)
    X = data[:, 1:].astype(float)
    y_raw = data[:, 0]
    classes, y = np.unique(y_raw, return_inverse=True)
    return X, y, classes

def log_loss(y_true, probs, n_classes, eps=1e-15):
    N = len(y_true)
    Y = np.zeros((N, n_classes))
    for i, c in enumerate(y_true):
        Y[i, int(c)] = 1.0
    return -np.sum(Y * np.log(np.clip(probs, eps, 1))) / N

def train_test_split(X, y, test_size=0.2, seed=67):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    split = int(len(y) * (1 - test_size))
    return X[idx[:split]], X[idx[split:]], y[idx[:split]], y[idx[split:]]

def section(title):
    print("\n" + "=" * 50)
    print(f"  {title}")
    print("=" * 50)


def run_gradient_check():
    section("GRADIENT CHECK  (custom ANN)")
    rng = np.random.default_rng(0)
    X_gc  = rng.standard_normal((12, 4))
    y_gc  = rng.integers(0, 3, size=12)
    n_cls = 3
    Y_gc  = np.zeros((12, n_cls))
    for i, c in enumerate(y_gc):
        Y_gc[i, c] = 1.0
 
    model = ANNClassification(layers=[5, 4], learning_rate=0.1, epochs=1, seed=42)
    model.classes_ = np.arange(n_cls)
    model.init_params(4, n_cls)
    model.hidden_activations = ["sigmoid", "sigmoid"]
 
    err = gradient_check(model, X_gc, Y_gc)
    status = " PASS" if err < 1e-5 else " FAIL"
    print(f"  Max relative gradient error : {err:.2e}   {status}")
    print(f"  (threshold 1e-5; typical correct value < 1e-7)")

ARCHITECTURES = [
    [3],
    [4], [6], [8],
    [3, 3],
    [4, 4], [6, 6],
    [4, 4, 4],
]
LRS   = [0.1, 0.3]
SEEDS = list(range(5))

def grid_search_custom(X, y, n_classes, epochs, batch_size, label=""):
    print(f"\n  {'Arch':<12} {'LR':<5}  ", end="")
    print("  ".join(f"s{s}" for s in SEEDS), "  mean    best")
    print("  " + "-" * 65)
    best = {"acc": 0, "layers": None, "lr": None}
    for layers in ARCHITECTURES:
        for lr in LRS:
            accs = []
            for seed in SEEDS:
                m = ANNClassification(layers=layers, learning_rate=lr,
                                      epochs=epochs, batch_size=batch_size,
                                      momentum=0.9, seed=seed)
                m.fit(X, y)
                accs.append(np.mean(m.predict_class(X) == y))
            mean_acc = np.mean(accs)
            if mean_acc > best["acc"]:
                best = {"acc": mean_acc, "layers": layers, "lr": lr}
            row = "  ".join(f"{a*100:5.1f}" for a in accs)
            print(f"  {str(layers):<12} {lr:<5}  {row}  {mean_acc*100:5.1f}  "
                  f"{'★' if mean_acc == 1.0 else ''}")
    print(f"\n  Best: layers={best['layers']}  lr={best['lr']}  "
          f"mean_acc={best['acc']*100:.1f}%")
    return best

def grid_search_torch(X, y, n_classes, epochs, batch_size, label=""):
    print(f"\n  {'Arch':<12} {'LR':<5}  ", end="")
    print("  ".join(f"s{s}" for s in SEEDS), "  mean    best")
    print("  " + "-" * 65)
    best = {"acc": 0, "layers": None, "lr": None}
    for layers in ARCHITECTURES:
        for lr in LRS:
            accs = []
            for seed in SEEDS:
                net     = ANNClassificationTorch(X.shape[1], n_classes, layers, seed)
                trainer = PytorchTrainer(net, lr, epochs, batch_size, 0.9)
                trainer.fit(X, y)
                preds = trainer.predict(X)
                accs.append(np.mean(preds == y))
            mean_acc = np.mean(accs)
            if mean_acc > best["acc"]:
                best = {"acc": mean_acc, "layers": layers, "lr": lr}
            row = "  ".join(f"{a*100:5.1f}" for a in accs)
            print(f"  {str(layers):<12} {lr:<5}  {row}  {mean_acc*100:5.1f}  "
                  f"{'★' if mean_acc == 1.0 else ''}")
    print(f"\n  Best: layers={best['layers']}  lr={best['lr']}  "
          f"mean_acc={best['acc']*100:.1f}%")
    return best

def evaluate(X, y, n_classes, layers, lr, epochs, batch_size, seed=0):
    results = {}
 
    # custom
    t0 = time.time()
    m  = ANNClassification(layers=layers, learning_rate=lr, epochs=epochs,
                           batch_size=batch_size, momentum=0.9, seed=seed)
    m.fit(X, y)
    elapsed_custom = time.time() - t0
 
    acc  = np.mean(m.predict_class(X) == y)
    loss = log_loss(y, m.predict(X), n_classes)
    results["custom"] = dict(acc=acc, loss=loss, time=elapsed_custom)
 
    # pytorch
    t0  = time.time()
    net = ANNClassificationTorch(X.shape[1], n_classes, layers, seed)
    trainer = PytorchTrainer(net, lr, epochs, batch_size, 0.9)
    trainer.fit(X, y)
    elapsed_torch = time.time() - t0

    acc_t  = np.mean(trainer.predict(X) == y)
    loss_t = log_loss(y, trainer.predict_proba(X), n_classes)
    results["torch"] = dict(acc=acc_t, loss=loss_t, time=elapsed_torch)
    
    return results

def print_eval_table(results):
    print(f"\n  ┌{'─'*12}┬{'─'*10}┬{'─'*12}┬{'─'*9}┐")
    print(f"  │{'Model':^12}│{'Accuracy':^10}│{'Log-Loss':^12}│{'Time(s)':^9}│")
    print(f"  ├{'─'*12}┼{'─'*10}┼{'─'*12}┼{'─'*9}┤")
    for name, r in results.items():
        print(f"  │{name:^12}│{r['acc']*100:^9.2f}%│{r['loss']:^12.4f}│{r['time']:^9.2f}│")
    print(f"  └{'─'*12}┴{'─'*10}┴{'─'*12}┴{'─'*9}┘")


def convergence_comparison(X, y, n_classes, layers, lr, epochs, seed=0,
                           dataset_name=""):
    section(f"WEIGHT CONVERGENCE — {dataset_name}")
    print(f"  Architecture: {layers}  |  LR: {lr}  |  Epochs: {epochs}\n")
 
    bs_full = len(X)
    bs_mini = 32
    histories = {}
 
    for act in ["sigmoid", "relu"]:
        act_list = [act] * len(layers)
        for label, bs in [(f"{act}_full", bs_full), (f"{act}_mini", bs_mini)]:
            m = ANNClassification(layers=layers, learning_rate=lr, epochs=epochs,
                                  batch_size=bs, momentum=0.9, seed=seed,
                                  hidden_activations=act_list)
            m.fit(X, y, track_loss=True)
            histories[label] = m.loss_history_
 
    checkpoints = np.linspace(0, epochs - 1, 10, dtype=int)
    w = 14
    keys   = ["sigmoid_full", "sigmoid_mini", "relu_full", "relu_mini"]
    labels = ["Sigmoid-Full", "Sigmoid-Mini", "ReLU-Full",  "ReLU-Mini"]
 
    print(f"  {'Epoch':>7}  " + "  ".join(f"{l:>{w}}" for l in labels))
    print(f"  {'-'*7}  " + "  ".join([f"{'-'*w}"] * 4))
    for ep in checkpoints:
        vals = [histories[k][ep] for k in keys]
        print(f"  {ep+1:>7}  " + "  ".join(f"{v:>{w}.6f}" for v in vals))
 
    print("\n  Final losses:")
    for k, l in zip(keys, labels):
        print(f"    {l:<14}: {histories[k][-1]:.6f}")

 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=".",
                        help="Directory containing doughnut.tab and squares.tab")
    parser.add_argument("--epochs",     type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=0,
                        help="0 = full batch")
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()
 
    import os
    doughnut_path = os.path.join(args.data_dir, "doughnut.tab")
    squares_path  = os.path.join(args.data_dir, "squares.tab")
 
    # ── load ──────────────────────────────────────────────────────────────────
    X_d, y_d, cls_d = load_tab(doughnut_path)
    X_s, y_s, cls_s = load_tab(squares_path)
    n_cls_d, n_cls_s = len(cls_d), len(cls_s)
 
    bs_d = args.batch_size if args.batch_size > 0 else len(X_d)
    bs_s = args.batch_size if args.batch_size > 0 else len(X_s)
 
    print(f"\n  Loaded doughnut : {X_d.shape}  classes={cls_d}")
    print(f"  Loaded squares  : {X_s.shape}  classes={cls_s}")
 
    # ── gradient check ────────────────────────────────────────────────────────
    run_gradient_check()
 
    """# ── grid search — doughnut ────────────────────────────────────────────────
    section("GRID SEARCH — DOUGHNUT  (Custom ANN)")
    best_d_custom = grid_search_custom(X_d, y_d, n_cls_d,
                                       args.epochs, bs_d, "doughnut")
 
    
    section("GRID SEARCH — DOUGHNUT  (PyTorch ANN)")
    best_d_torch = grid_search_torch(X_d, y_d, n_cls_d,
                                        args.epochs, bs_d, "doughnut")
 
    # ── grid search — squares ─────────────────────────────────────────────────
    section("GRID SEARCH — SQUARES  (Custom ANN)")
    best_s_custom = grid_search_custom(X_s, y_s, n_cls_s,
                                       args.epochs, bs_s, "squares")
 
    
    section("GRID SEARCH — SQUARES  (PyTorch ANN)")
    best_s_torch = grid_search_torch(X_s, y_s, n_cls_s,
                                        args.epochs, bs_s, "squares")"""
 
    layers_d = [3,3,6,7,9]
    lr_d     = 0.3
    layers_s = [3,3,6,7,9]
    lr_s     = 0.3
 
    print(f"\n  --- DOUGHNUT  (arch={layers_d}, lr={lr_d}) ---")
    res_d = evaluate(X_d, y_d, n_cls_d, layers_d, lr_d,
                     args.epochs, bs_d, args.seed)
    print_eval_table(res_d)
 
    print(f"\n  --- SQUARES  (arch={layers_s}, lr={lr_s}) ---")
    res_s = evaluate(X_s, y_s, n_cls_s, layers_s, lr_s,
                     args.epochs, bs_s, args.seed)
    print_eval_table(res_s)
 
 
    # convergence 
    convergence_comparison(X_d, y_d, n_cls_d, layers_d, lr_d,
                           args.epochs, args.seed, "DOUGHNUT")
    convergence_comparison(X_s, y_s, n_cls_s, layers_s, lr_s,
                           args.epochs, args.seed, "SQUARES")
 

 
 
if __name__ == "__main__":
    main()
 




