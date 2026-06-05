import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers

# cvxopt nastavitve (brez vmesnih printov in nastavitev toleranc za resitve)
solvers.options['show_progress'] = False
for tol in ['abstol', 'reltol', 'feastol']:
    solvers.options[tol] = 1e-10

#ce hocmo passat teste mormo tud to pohendlat
def _to_2d(x):
    x_arr = np.asarray(x, dtype=float)
    return (x_arr[None, :], True) if x_arr.ndim == 1 else (x_arr, False)

def _squeeze(K, a_was_1d, b_was_1d):
    if a_was_1d and b_was_1d: 
        return float(K[0, 0])
    if a_was_1d: 
        return K[0]
    if b_was_1d: 
        return K[:, 0]
    return K



class Polynomial:
    def __init__(self, M=2):
        self.M = M

    def __call__(self, A, B):
        A2, a1 = _to_2d(A)
        B2, b1 = _to_2d(B)
        K = (A2 @ B2.T + 1.0) ** self.M
        return _squeeze(K, a1, b1)


class RBF:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def __call__(self, A, B):
        A2, a1 = _to_2d(A)
        B2, b1 = _to_2d(B)
        
        distsq = np.sum(A2**2, axis=1)[:, None] + np.sum(B2**2, axis=1)[None, :] - 2.0 * (A2 @ B2.T) #absolutna razlika na kvadrat
        np.maximum(distsq, 0.0, out=distsq)
        K = np.exp(-distsq / (2.0 * self.sigma**2))
        
        return _squeeze(K, a1, b1)


class KernelizedRidgeRegression:
    def __init__(self, kernel, lambda_=1.0):
        self.kernel = kernel
        self.lambda_ = lambda_

    def fit(self, X, y):
        self.X_ = X
        K = self.kernel(X, X)
        # resujem (K + lambda * I) * alpha = y
        self.alpha_ = np.linalg.solve(K + self.lambda_ * np.eye(len(X)), y)
        return self

    def predict(self, X):
        return self.kernel(X, self.X_) @ self.alpha_


class SVR:
    def __init__(self, kernel, lambda_=1.0, epsilon=0.1):
        self.kernel = kernel
        self.lambda_ = lambda_
        self.epsilon = epsilon

    def fit(self, X, y):
        self.X_ = X
        n = len(X)
        C = 1.0 / self.lambda_

        K = self.kernel(X, X)
        K = 0.5 * (K + K.T)
        #mora bit simetricna matrika
        
        #kvadraticno programiranje formulacija problema
        #q.tPq + 
        P = np.kron(K, np.array([[1.0, -1.0], [-1.0, 1.0]])) + 1e-10 * np.eye(2 * n)
        
        q = np.empty(2 * n)
        q[0::2] = -y + self.epsilon #lihi za alfe
        q[1::2] =  y + self.epsilon #sodi za alfe*
        
        #vezi
        G = np.vstack([-np.eye(2 * n), np.eye(2 * n)])
        h = np.concatenate([np.zeros(2 * n), C * np.ones(2 * n)])
        A = np.tile([1.0, -1.0], n)[None, :]
        b = np.zeros(1)

        sol = solvers.qp(matrix(P), matrix(q), matrix(G), matrix(h), matrix(A), matrix(b))

        #rezultat so alfe [alfa1, alfa1*, alfa2, alfa2* ,....]
        self.alpha_full_ = np.array(sol['x']).ravel().reshape(n, 2)
        self.beta_ = self.alpha_full_[:, 0] - self.alpha_full_[:, 1]
        self.b_ = float(sol['y'][0])
        self.support_ = (self.alpha_full_[:, 0] > 1e-5 * C) | (self.alpha_full_[:, 1] > 1e-5 * C)
        
        return self

    def predict(self, X):
        return self.kernel(X, self.X_) @ self.beta_ + self.b_

    def get_alpha(self):
        return self.alpha_full_

    def get_b(self):
        return self.b_


def run_sine():
    df = pd.read_csv('sine.csv')
    X_raw, y = df[['x']].values, df['y'].values
    X = (X_raw - X_raw.mean()) / X_raw.std()
    #standardizacija
    x_plot_raw = np.linspace(X_raw.min() - 0.5, X_raw.max() + 0.5, 400)[:, None]
    x_plot = (x_plot_raw - X_raw.mean()) / X_raw.std()
    #ročno tweakanje in pregled najbolših prileganje
    models = [
        ("KRR + Polynomial (M=12)", KernelizedRidgeRegression(Polynomial(M=12), lambda_=1e-5)),
        ("KRR + RBF (sigma=0.2)",   KernelizedRidgeRegression(RBF(sigma=0.2), lambda_=1e-3)),
        ("SVR + Polynomial (M=12)", SVR(Polynomial(M=12), lambda_=1e-5, epsilon=0.4)),
        ("SVR + RBF (sigma=0.2)",   SVR(RBF(sigma=0.2), lambda_=1e-3, epsilon=0.4))
    ]
    #plotanje
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (title, model) in zip(axes.flat, models):
        model.fit(X, y)
        preds = model.predict(x_plot)
        
        ax.scatter(X_raw, y, alpha=0.5, label='Data')
        ax.plot(x_plot_raw, preds, color='red', label='Fit')
        
        if isinstance(model, SVR):
            ax.scatter(X_raw[model.support_], y[model.support_], 
                       facecolors='none', edgecolors='k', s=80, label='SVs')
            
        ax.set_title(title)
        ax.legend()
        
    plt.tight_layout()
    plt.savefig('part1_sine.png')


def load_housing(test_frac=0.2, seed=2):
    df = pd.read_csv('housing2r.csv')
    X = df.iloc[:, :-1].values.astype(float)
    y = df.iloc[:, -1].values.astype(float)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_test = int(test_frac * len(X))
    te, tr = idx[:n_test], idx[n_test:]

    Xtr, Xte = X[tr], X[te]
    ytr, yte = y[tr], y[te]

    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    return Xtr, ytr, Xte, yte


def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


def cv_score(model_factory, X, y, folds):
    errs = []
    for i in range(len(folds)):
        val = folds[i]
        tr = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
        m = model_factory().fit(X[tr], y[tr])
        pred = m.predict(X[val])
        errs.append(np.mean((pred - y[val]) ** 2))
    return float(np.mean(errs))


def best_lambda_krr(kernel_factory, X, y, lambdas, folds):
    best, best_score = None, np.inf
    for lam in lambdas:
        score = cv_score(lambda lam=lam: KernelizedRidgeRegression(kernel_factory(), lambda_=lam),
                         X, y, folds)
        if score < best_score:
            best_score, best = score, lam
    return best


def best_lambda_svr(kernel_factory, X, y, lambdas, epsilon, folds):
    best, best_score = None, np.inf
    for lam in lambdas:
        score = cv_score(lambda lam=lam: SVR(kernel_factory(), lambda_=lam, epsilon=epsilon),
                         X, y, folds)
        if score < best_score:
            best_score, best = score, lam
    return best


def sweep(kernel_name, kernel_param_values, kernel_ctor, Xtr, ytr, Xte, yte,
          lambdas, svr_epsilon, folds):
    out = {
        'param': np.array(kernel_param_values, dtype=float),
        'krr_lam1':   [],
        'krr_cv':     [],
        'krr_cv_lam': [],
        'svr_lam1':       [],
        'svr_lam1_nsv':   [],
        'svr_cv':         [],
        'svr_cv_lam':     [],
        'svr_cv_nsv':     [],
    }
    for p in kernel_param_values:
        kf = lambda p=p: kernel_ctor(p)

        m = KernelizedRidgeRegression(kf(), lambda_=1.0).fit(Xtr, ytr)
        out['krr_lam1'].append(np.mean((m.predict(Xte) - yte) ** 2))

        lam_cv = best_lambda_krr(kf, Xtr, ytr, lambdas, folds)
        m = KernelizedRidgeRegression(kf(), lambda_=lam_cv).fit(Xtr, ytr)
        out['krr_cv'].append(np.mean((m.predict(Xte) - yte) ** 2))
        out['krr_cv_lam'].append(lam_cv)

        m = SVR(kf(), lambda_=1.0, epsilon=svr_epsilon).fit(Xtr, ytr)
        out['svr_lam1'].append(np.mean((m.predict(Xte) - yte) ** 2))
        out['svr_lam1_nsv'].append(int(m.support_.sum()))

        lam_cv = best_lambda_svr(kf, Xtr, ytr, lambdas, svr_epsilon, folds)
        m = SVR(kf(), lambda_=lam_cv, epsilon=svr_epsilon).fit(Xtr, ytr)
        out['svr_cv'].append(np.mean((m.predict(Xte) - yte) ** 2))
        out['svr_cv_lam'].append(lam_cv)
        out['svr_cv_nsv'].append(int(m.support_.sum()))


    for k in list(out.keys()):
        if k != 'param':
            out[k] = np.array(out[k])
    return out


def plot_panel(ax_mse, ax_sv, res, xlabel, title, log_x=False, log_y=False):
    p = res['param']

    ax_mse.plot(p, res['krr_lam1'], 'o-',  color='C0', label='KRR, λ=1')
    ax_mse.plot(p, res['krr_cv'],   's--', color='C0', label='KRR, λ by CV', alpha=0.75)
    ax_mse.plot(p, res['svr_lam1'], 'o-',  color='C3', label='SVR, λ=1')
    ax_mse.plot(p, res['svr_cv'],   's--', color='C3', label='SVR, λ by CV', alpha=0.75)

    if log_x:
        ax_mse.set_xscale('log')
    if log_y:
        ax_mse.set_yscale('log')
    ax_mse.set_xlabel(xlabel)
    ax_mse.set_ylabel('Test MSE' + (' (log)' if log_y else ''))
    ax_mse.set_title(title)
    ax_mse.grid(alpha=0.3, which='both')
    ax_mse.legend(fontsize=9)

    ax_sv.plot(p, res['svr_lam1_nsv'], 'o-',  color='C3', label='SVR, λ=1')
    ax_sv.plot(p, res['svr_cv_nsv'],   's--', color='C3', label='SVR, λ by CV', alpha=0.75)
    if log_x:
        ax_sv.set_xscale('log')
    ax_sv.set_xlabel(xlabel)
    ax_sv.set_ylabel('# support vectors')
    ax_sv.grid(alpha=0.3)
    ax_sv.legend(fontsize=9)


def run_housing():
    Xtr, ytr, Xte, yte = load_housing(test_frac=0.2, seed=2)
    folds = kfold_indices(len(Xtr), k=5, seed=1)
    lambdas = np.logspace(-3, 3, 13)
    svr_epsilon = 2.0

    
    M_values = list(range(1, 11))
    poly_res = sweep('M', M_values, lambda M: Polynomial(M=int(M)),
                     Xtr, ytr, Xte, yte, lambdas, svr_epsilon, folds)

    
    sigma_values = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    rbf_res = sweep('sigma', sigma_values, lambda s: RBF(sigma=float(s)),
                    Xtr, ytr, Xte, yte, lambdas, svr_epsilon, folds)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    plot_panel(axes[0, 0], axes[1, 0], poly_res,
               xlabel='Polynomial degree M',
               title='Polynomial kernel: MSE on test',
               log_x=False, log_y=True)
    plot_panel(axes[0, 1], axes[1, 1], rbf_res,
               xlabel='RBF σ',
               title='RBF kernel: MSE on test',
               log_x=True, log_y=False)
    fig.suptitle('Part 2: housing2r — KRR vs SVR with polynomial and RBF kernels',
                 fontsize=12)
    fig.tight_layout()
    fig.savefig('part2_housing2r.png', dpi=130, bbox_inches='tight')

    rows = []
    for label, res, pname in [('Polynomial', poly_res, 'M'),
                              ('RBF', rbf_res, 'sigma')]:
        for i, p in enumerate(res['param']):
            rows.append({
                'kernel': label, pname: p,
                'KRR_lam1_MSE': res['krr_lam1'][i],
                'KRR_CV_lambda': res['krr_cv_lam'][i],
                'KRR_CV_MSE': res['krr_cv'][i],
                'SVR_lam1_MSE': res['svr_lam1'][i],
                'SVR_lam1_nSV': res['svr_lam1_nsv'][i],
                'SVR_CV_lambda': res['svr_cv_lam'][i],
                'SVR_CV_MSE': res['svr_cv'][i],
                'SVR_CV_nSV': res['svr_cv_nsv'][i],
            })
    pd.DataFrame(rows).to_csv('part2_results.csv', index=False)
    

if __name__ == '__main__':
    run_sine()
    run_housing()