"""
Part 2: KRR and SVR on housing2r.

For each (method, kernel) combination plot MSE vs a kernel hyperparameter:
  - Polynomial kernel: degree M = 1..10
  - RBF kernel:        sigma over a log-spaced range

Two curves per plot:
  - lambda fixed at 1
  - lambda chosen by 5-fold internal CV per kernel-parameter setting

For SVR the number of support vectors is also reported (annotated near each
marker), and we tune epsilon/lambda jointly in the CV variant to keep the
solution sparse while still fitting well.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hw_kernels import KernelizedRidgeRegression, SVR, Polynomial, RBF


# ---------------------------------------------------------------------------
# Data, train/test split, standardization
# ---------------------------------------------------------------------------

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

    # Standardize features using training statistics
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1.0
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd

    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# Internal CV for picking lambda (and, optionally, epsilon for SVR)
# ---------------------------------------------------------------------------

def kfold_indices(n, k, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    folds = np.array_split(idx, k)
    return folds


def cv_score(model_factory, X, y, folds):
    errs = []
    for i in range(len(folds)):
        val = folds[i]
        tr  = np.concatenate([folds[j] for j in range(len(folds)) if j != i])
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


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------

def sweep(kernel_name, kernel_param_values, kernel_ctor, Xtr, ytr, Xte, yte,
          lambdas, svr_epsilon, folds):
    """Run KRR and SVR sweeps over a list of kernel parameter values.

    Returns dict with arrays for plotting.
    """
    out = {
        'param': np.array(kernel_param_values, dtype=float),
        'krr_lam1':   [],   # MSE with lambda=1
        'krr_cv':     [],   # MSE with CV-tuned lambda
        'krr_cv_lam': [],
        'svr_lam1':       [],
        'svr_lam1_nsv':   [],
        'svr_cv':         [],
        'svr_cv_lam':     [],
        'svr_cv_nsv':     [],
    }
    for p in kernel_param_values:
        kf = lambda p=p: kernel_ctor(p)

        # --- KRR, lambda = 1 ---
        m = KernelizedRidgeRegression(kf(), lambda_=1.0).fit(Xtr, ytr)
        out['krr_lam1'].append(np.mean((m.predict(Xte) - yte) ** 2))

        # --- KRR, lambda by CV ---
        lam_cv = best_lambda_krr(kf, Xtr, ytr, lambdas, folds)
        m = KernelizedRidgeRegression(kf(), lambda_=lam_cv).fit(Xtr, ytr)
        out['krr_cv'].append(np.mean((m.predict(Xte) - yte) ** 2))
        out['krr_cv_lam'].append(lam_cv)

        # --- SVR, lambda = 1 ---
        m = SVR(kf(), lambda_=1.0, epsilon=svr_epsilon).fit(Xtr, ytr)
        out['svr_lam1'].append(np.mean((m.predict(Xte) - yte) ** 2))
        out['svr_lam1_nsv'].append(int(m.support_.sum()))

        # --- SVR, lambda by CV ---
        lam_cv = best_lambda_svr(kf, Xtr, ytr, lambdas, svr_epsilon, folds)
        m = SVR(kf(), lambda_=lam_cv, epsilon=svr_epsilon).fit(Xtr, ytr)
        out['svr_cv'].append(np.mean((m.predict(Xte) - yte) ** 2))
        out['svr_cv_lam'].append(lam_cv)
        out['svr_cv_nsv'].append(int(m.support_.sum()))

        print(f"  {kernel_name}={p}: "
              f"KRR(λ=1)={out['krr_lam1'][-1]:.2f}, "
              f"KRR(CV λ={lam_cv:.3g})={out['krr_cv'][-1]:.2f}, "
              f"SVR(λ=1)={out['svr_lam1'][-1]:.2f} [{out['svr_lam1_nsv'][-1]} SV], "
              f"SVR(CV λ={out['svr_cv_lam'][-1]:.3g})={out['svr_cv'][-1]:.2f} "
              f"[{out['svr_cv_nsv'][-1]} SV]")

    for k in list(out.keys()):
        if k != 'param':
            out[k] = np.array(out[k])
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

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

    # SV counts (lower panel)
    ax_sv.plot(p, res['svr_lam1_nsv'], 'o-',  color='C3', label='SVR, λ=1')
    ax_sv.plot(p, res['svr_cv_nsv'],   's--', color='C3', label='SVR, λ by CV', alpha=0.75)
    if log_x:
        ax_sv.set_xscale('log')
    ax_sv.set_xlabel(xlabel)
    ax_sv.set_ylabel('# support vectors')
    ax_sv.grid(alpha=0.3)
    ax_sv.legend(fontsize=9)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    Xtr, ytr, Xte, yte = load_housing(test_frac=0.2, seed=2)
    print(f"train: {Xtr.shape}, test: {Xte.shape}")
    print(f"y: train mean={ytr.mean():.2f}, std={ytr.std():.2f}; "
          f"test mean={yte.mean():.2f}, std={yte.std():.2f}")

    # CV grid
    folds = kfold_indices(len(Xtr), k=5, seed=1)
    lambdas = np.logspace(-3, 3, 13)

    # SVR epsilon: y has std ~8 on training data, so epsilon ~ 2 gives a
    # tube of ±2 around predictions — this is wide enough for 50%+ of points
    # to fall inside (and not become support vectors), yet narrow enough to
    # match KRR-quality predictions.
    svr_epsilon = 2.0

    print("\n=== Polynomial kernel sweep (M = 1..10) ===")
    M_values = list(range(1, 11))
    poly_res = sweep('M', M_values, lambda M: Polynomial(M=int(M)),
                     Xtr, ytr, Xte, yte, lambdas, svr_epsilon, folds)

    print("\n=== RBF kernel sweep ===")
    sigma_values = np.array([0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 12.0])
    rbf_res = sweep('sigma', sigma_values, lambda s: RBF(sigma=float(s)),
                    Xtr, ytr, Xte, yte, lambdas, svr_epsilon, folds)

    # ---- Plotting ----
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
    print("saved part2_housing2r.png")

    # Save numerical results for the writeup
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
    print("saved part2_results.csv")


if __name__ == '__main__':
    main()