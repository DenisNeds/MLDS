"""
Part 1: Apply KRR and SVR with polynomial and RBF kernels to the `sine` dataset.

Goal: tune kernel/regularization parameters by hand so each fit looks reasonable.
For SVR, also pick epsilon to keep the solution sparse (few support vectors).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from hw_kernels import KernelizedRidgeRegression, SVR, Polynomial, RBF


def main():
    df = pd.read_csv('sine.csv')
    X_raw = df[['x']].values
    y = df['y'].values

    # Standardize x. The polynomial kernel (x.x + 1)^M explodes on large x,
    # so we work in standardized coordinates. Predictions are made on a grid
    # in the standardized space and plotted against the original x.
    mu, sd = X_raw.mean(0), X_raw.std(0)
    X = (X_raw - mu) / sd

    x_grid_raw = np.linspace(X_raw.min() - 0.5, X_raw.max() + 0.5, 400)[:, None]
    x_grid = (x_grid_raw - mu) / sd

    # --- Hand-tuned settings ---------------------------------------------
    # In standardized x the data spans roughly [-1.7, 1.7] with about one
    # full sine cycle. Polynomial degree ~10-12 captures the wiggles; an
    # RBF with sigma ~0.2 (in standardized scale, i.e. ~1.2 in raw x) tracks
    # the local oscillation cleanly.
    settings = [
        ("KRR + Polynomial (M=12)", KernelizedRidgeRegression(Polynomial(M=12), lambda_=1e-5)),
        ("KRR + RBF (sigma=0.2)",   KernelizedRidgeRegression(RBF(sigma=0.2),   lambda_=1e-3)),
        ("SVR + Polynomial (M=12)", SVR(Polynomial(M=12), lambda_=1e-5, epsilon=0.4)),
        ("SVR + RBF (sigma=0.2)",   SVR(RBF(sigma=0.2),   lambda_=1e-3, epsilon=0.4)),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)

    for ax, (title, model) in zip(axes.flat, settings):
        model.fit(X, y)
        y_grid = model.predict(x_grid)

        ax.scatter(X_raw.ravel(), y, s=14, color='steelblue', alpha=0.55,
                   edgecolor='none', label='data')
        ax.plot(x_grid_raw.ravel(), y_grid, color='crimson', lw=2, label='fit')

        if isinstance(model, SVR):
            sv = model.support_
            ax.scatter(X_raw[sv].ravel(), y[sv], s=70, facecolor='none',
                       edgecolor='black', lw=1.4,
                       label=f'support vectors (n={sv.sum()})')
            # Draw the epsilon-tube around the fit
            ax.plot(x_grid_raw.ravel(), y_grid + model.epsilon, '--',
                    color='gray', lw=0.8, alpha=0.7)
            ax.plot(x_grid_raw.ravel(), y_grid - model.epsilon, '--',
                    color='gray', lw=0.8, alpha=0.7)
            train_mse = np.mean((model.predict(X) - y) ** 2)
            ax.set_title(f"{title}\nMSE={train_mse:.3f}, |SV|={sv.sum()}",
                         fontsize=10)
        else:
            train_mse = np.mean((model.predict(X) - y) ** 2)
            ax.set_title(f"{title}\nMSE={train_mse:.3f}", fontsize=10)

        ax.legend(loc='lower left', fontsize=8, frameon=True)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(alpha=0.3)

    fig.suptitle('Part 1: KRR and SVR fits on the sine dataset', fontsize=12)
    fig.tight_layout()
    fig.savefig('part1_sine.png', dpi=130, bbox_inches='tight')
    print("saved part1_sine.png")


if __name__ == '__main__':
    main()