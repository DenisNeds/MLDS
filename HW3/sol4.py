import pandas as pd
import numpy as np
from scipy.stats import norm
from solution2 import MultinomialLogReg

def main():
    
    df = pd.read_csv('dataset.csv', sep=';')
    unique_shots = df['ShotType'].unique().tolist()
    target_baseline = 'above head'
    if target_baseline in unique_shots:
        unique_shots.remove(target_baseline)
        unique_shots.append(target_baseline)

    shot_mapping = {shot: idx for idx, shot in enumerate(unique_shots)}
    reverse_shot_mapping = {idx: shot for shot, idx in shot_mapping.items()}
    y = df['ShotType'].map(shot_mapping).values
    
    ref_class_idx = max(shot_mapping.values())
    ref_class_name = reverse_shot_mapping[ref_class_idx]
    
    X_df = df.drop('ShotType', axis=1)
    X_encoded = pd.get_dummies(X_df, drop_first=True)
    
    std_devs = {}
    for col in ['Angle', 'Distance']:
        if col in X_encoded.columns:
            mean_val = X_encoded[col].mean()
            std_val = X_encoded[col].std()
            std_devs[col] = std_val
            X_encoded[col] = (X_encoded[col] - mean_val) / std_val

    X = X_encoded.astype(float).values
    feature_names = X_encoded.columns.tolist()


    model = MultinomialLogReg()
    model.build(X, y)
    
    variances = np.diag(model.hess_inv)
    se = np.sqrt(np.maximum(variances, 0)).reshape((len(feature_names), model.num_classes - 1))
    
    weights = model.weights
    z_scores = weights / np.where(se == 0, 1e-15, se)
    p_values = 2 * (1 - norm.cdf(np.abs(z_scores)))
    
    odds_ratios = np.exp(weights)
    
    significant_findings = []

    for j in range(model.num_classes - 1):
        target_name = reverse_shot_mapping[j]
        print(f"\n➤ PREDICTING: {target_name.upper()} (vs {ref_class_name})")
        print("-" * 80)
        print(f"{'Feature':<20} | {'Coef (β)':<10} | {'Std Err':<10} | {'z':<8} | {'P>|z|':<8} | {'Odds Ratio':<10}")
        print("-" * 80)
        for k, feature in enumerate(feature_names):
            w, s, z, p, or_val = weights[k, j], se[k, j], z_scores[k, j], p_values[k, j], odds_ratios[k, j]
            stars = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            
            print(f"{feature:<20} | {w:>9.4f} | {s:>9.4f} | {z:>8.2f} | {p:>8.4f}{stars:<3} | {or_val:>9.4f}")
            
            if p < 0.01:
                significant_findings.append({
                    'target': target_name,
                    'feature': feature,
                    'weight': w,
                    'p_val': p,
                    'odds_ratio': or_val
                })
    print("\n" + "="*80)

if __name__ == "__main__":
    main()