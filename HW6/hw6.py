import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('toydataset.csv')
X = df[['x1', 'x2', 'x3']]
y = df['y']

X_encoded = pd.get_dummies(X, columns=['x3'])

gbm_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
gbm_model.fit(X_encoded, y)

def bb_predict(raw_samples_df):

    encoded_samples = pd.get_dummies(raw_samples_df, columns=['x3'])

    for col in X_encoded.columns:
        if col not in encoded_samples.columns:
            encoded_samples[col] = 0

    encoded_samples = encoded_samples[X_encoded.columns]
    return gbm_model.predict(encoded_samples)



class CustomLIME:
    def __init__(self, training_data, categorical_features, continuous_features):
        self.training_data = training_data.copy()
        self.categorical_features = categorical_features
        self.continuous_features = continuous_features
        
       
        self.scaler = StandardScaler()
        self.scaler.fit(self.training_data[self.continuous_features])
        self.stds = self.training_data[self.continuous_features].std().values
        
       
        self.cat_distributions = {}
        for cat in self.categorical_features:
            self.cat_distributions[cat] = self.training_data[cat].value_counts(normalize=True)

    def explain_instance(self, instance, predict_fn, num_samples=1000, kernel_width=0.75):
       
        samples = np.zeros((num_samples, len(instance)))
        samples = pd.DataFrame(samples, columns=instance.index)
        
       
        samples.iloc[0] = instance
        
       
        for i, col in enumerate(self.continuous_features):
            samples.iloc[1:, samples.columns.get_loc(col)] = np.random.normal(
                instance[col], self.stds[i], num_samples - 1
            )
            
       
        for cat in self.categorical_features:
            probs = self.cat_distributions[cat]
            samples.iloc[1:, samples.columns.get_loc(cat)] = np.random.choice(
                probs.index, size=num_samples - 1, p=probs.values
            )
            
        
        bb_preds = predict_fn(samples)
        
        
        scaled_samples = samples.copy()
        scaled_samples[self.continuous_features] = self.scaler.transform(samples[self.continuous_features])
        scaled_instance = scaled_samples.iloc[0].values.reshape(1, -1)
        
        
        distances = np.zeros(num_samples)
        for i in range(num_samples):
        
            cont_dist = np.sum((scaled_samples.iloc[i][self.continuous_features] - scaled_instance[0][:2])**2)
        
            cat_dist = 0 if samples.iloc[i]['x3'] == instance['x3'] else 1 
            distances[i] = np.sqrt(cont_dist + cat_dist)
            
        
        weights = np.exp(-(distances ** 2) / (kernel_width ** 2))
        
        
        Z_prime = samples.copy()
        for cat in self.categorical_features:
            Z_prime[cat] = (Z_prime[cat] == instance[cat]).astype(int)
            
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(Z_prime, bb_preds, sample_weight=weights)
        
        explanation = dict(zip(Z_prime.columns, lasso.coef_))
        return explanation, lasso.intercept_, samples, weights




def plot_lime_bar_chart(explanation, instance_idx, save_path):
    features = list(explanation.keys())
    weights = list(explanation.values())
    colors = ['green' if w > 0 else 'red' for w in weights]
    
    plt.figure(figsize=(7, 4))
    plt.barh(features, weights, color=colors)
    plt.axvline(0, color='black', linewidth=1.5)
    plt.title(f'LIME Local Explanation (Instance {instance_idx})')
    plt.xlabel('Feature Contribution (Lasso Coefficient)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_decision_surface(predict_fn, instance, samples, weights, instance_idx, save_path):
    x1_min, x1_max = df['x1'].min() - 0.5, df['x1'].max() + 0.5
    x2_min, x2_max = df['x2'].min() - 0.5, df['x2'].max() + 0.5
    xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
    
    grid_X = pd.DataFrame({'x1': xx1.ravel(), 'x2': xx2.ravel(), 'x3': instance['x3']})
    preds = predict_fn(grid_X).reshape(xx1.shape)
    
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(xx1, xx2, preds, alpha=0.6, cmap='viridis', levels=30)
    plt.colorbar(contour, label='Black-Box Prediction (y)')
    plt.scatter(samples['x1'], samples['x2'], c=weights, cmap='Reds', 
                edgecolors='k', alpha=0.8, s=40, label='LIME Samples (color=weight)')
    plt.scatter([instance['x1']], [instance['x2']], color='cyan', marker='*', 
                s=500, edgecolors='black', linewidths=1.5, label='Target Instance')
    plt.title(f"Black-Box Surface & LIME Neighborhood (Instance {instance_idx}, x3={int(instance['x3'])})")
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def run_sensitivity_analysis(explainer, instance, predict_fn):
    print("\nRunning Sensitivity Analysis (Stability vs Neighborhood Size N)...")
    n_samples_grid = [100, 300, 500, 1000, 2000, 3000]
    variances = []
    
    for n in n_samples_grid:
        coefs_list = []
        for _ in range(10): # Run 10 times to measure variance
            explanation, _, _, _ = explainer.explain_instance(instance, predict_fn, num_samples=n)
            coefs_list.append(explanation['x1']) # Track stability of x1 coefficient
        variances.append(np.std(coefs_list))
        
    plt.figure(figsize=(7, 4))
    plt.plot(n_samples_grid, variances, marker='o', linestyle='-', color='b')
    plt.title("LIME Explanation Stability")
    plt.xlabel("Number of Samples (N)")
    plt.ylabel("Variance of 'x1' Coefficient")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("lime_sensitivity_analysis.png", dpi=300)
    plt.close()
    print("Saved 'lime_sensitivity_analysis.png'")

from sklearn.metrics import r2_score

def run_advanced_parameter_analysis(explainer, instance, predict_fn, instance_idx):
    print(f"\n--- Running Advanced Parameter Analysis for Instance {instance_idx} ---")
    num_samples = 2000
    _, _, samples, _ = explainer.explain_instance(instance, predict_fn, num_samples=num_samples)
    bb_preds = predict_fn(samples)

    Z_prime = samples.copy()
    for cat in explainer.categorical_features:
        Z_prime[cat] = (Z_prime[cat] == instance[cat]).astype(int)
        
    scaled_samples = samples.copy()
    scaled_samples[explainer.continuous_features] = explainer.scaler.transform(samples[explainer.continuous_features])
    scaled_instance = scaled_samples.iloc[0].values.reshape(1, -1)
    
    distances = np.zeros(num_samples)
    for i in range(num_samples):
        cont_dist = np.sum((scaled_samples.iloc[i][explainer.continuous_features] - scaled_instance[0][:2])**2)
        cat_dist = 0 if samples.iloc[i]['x3'] == instance['x3'] else 1 
        distances[i] = np.sqrt(cont_dist + cat_dist)

    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    weights_fixed_k = np.exp(-(distances ** 2) / (0.75 ** 2)) # Fix Kernel Width to 0.75
    
    coef_paths = {col: [] for col in Z_prime.columns}
    
    for a in alphas:
        lasso = Lasso(alpha=a, random_state=42)
        lasso.fit(Z_prime, bb_preds, sample_weight=weights_fixed_k)
        for i, col in enumerate(Z_prime.columns):
            coef_paths[col].append(lasso.coef_[i])
            
    plt.figure(figsize=(7, 4))
    for col in Z_prime.columns:
        plt.plot(alphas, coef_paths[col], marker='o', label=f"Feature: {col}")
    plt.xscale('log')
    plt.title("Lasso Regularization Path (How Alpha affects LIME)")
    plt.xlabel("Lasso Alpha Penalty (Log Scale)")
    plt.ylabel("Feature Coefficient Weight")
    plt.axvline(0.01, color='red', linestyle='--', alpha=0.5, label='Chosen Alpha (0.01)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"lime_lasso_alpha_analysis_{instance_idx}.png", dpi=300)
    plt.close()

    kernel_widths = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    fidelities = []
    
    for k in kernel_widths:
        weights_k = np.exp(-(distances ** 2) / (k ** 2))
        
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(Z_prime, bb_preds, sample_weight=weights_k)
        
        lasso_preds = lasso.predict(Z_prime)
        r2 = r2_score(bb_preds, lasso_preds, sample_weight=weights_k)
        fidelities.append(r2)
        
    plt.figure(figsize=(7, 4))
    plt.plot(kernel_widths, fidelities, marker='s', color='purple')
    plt.title("Local Fidelity vs. Kernel Width")
    plt.xlabel("Kernel Width (Size of Neighborhood)")
    plt.ylabel("Local Fidelity (Weighted R² Score)")
    plt.axvline(0.75, color='red', linestyle='--', alpha=0.5, label='Chosen Kernel (0.75)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"lime_kernel_width_analysis_{instance_idx}.png", dpi=300)
    plt.close()
    
    print(f"Saved Alpha and Kernel Width analysis graphs for Instance {instance_idx}")


if __name__ == "__main__":
    instances_to_explain = [
        pd.Series({'x1': 0.4, 'x2': -0.4, 'x3': 1}),
        pd.Series({'x1': 0.2, 'x2': -0.4, 'x3': 1}),
        pd.Series({'x1': 0.4, 'x2': -0.4, 'x3': 2}),
        pd.Series({'x1': 0.4, 'x2':  0.2, 'x3': 2})
    ]
    
    lime = CustomLIME(training_data=X, categorical_features=['x3'], continuous_features=['x1', 'x2'])
    
    for i, inst in enumerate(instances_to_explain):
        explanation, intercept, samples, weights = lime.explain_instance(inst, predict_fn=bb_predict, num_samples=1000)
        
        bb_pred = bb_predict(inst.to_frame().T)[0]
        lime_pred = intercept + sum([explanation[f] * (1 if f=='x3' else inst[f]) for f in explanation.keys()])
        
        print(f"Black-Box Prediction: {bb_pred:.4f}")
        print(f"LIME Local Prediction: {lime_pred:.4f}")
        print("Feature Contributions:")
        for feat, weight in explanation.items():
            print(f"  {feat}: {weight:.4f}")
            
        
        plot_lime_bar_chart(explanation, i+1, f"lime_bar_inst_{i+1}.png")
        plot_decision_surface(bb_predict, inst, samples, weights, i+1, f"lime_surface_inst_{i+1}.png")

    run_sensitivity_analysis(lime, instances_to_explain[0], bb_predict)
    run_advanced_parameter_analysis(lime, instances_to_explain[0], bb_predict, 1)
