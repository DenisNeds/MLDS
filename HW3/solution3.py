import numpy as np
import solution2 as s2

def multinomial_bad_ordinal_good(n_samples=1000):
    np.random.seed(42)
    
    # 1. Generate features (e.g., Age and a Biomarker)
    X = np.random.normal(0, 1, size=(n_samples, 2))
    
    # 2. Create the "Latent" variable (True underlying disease progression)
    true_weights = np.array([1.5, 2.0])
    latent_progression = np.dot(X, true_weights) + np.random.logistic(0, 1, size=n_samples)
    
    # 3. Apply thresholds to create the classes
    y = np.zeros(n_samples)
    y[latent_progression > 1.0] = 1 # Moderate
    y[latent_progression > 6.0] = 2 # Severe (Rare threshold)
    
    return X, y

def main():
    X, y = multinomial_bad_ordinal_good(n_samples=1000)
    unique, counts = np.unique(y, return_counts=True)
    print("Class Distribution:")
    for u, c in zip(unique, counts):
        class_name = ["Mild", "Moderate", "Severe"][int(u)]
    
    # Train Multinomial
    model_multi = s2.MultinomialLogReg()
    model_multi.build(X, y)
    probs_multi = model_multi.predict(X)
    
    # Train Ordinal
    model_ord = s2.OrdinalLogReg()
    model_ord.build(X, y)
    probs_ord = model_ord.predict(X)
    rare_indices = np.where(y == 2)[0]
    
    multi_rare_preds = np.argmax(probs_multi[rare_indices], axis=1)
    ord_rare_preds = np.argmax(probs_ord[rare_indices], axis=1)
    
    multi_correct = np.sum(multi_rare_preds == 2)
    ord_correct = np.sum(ord_rare_preds == 2)
    
    print(f"Multinomial caught: {multi_correct} out of {len(rare_indices)} severe cases")
    print(f"Ordinal caught:     {ord_correct} out of {len(rare_indices)} severe cases")
    
    print("\nAverage predicted probability for the TRUE Severe class:")
    print(f"Multinomial: {np.mean(probs_multi[rare_indices, 2]):.4f}")
    print(f"Ordinal:     {np.mean(probs_ord[rare_indices, 2]):.4f}")
    
    print("="*60)

if __name__ == "__main__":
    main()