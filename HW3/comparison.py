import time
import numpy as np
import solution1 as s1
import solution2 as s2

def main():
    print("="*60)
    print(" GLM IMPLEMENTATION COMPARISON (PART 1) ")
    print("="*60)

    # 1. Generate a small dummy dataset for testing
    # We use a small number of samples because pure Python autograd is computationally heavy
    np.random.seed(42)
    n_samples = 50
    n_features = 3
    
    X_dummy = np.random.randn(n_samples, n_features)
    # 3 classes for multinomial and ordinal
    y_dummy = np.random.randint(0, 3, size=n_samples) 
    
    # Sort y for ordinal to make the thresholds conceptually easier to fit
    y_dummy_ord = np.sort(y_dummy)

    print("\n--- 1. MULTINOMIAL LOGISTIC REGRESSION ---")
    
    # Solution 1: Custom Autograd
    print("\nStarting Solution 1 (Autograd)...")
    start_time = time.time()
    # Using 50 epochs to keep wait times reasonable
    model_m1 = s1.MultinomialLogReg(learning_rate=0.1, epochs=200)
    model_m1.build(X_dummy.tolist(), y_dummy.tolist())
    s1_time = time.time() - start_time
    print(f"Solution 1 Time: {s1_time:.4f} seconds")

    # Solution 2: Scipy Optimize
    print("\nStarting Solution 2 (Scipy Optimize)...")
    start_time = time.time()
    model_m2 = s2.MultinomialLogReg()
    model_m2.build(X_dummy, y_dummy)
    s2_time = time.time() - start_time
    print(f"Solution 2 Time: {s2_time:.4f} seconds")

    print(f"\nSpeed Ratio: Scipy is roughly {s1_time / s2_time:.1f}x faster.")


    print("\n" + "="*60)
    print("--- 2. ORDINAL LOGISTIC REGRESSION ---")
    
    # Solution 1: Custom Autograd
    print("\nStarting Solution 1 (Autograd)...")
    start_time = time.time()
    model_o1 = s1.OrdinalLogReg(learning_rate=0.001, epochs=200)
    model_o1.build(X_dummy.tolist(), y_dummy_ord.tolist())
    o1_time = time.time() - start_time
    print(f"Solution 1 Time: {o1_time:.4f} seconds")

    # Solution 2: Scipy Optimize
    print("\nStarting Solution 2 (Scipy Optimize)...")
    start_time = time.time()
    model_o2 = s2.OrdinalLogReg()
    model_o2.build(X_dummy, y_dummy_ord)
    o2_time = time.time() - start_time
    print(f"Solution 2 Time: {o2_time:.4f} seconds")

    print(f"\nSpeed Ratio: Scipy is roughly {o1_time / o2_time:.1f}x faster.")
    print("="*60)
    print(f"\nSpeed Ratio: Scipy is roughly {o1_time / o2_time:.1f}x faster.")
    print("\n" + "="*60)
    print("--- 3. CONVERGENCE METRICS (ORDINAL) ---")
    
    # 1. Extract weights
    # solution1 weights are custom Value objects, so we extract the .data
    w1 = np.array([w.data for w in model_o1.weights])
    w2 = model_o2.weights # solution2 weights are already a numpy array
    
    # Calculate how different the weights are
    weight_diff = np.max(np.abs(w1 - w2))
    print(f"Max Absolute Difference in Weights: {weight_diff:.6f}")
    if weight_diff < 0.1:
        print("-> SUCCESS: Both models converged to practically the same weights!")
    else:
        print("-> NOTE: Weights differ. Autograd might need more epochs/different LR to fully converge.")

    # 2. Compare Predictions
    probs1 = model_o1.predict(X_dummy)
    probs2 = model_o2.predict(X_dummy)
    
    # Calculate Mean Absolute Error of probabilities
    prob_mae = np.mean(np.abs(probs1 - probs2))
    print(f"\nPrediction Mean Absolute Error (MAE): {prob_mae:.6f}")
    if prob_mae < 0.05:
        print("-> SUCCESS: Both models predict almost identical probabilities!")

    print("="*60)

if __name__ == "__main__":
    main()