import numpy as np
from solution2 import MultinomialLogReg, OrdinalLogReg

def multinomial_bad_ordinal_good(n_samples=40, n_features=10):
    """The Data-Generating Process (DGP)"""
    np.random.seed(np.random.randint(0, 10000)) # Randomize seed for each call
    
    X = np.random.randn(n_samples, n_features)
    
    # Only the first two features matter
    true_beta = np.zeros(n_features)
    true_beta[0] = 3.0  
    true_beta[1] = 1.0  
    
    # Latent variable with logistic noise
    logistic_noise = np.random.logistic(loc=0.0, scale=1.0, size=n_samples)
    y_star = np.dot(X, true_beta) + logistic_noise
    
    # Thresholds to create classes 0, 1, 2, 3 (Bald, Short, Medium, Long)
    y = np.zeros(n_samples, dtype=int)
    y[y_star > -1.0] = 1 
    y[y_star > 1.0]  = 2 
    y[y_star > 3.0]  = 3 
    
    return X, y

def main():
    print("="*60)
    print("🥊 MULTINOMIAL VS ORDINAL: THE SHOWDOWN 🥊")
    print("="*60)

    # 1. Generate the datasets
    # We use a tiny training set to force the Multinomial model to overfit
    print("Generating 40 training samples and 1000 testing samples...")
    X_train, y_train = multinomial_bad_ordinal_good(n_samples=100, n_features=10)
    X_test, y_test = multinomial_bad_ordinal_good(n_samples=100, n_features=10)

    # 2. Train Multinomial Model
    print("\nTraining Multinomial Logistic Regression...")
    multi_model = MultinomialLogReg()
    multi_model.build(X_train, y_train)

    # 3. Train Ordinal Model
    print("Training Ordinal Logistic Regression...")
    ordinal_model = OrdinalLogReg()
    ordinal_model.build(X_train, y_train)

    # 4. Predict on the unseen TEST set
    # predict() returns probabilities, so we use argmax to pick the highest percentage class
    multi_probs = multi_model.predict(X_test)
    multi_preds = np.argmax(multi_probs, axis=1)

    ordinal_probs = ordinal_model.predict(X_test)
    ordinal_preds = np.argmax(ordinal_probs, axis=1)

    # 5. Calculate Accuracy
    multi_accuracy = np.mean(multi_preds == y_test) * 100
    ordinal_accuracy = np.mean(ordinal_preds == y_test) * 100

    print("\n" + "="*60)
    print("RESULTS ON UNSEEN DATA (TEST ACCURACY)")
    print("="*60)
    print(f"Multinomial Model Accuracy: {multi_accuracy:.2f}%")
    print(f"Ordinal Model Accuracy:     {ordinal_accuracy:.2f}%")
    
    if ordinal_accuracy > multi_accuracy:
        diff = ordinal_accuracy - multi_accuracy
        print(f"\n🏆 Ordinal Wins by {diff:.2f}%! It successfully avoided overfitting.")
    else:
        print("\nTie or Multinomial win (Run again, random chance can sometimes be weird!)")

if __name__ == "__main__":
    main()