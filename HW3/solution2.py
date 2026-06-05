import numpy as np
from scipy.optimize import minimize

class MultinomialLogReg:
    def __init__(self):
        self.weights = None
        self.num_classes = None
        self.hess_inv = None 


    def nll(self, params, X, y, num_features):
        W = params.reshape((num_features, self.num_classes - 1))
        scores = np.dot(X, W)

        zeros_col = np.zeros((X.shape[0], 1))
        scores = np.hstack([scores, zeros_col]) 
        
        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        true_probs = probs[np.arange(len(y)), y]
        
        log_probs = np.log(np.clip(true_probs, 1e-15, 1.0))
        return -np.sum(log_probs)
    
    def build(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten().astype(int)
        num_samples, num_features = X.shape
        self.num_classes = int(np.max(y)) + 1

        first_params = np.zeros(num_features * (self.num_classes - 1))

        result = minimize(
            fun = self.nll,
            x0 = first_params,
            args=(X,y,num_features),
            method='BFGS'
        )
        self.hess_inv = result.hess_inv
        self.weights = result.x.reshape((num_features, self.num_classes - 1))
        return self

    def predict(self, X):
        X = np.array(X)
        scores = np.dot(X, self.weights)
        zeros_col = np.zeros((X.shape[0], 1))
        scores = np.hstack([scores, zeros_col])

        max_scores = np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(scores - max_scores)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        return probs
    
    def standard_error(self):
        if self.hess_inv is None:
            raise ValueError("Not fitted")
        se_flat = np.sqrt(np.diag(self.hess_inv))
        return se_flat.reshape(self.weights.shape)
    

class OrdinalLogReg:
    def __init__(self):
        self.weights = None
        self.num_classes = None
        self.deltas = None

    def sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def nll(self, params, X, y, num_features):
        beta = params[:num_features]
        deltas = params[num_features:]

        u = np.dot(X, beta)

        thresholds = np.zeros(self.num_classes - 1)
        if len(deltas) > 0:
            thresholds[1:] = np.cumsum(deltas)

        probs = np.zeros(len(y))
        for i in range(len(y)):
            yi = y[i]
            ui = u[i]

            if yi == 0:
                probs[i] = self.sigmoid(thresholds[0] - ui)
            elif yi == self.num_classes - 1:
                probs[i] = 1 - self.sigmoid(thresholds[-1] - ui)
            else:
                upper = thresholds[yi]
                lower = thresholds[yi - 1]
                probs[i] = self.sigmoid(upper - ui) - self.sigmoid(lower -ui)

        log_probs = np.log(np.clip(probs, 1e-15, 1.0))
        return -np.sum(log_probs)

    def build(self, X, y):
        X = np.array(X)
        y = np.array(y).flatten().astype(int)
        num_samples, num_features = X.shape
        self.num_classes = np.max(y) + 1

        first_beta = np.zeros(num_features)
        first_deltas = np.ones(max(0, self.num_classes - 2))
        first_params = np.concatenate([first_beta, first_deltas])

        bounds = [(None, None)] * num_features + [(1e-5, None)] * len(first_deltas)
        
        result = minimize(
            fun = self.nll,
            x0 = first_params,
            args = (X,y, num_features),
            method="L-BFGS-B",
            bounds = bounds
        )

        self.weights = result.x[:num_features]
        self.deltas = result.x[num_features:]
        return self

    def predict(self, X):
        X = np.array(X)
        u = np.dot(X, self.weights)

        thresholds = np.zeros(self.num_classes - 1)
        if len(self.deltas) >0 :
            thresholds[1:] = np.cumsum(self.deltas)

        probs = np.zeros((X.shape[0], self.num_classes))

        for i in range(X.shape[0]):
            ui = u[i]
            for class_idx in range(self.num_classes):
                if class_idx == 0:
                    probs[i, class_idx] = self.sigmoid(thresholds[0] - ui)
                elif class_idx == self.num_classes - 1:
                    probs[i, class_idx] = 1 - self.sigmoid(thresholds[-1] - ui)
                else:
                    upper = thresholds[class_idx]
                    lower = thresholds[class_idx - 1]
                    probs[i, class_idx] = self.sigmoid(upper - ui) - self.sigmoid(lower - ui)

        return probs
                            