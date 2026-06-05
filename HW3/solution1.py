import math
import numpy as np
import unittest

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value({self.label}: {self.data})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def __pow__(self, other):
        out = Value(self.data ** other, (self, ), f'**{other}')
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other): # other + self
        return self + other
    def __neg__(self): # - self
        return self * -1
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other): # other - self
        return other + (-self)
    def __rmul__(self, other): # other * self
        return self * other
    def exp(self):
        x = max(min(self.data, 500), -500)
        out = Value(math.exp(x), (self,), 'exp')

        def _backward():
            self.grad += out.data * out.grad # Derivative of e^x is e^x
        out._backward = _backward
        return out

    def log(self):
        epsilon = 1e-15

        x = self.data if self.data>epsilon else epsilon
        out = Value(math.log(x), (self,), 'log')

        def _backward():
            self.grad += (1.0 / x) * out.grad # Derivative of ln(x) is 1/x
        out._backward = _backward
        return out    
    
    def backward(self):
    # 1. Build the topological order
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev: # _prev stores the inputs to this operation
                    build_topo(child)
                topo.append(v)
        
        build_topo(self)
        
        # 2. Set the gradient of the root (the loss) to 1.0
        self.grad = 1.0
    
    # 3. Go backward through the topological order
        for node in reversed(topo):
            node._backward()
    

class MultinomialLogReg:
    def __init__(self, learning_rate = 0.1, epochs=200):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.num_classes = None

    def build(self, X, y):
        num_samples = len(X)
        num_features = len(X[0])
        self.num_classes = max(y) + 1

        self.weights = [[Value(0.0) for _ in range(num_features)] for _ in range(self.num_classes - 1)]

        for epoch in range(self.epochs):
            total_loss = Value(0.0)

            for i in range(num_samples):
                xi = X[i]
                yi = y[i]

                scores = []

                for j in range(self.num_classes - 1):
                    # Dot product: xi * weights[j]
                    score = sum((xi[k] * self.weights[j][k] for k in range(num_features)), Value(0.0))
                    scores.append(score)
                # Reference class has a score of 0
                scores.append(Value(0.0))


                exp_scores = [s.exp() for s in scores]
                sum_exp = sum(exp_scores, Value(0.0))

                prob_true_class = exp_scores[yi] * (sum_exp ** -1)
                nll = prob_true_class.log() * (-1.0)
                total_loss = total_loss + nll   

            avg_loss = total_loss * (1.0 / num_samples)   
            avg_loss.backward()

            for j in range(self.num_classes - 1):
                for k in range(num_features):
                    self.weights[j][k].data -= self.lr * self.weights[j][k].grad
                    self.weights[j][k].grad = 0.0

            if epoch % 10 == 0:
                print(f"epoch {epoch} | Loss:{avg_loss.data}")


        return self

    def predict(self, X):
        num_samples = len(X)
        num_features = len(X[0])
        probabilities = []

        for i in range(num_samples):
            xi = X[i]
            scores= []
            for j in range(self.num_classes - 1):
                score = sum((xi[k] * self.weights[j][k].data for k in range(num_features)))
                scores.append(score)
            scores.append(0.0)
            
            max_score = max(scores)
            exp_scores = [math.exp(s - max_score) for s in scores]
            sum_exp = sum(exp_scores)
            probs = [es / sum_exp for es in exp_scores]
            probabilities.append(probs)
        return np.array(probabilities)


class OrdinalLogReg:
    def __init__(self, learning_rate = 0.01, epochs= 200):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.deltas = None
        self.num_classes = None

    def sigmoid(self, z):
        exp_neg_z = (z * -1.0).exp()
        return (Value(1.0) + exp_neg_z) ** -1.0

    def build(self, X, y):
        num_samples = len(X)
        num_features = len(X[0])
        self.num_classes = max(y) + 1

        self.weights = [Value(0.0) for _ in range(num_features)]
        self.deltas = [Value(1.0) for _ in range(self.num_classes - 1)]

        for epoch in range(self.epochs):
            total_loss = Value(0.0)

            for i in range(num_samples):
                xi = X[i]
                yi = y[i]

                u = sum((xi[k] * self.weights[k] for k in range(num_features)), Value(0.0))
                thresholds = [Value(0.0)]
                for d in self.deltas :
                    thresholds.append(thresholds[-1] + d)

                if yi == 0:
                    prob = self.sigmoid(thresholds[0] - u)

                elif yi == self.num_classes - 1:
                    prob = Value(1.0) - self.sigmoid(thresholds[-1] - u)
                else:
                    upper_t = thresholds[yi]
                    lower_t = thresholds[yi - 1]
                    prob = self.sigmoid(upper_t - u) - self.sigmoid(lower_t - u)

                prob = prob + Value(1e-15)
                nll= prob.log() * -1.0                
                total_loss = total_loss + nll

            avg_loss = total_loss * (1 / num_samples)
            avg_loss.backward()

            for k in range(num_features):
                self.weights[k].data -= self.lr *self.weights[k].grad
                self.weights[k].grad = 0.0

            for idx in range(len(self.deltas)):
                if self.deltas[idx].data <= 1e-5:
                    self.deltas[idx].data = 1e-5
                self.deltas[idx].grad = 0.0

            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {avg_loss.data}")

        return self

    def predict(self, X):
        num_samples = len(X)
        num_features = len(X[0])
        probabilities = []

        def sigmoid_float(z):
            if z >= 0:
                return 1.0 / (1.0 + math.exp(-z))
            else:
                exp_z = math.exp(z)
                return exp_z / (1.0 + exp_z)
        
        thresholds = [0.0]
        for d in self.deltas:
            thresholds.append(thresholds[-1] + d.data)

        for i in range(num_samples):
            xi = X[i]

            u = sum(xi[k] * self.weights[k].data for k in range(num_features))

            probs = []
            for class_idx in range(self.num_classes):
                if class_idx == 0:
                    p = sigmoid_float(thresholds[0] - u)
                elif class_idx == self.num_classes - 1:
                    p = 1 - sigmoid_float(thresholds[class_idx] - u)
                else:
                    upper_thresh = thresholds[class_idx]
                    lower_thresh = thresholds[class_idx - 1]
                    p = sigmoid_float(upper_thresh - u) - sigmoid_float(lower_thresh - u)
                p = p + 1e-15
                probs.append(p)
            probabilities.append(probs)

        return np.array(probabilities)                    
                                    


class MyTests(unittest.TestCase):

    def setUp(self):
        self.X_mult = np.array([0.0, 0.0], [0.1, 0.1], [0.9, 0.9], [1.0, 1.0])
        self.y_mult = np.array([0, 0, 1, 1])

        self.X_ord = np.array([1.0], [2.0], [3.0], [4.0], [5.0])
        self.y_ord = np.array([0, 0, 1, 1, 2])

    def test_multinomial_learning(self):
        model = MultinomialLogReg(learning_rate=0.5, epochs = 100)
        model.build(self.X_mult, self.y_mult)

        probs = model.predict(self.X_mult)

        self.assertGreater(probs[0][0], 0.5)
        self.assertGreater(probs[3][1], 0.5)

    def test_ordinal_shape_sum(self):
        model = OrdinalLogReg(learning_rate=0.1, epochs=5) 
        model.build(self.X_ord, self.y_ord)

        probs = model.predict(self.X_ord)

        self.assertEqual(probs.shape, (5,3))

        np.testing.assert_almost_equal(probs.sum(axis=1), 1)

if __name__ == "main":
    unittest.main()
    
