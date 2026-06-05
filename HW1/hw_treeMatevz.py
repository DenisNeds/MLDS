import csv
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm


def all_columns(X, rand):
    return range(X.shape[1])

# sqrt of all columns!
def random_sqrt_columns(X, rand):
    n = X.shape[1]
    # sqrt_n = int(math.ceil(math.sqrt(n)))
    sqrt_n = int((math.sqrt(n)))
    return rand.sample(range(n),sqrt_n)

#Node class
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, class_label=None):
        self.feature = feature      # index of feature used for split
        self.threshold = threshold  # split threshold
        self.left = left            # left child Node
        self.right = right          # right child Node
        self.class_label = class_label  # if leaf, store class

class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2,
                 root_only = False):
        # TODO: implement rand for seed
        self.rand = rand  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples
        self.root_only = root_only

    # gini
    @staticmethod
    def gini(y):
        _, counts = np.unique(y, return_counts=True)
        p = counts / counts.sum()
        return 1 - np.sum(p ** 2)

    def best_split(self, X, y):
        n_samples, _ = X.shape
        # Columns to consider
        candidate_columns = self.get_candidate_columns(X, self.rand)

        parent_gini = self.gini(y)  # current node impurity
        best_feature, best_threshold, best_gini = None, None, parent_gini
        # best_feature, best_threshold, best_gini = None, None, 1.0

        for feature in candidate_columns:
            thresholds = np.unique(X[:, feature])
            for t in thresholds:
                left_mask = X[:, feature] <= t
                right_mask = X[:, feature] > t
                if left_mask.sum() < self.min_samples or right_mask.sum() < self.min_samples:
                    continue
                gini_left = self.gini(y[left_mask])
                gini_right = self.gini(y[right_mask])
                gini_split = (left_mask.sum() * gini_left + right_mask.sum() * gini_right) / n_samples
                if gini_split < best_gini:
                    best_gini = gini_split
                    best_feature = feature
                    best_threshold = t

        return best_feature, best_threshold

    def build_tree(self, X, y):
        if len(y)<self.min_samples or len(np.unique(y)) == 1:
            return Node(class_label=np.bincount(y).argmax())
        
        feature, threshold = self.best_split(X, y)
        if feature is None: #no split would improve score
            return Node(class_label=np.bincount(y).argmax())
        
        if self.root_only:
            return Node(feature=feature, threshold=threshold)

        # mask for left and right node        
        left_mask = X[:,feature] <= threshold
        right_mask = X[:,feature] > threshold

        left_node = self.build_tree(X[left_mask], y[left_mask])
        right_node = self.build_tree(X[right_mask], y[right_mask])
        return Node(feature=feature, threshold=threshold, left=left_node, right=right_node)

    def build(self, X, y):
        X = np.array(X)
        y = np.array(y)
        tree_root = self.build_tree(X, y)
        return TreeModel(tree_root)

class TreeModel:

    def __init__(self, node: Node):
        self.node = node
        return

    def predict(self, X):
        # ...
        X = np.atleast_2d(X) # to fix 1 entry search!!!

        def traverse(x, node: Node):
            if node.class_label is not None:
                return node.class_label
            if x[node.feature] <= node.threshold:
                return traverse(x, node.left)
            else:
                return traverse(x, node.right)
        
        return np.array([traverse(x, self.node) for x in X])


class RandomForest:

    def __init__(self, rand=random.Random(42), n=50, min_samples=2): # Added minsamples in case i need to increase this
        # implement random-seed
        self.n = n
        self.rand = rand
        # self.rng = random.Random(self.rand)
        self.rftree = Tree(rand=self.rand,get_candidate_columns=random_sqrt_columns,min_samples=min_samples)  # initialize the tree properly
        

    def build(self, X, y):
        # ...
        trees = []
        oob_indices = []
        n_samples = len(X)
        #DO I NEED TO BOOTSTRAP? OR ARE THE SAMPLES GIVEN ALREADY BOOTSTRAPPED?!
        for _ in range(self.n):

            # bootstrap
            # idx = self.rng.integers(0, n_samples, n_samples)
            idx = self.rand.choices(range(n_samples), k=n_samples)

            X_boot = X[idx]
            y_boot = y[idx]

            model = self.rftree.build(X_boot, y_boot)

            # all_indices = set(range(n_samples))
            # oob_indices = list(all_indices - set(idx))
            # X_oob = X[oob_indices]
            # y_oob = y[oob_indices]
            oob_idx = list(set(range(n_samples)) - set(idx))
            oob_indices.append(oob_idx)
            # model = self.rftree.build(X, y)

            trees.append(model)

        return RFModel(trees, X, y, oob_indices)


        # return RFModel(...)  # return an object that can do prediction


class RFModel:

    def __init__(self, trees: list[TreeModel],X: np.ndarray ,y: np.ndarray, oob_indices):
        self.trees = trees
        # This X and y are OOB samples
        self.X = X
        self.y = y
        self.oob_indices = oob_indices
        return

    def predict(self, X):
        # ...
        X = np.atleast_2d(X)
        predicts = np.array([tree.predict(X) for tree in self.trees])
        result = []
        for i in range(X.shape[0]):
            votes = predicts[:, i]
            result.append(np.bincount(votes).argmax())

        return np.array(result)
    
    def importance(self):
        n_features = self.X.shape[1]
        importances = np.zeros(n_features)

        for j in range(n_features):
            imp_per_tree = []
            for b, tree in enumerate(self.trees):
                oob_idx = self.oob_indices[b]
                if len(oob_idx) == 0:
                    # print("SKIP")
                    continue  # -> skip trees with no OOB samples
                # print("We did it")
                X_oob = self.X[oob_idx]
                y_oob = self.y[oob_idx]

                # baseline error for this tree
                pred_baseline = tree.predict(X_oob)
                err_baseline = (pred_baseline != y_oob).mean()

                # permute feature j in OOB
                X_perm = X_oob.copy()
                np.random.shuffle(X_perm[:, j])
                pred_perm = tree.predict(X_perm)
                err_perm = (pred_perm != y_oob).mean()

                imp_per_tree.append(err_perm - err_baseline)

            # average importance across trees
            importances[j] = np.mean(imp_per_tree)
        return importances


def read_tab(fn, adict):
    content = list(csv.reader(open(fn, "rt"), delimiter="\t"))

    legend = content[0][1:]
    data = content[1:]

    X = np.array([d[1:] for d in data], dtype=float)
    y = np.array([adict[d[0]] for d in data])

    return legend, X, y


def tki():
    legend, Xt, yt = read_tab("tki-train.tab", {"Bcr-abl": 1, "Wild type": 0})
    _, Xv, yv = read_tab("tki-test.tab", {"Bcr-abl": 1, "Wild type": 0})
    return (Xt, yt), (Xv, yv), legend

#TODO: CROSSVALIDATION!!! - For uncertanty?
def hw_tree_full(learn, test):
    X_learn, y_learn = learn
    X_test, y_test = test
        
    # Build the tree
    tree_model = Tree().build(X_learn, y_learn)
    
    # Predict on training 
    y_pred_train = tree_model.predict(X_learn)
    misclass_train = (y_pred_train != y_learn).mean()
    se_train = ((misclass_train * (1 - misclass_train)) / len(y_learn)) ** 0.5  # standard error
    
    # Predict on test 
    y_pred_test = tree_model.predict(X_test)
    misclass_test = (y_pred_test != y_test).mean()
    se_test = ((misclass_test * (1 - misclass_test)) / len(y_test)) ** 0.5  # standard error
    
    return (misclass_train, se_train), (misclass_test, se_test)

# def hw_tree_full(learn, test):
#     return
def hw_randomforests(learn, test):
    X_learn, y_learn = learn
    X_test, y_test = test

    forest_model = RandomForest(n=100).build(X_learn,y_learn)

    # Predict on training 
    y_pred_train = forest_model.predict(X_learn)
    misclass_train = (y_pred_train != y_learn).mean()
    se_train = ((misclass_train * (1 - misclass_train)) / len(y_learn)) ** 0.5  # standard error
    
    # Predict on test 
    y_pred_test = forest_model.predict(X_test)
    misclass_test = (y_pred_test != y_test).mean()
    se_test = ((misclass_test * (1 - misclass_test)) / len(y_test)) ** 0.5  # standard error
    
    return (misclass_train, se_train), (misclass_test, se_test)



def plot_different_forest_sizes(learn, test):

    X_learn, y_learn = learn
    X_test, y_test = test

    ns = [1, 5, 10, 20, 50, 100, 150, 200, 400]

    train_errors = []
    test_errors = []

    for n in ns:
        forest_model = RandomForest(n=n).build(X_learn, y_learn)

        # train error
        y_pred_train = forest_model.predict(X_learn)
        train_errors.append((y_pred_train != y_learn).mean())

        # test error
        y_pred_test = forest_model.predict(X_test)
        test_errors.append((y_pred_test != y_test).mean())

    plt.plot(ns, train_errors, marker='o', label="train error")
    plt.plot(ns, test_errors, marker='o', label="test error") # 'o'to x?

    plt.xlabel("Number of trees (n)")
    plt.ylabel("Misclassification rate")
    plt.title("Random Forest performance vs number of trees")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_importance(learn, test):
    X_train, y_train = learn
    X_test, y_test = test

    # --- 1. Build random forest for OOB permutation importance ---
    rf_model = RandomForest(rand=random.Random(0), n=100).build(X_train, y_train)
    perm_importances = rf_model.importance()  # ESL-style OOB permutation importance

    # --- 2. Build 100 truly non-random trees on randomized labels for baseline ---
      # shuffle labels

    n_features = X_train.shape[1]
    n_trees = 100
    root_counts = np.zeros(n_features)
    t = Tree(rand=random.Random(1), get_candidate_columns=all_columns,root_only=True)
    for _ in tqdm(range(n_trees)):
        perm = np.random.permutation(len(y_train))
        y_random = y_train[perm]
        # y_random = np.random.permutation(y_train)
        # Non-random tree: use all features at every split, no bootstrap needed
        tree_model = t.build(X_train, y_random)

        # Record root feature
        # root_feature = tree_model.node.feature
        # if root_feature is not None:
        #     root_counts[root_feature] += 1

        root_feature = tree_model.node.feature
        if root_feature is not None:
            root_counts[root_feature] += 1

    # Normalize to fraction
    root_counts = root_counts / n_trees

    # --- 3. Plot ---
    features = list(range(n_features))  # keep original feature order
    plt.figure(figsize=(10,5))

    # Permutation importance (OOB) as bars
    plt.bar(features, perm_importances, alpha=0.6, label='Permutation Importance (OOB)')

    # Root-feature frequency as red dots, scaled to match max height of bars
    plt.bar(features, root_counts * max(perm_importances),alpha = 0.6, color='red', label='Root Feature Frequency (randomized baseline)', zorder=10)

    plt.xlabel("Features")
    plt.ylabel("Importance")
    plt.title("Variable Importance in Random Forest (n=100 trees)")
    plt.xticks(features)  # preserve feature order
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_importance2(learn, test):
    X_train, y_train = learn
    X_test, y_test = test

    # --- 1. Random forest permutation importance ---
    rf_model = RandomForest(rand=random.Random(0), n=100).build(X_train, y_train)
    perm_importances = rf_model.importance()

    # --- 2. Baseline: roots of 100 non-random trees on randomized labels ---
    n_features = X_train.shape[1]
    n_trees = 100
    root_counts = np.zeros(n_features)

    t = Tree(rand=random.Random(1), get_candidate_columns=all_columns, root_only=True)

    for _ in tqdm(range(n_trees)):
        perm = np.random.permutation(len(y_train))
        y_random = y_train[perm]

        tree_model = t.build(X_train, y_random)

        root_feature = tree_model.node.feature
        if root_feature is not None:
            root_counts[root_feature] += 1

    root_counts = root_counts / n_trees

    # --- 3. Plot with two y-axes ---
    features = np.arange(n_features)

    fig, ax1 = plt.subplots(figsize=(12,5))

    # permutation importance (left axis)
    ax1.bar(features, perm_importances, alpha=0.6, label="Permutation Importance (OOB)")
    ax1.set_xlabel("Features")
    ax1.set_ylabel("Permutation Importance")
    ax1.set_xticks(features)

    # second axis for root frequencies
    ax2 = ax1.twinx()
    ax2.bar(features, root_counts, alpha=0.4, color="red",
            label="Root Feature Frequency (randomized baseline)")
    ax2.set_ylabel("Root Frequency")

    # combine legends
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2)

    plt.title("Variable Importance in Random Forest (n=100 trees)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    learn, test, legend = tki()

    # print("full", hw_tree_full(learn, test))
    # print("random forests", hw_randomforests(learn, test))
    # plot_different_forest_sizes(learn,test)

    plot_importance2(learn, test)

#TODO:
# Test your implementation with unit tests that focus on the critical or hard parts and edge cases. Combine all tests in a class named MyTests.
# Do in notebook so we dont have to recompute!!