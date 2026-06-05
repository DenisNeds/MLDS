import csv
import numpy as np
import random
import unittest
import matplotlib.pyplot as plt


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    n_feats = X.shape[1]
    k = int(np.sqrt(n_feats))
    c = rand.sample(range(n_feats), k)

    return c

def gini_impurity(y):
    if len(y) == 0:
        return 0
    _, counts = np.unique(y, return_counts = True)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

class Node:
    def __init__(self,feature_id = None, left = None, right = None, threshold = None, value = None):
        self.feature_id = feature_id
        self.left = left
        self.right = right
        self.threshold = threshold
        self.value = value



class Tree:

    def __init__(self, rand=None,
                 get_candidate_columns=all_columns,
                 min_samples=2):
        self.rand = rand if rand is not None else random.Random()
        self.get_candidate_columns = get_candidate_columns  
        self.min_samples = min_samples

    def build(self, X, y):
        
        def buildTree(x_set, y_set):
            n_samples = len(y_set)
            if n_samples < self.min_samples or len(np.unique(y_set)) == 1:
                uni_class, counts = np.unique(y_set, return_counts = True)
                major_class = uni_class[np.argmax(counts)]
                return Node(value = major_class)
            
            best_gain = -1
            best_split = None

            candidates = self.get_candidate_columns(x_set, self.rand)
            parent_gini = gini_impurity(y_set)

            for cand in candidates:
                thresholds = np.unique(x_set[:, cand])

                for threshold in thresholds:
                    left_mask = x_set[:, cand] <= threshold
                    right_mask = ~left_mask
                    
                    y_left, y_right = y_set[left_mask], y_set[right_mask]

                    if len(y_left) == 0 or len(y_right) == 0:
                        continue
                    
                    left_weight = len(y_left)/n_samples
                    right_weight = len(y_right)/n_samples
                    weighted_gini =(left_weight * gini_impurity(y_left)) + (right_weight * gini_impurity(y_right))
                    
                    gain = parent_gini - weighted_gini

                    if gain > best_gain:
                        best_gain = gain
                        best_split = {
                            "feature": cand,
                            "threshold": threshold,
                            "left_x":x_set[left_mask],
                            "left_y":y_left,
                            "right_x":x_set[right_mask],
                            "right_y":y_right
                        }

            if best_gain > 0:
                left_child = buildTree(best_split["left_x"], best_split["left_y"])
                right_child = buildTree(best_split["right_x"], best_split["right_y"])
                return Node(feature_id=best_split["feature"], left = left_child, right = right_child, threshold=best_split["threshold"])

            else:
                uni_class, counts = np.unique(y_set, return_counts = True)
                major_class = uni_class[np.argmax(counts)]
                return Node(value = major_class)

        root_node = buildTree(X,y)
        return TreeModel(root_node)  


class TreeModel:

    def __init__(self, root):
        self.root = root

    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_id] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.right)        

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self.traverse_tree(sample, self.root))
        return np.array(predictions)
                
class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand if rand is not None else random.Random()
        self.rftree = Tree(rand = self.rand,
                           get_candidate_columns=random_sqrt_columns, #v random forest vzamem sqrt(#featurov)
                            min_samples=2 )

    def build(self, X, y):
        models = []
        oob_samples = []
        n_samples = len(X)
        indices = set(range(n_samples))

        #bootstrap
        for _ in range(self.n):
            #choices je za samplanje sz ponavljanjem
            random_indices = self.rand.choices(range(n_samples), k= n_samples)
            oob_indices = list(indices - set(random_indices))
            x_boot = X[random_indices]
            y_boot = y[random_indices]

            model = self.rftree.build(x_boot, y_boot)
            models.append(model)
            oob_samples.append(oob_indices)

        return RFModel(models, X, y, oob_samples, self.rand) 


class RFModel:

    def __init__(self, models, X, y, oob_samples, rand):
        self.models = models
        self.X = X
        self.y = y
        self.oob_samples = oob_samples
        self.rand = rand 


    def predict(self, X):
        all_predicts = np.array([model.predict(X) for model in self.models])

        predictions = []

        for i in range(len(X)):
            sample_result = all_predicts[:, i]

            classes, counts = np.unique(sample_result, return_counts = True)
            major_class = classes[np.argmax(counts)]

            predictions.append(major_class)

        return np.array(predictions)

    def importance(self):
        n_feats = self.X.shape[1]
        imps = np.zeros(n_feats)
        for i, tree_model in enumerate(self.models):
            oob_idx = self.oob_samples[i]

            if len(oob_idx) == 0:
                continue

            x_oob = self.X[oob_idx]
            y_oob = self.y[oob_idx]

            baseline_preds = tree_model.predict(x_oob)
            baseline_accuracy = np.mean(baseline_preds == y_oob)

            for j in range(n_feats):
                X_scrambled = x_oob.copy()
                #premesas vrednosti dolocenga featura
                col_values = list(X_scrambled[:, j])
                self.rand.shuffle(col_values)
                X_scrambled[:, j] = col_values

                scrambled_preds = tree_model.predict(X_scrambled)
                scrambled_accuracy = np.mean(scrambled_preds == y_oob)
                #primerjas z baseline accuracyjem da vids ce so rezultati tega featura rendom al ne 
                imps[j] += (baseline_accuracy - scrambled_accuracy)

        imps = imps/len(self.models)
        
        return imps
    


class MyTests(unittest.TestCase):
    
    def test_gini_impurity(self):
        #pure nam vrne 0 ker 1 - 1^2 - 0^2 = 0
        y_pure = np.array([1,1,1,1])
        self.assertEqual(gini_impurity(y_pure), 0.0)

        #50/50 nam vrne impurity 0.5, 1 - 0.5^2 - 0.5^2 = 0.5
        y_mixed = np.array([0,0,1,1])
        self.assertEqual(gini_impurity(y_mixed), 0.5)

    def test_pure_node_returns_leaf(self):
        """če so vsi classi enak pol ne splita"""
        x_dummy = np.array([[1.0, 2.0], [1.5, 2.5], [3.0, 4.0]])
        y_pure = np.array([1,1,1])
        my_tree = Tree()
        model = my_tree.build(x_dummy, y_pure)

        #root node bo takoj leaf, levi desni sta pa none
        self.assertIsNotNone(model.root.value)
        self.assertIsNone(model.root.left)
        self.assertIsNone(model.root.right)
        self.assertEqual(model.root.value, 1) #mora predictat 1 ker je pure enk

    def test_min_samples_limit(self):
        """Če je manj samplov kot min_samples potem se ustavi"""
        x_dummy = np.array([[1.0],[2.0],[3.0]])
        y_mixed = np.array([0,1,0])

        my_tree = Tree(min_samples=10)
        model = my_tree.build(x_dummy, y_mixed)

        self.assertIsNotNone(model.root.value)
        self.assertEqual(model.root.value, 0)

    def test_zero_variance_features(self):
        """Če so vsi featuri enaki potem ni gaina"""
        X_identical = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0], [5.0, 5.0]])
        y_mixed = np.array([0, 0, 1, 1])

        my_tree = Tree()
        model = my_tree.build(X_identical, y_mixed)

        self.assertIsNotNone(model.root.value)

    def test_forest_reproducability(self):

        x_dummy = np.random.rand(20, 5)
        y_dummy = np.random.randint(0, 2, 20)

        seed1 = random.Random(42)
        rf1 = RandomForest(rand=seed1, n=5)
        model1 = rf1.build(x_dummy, y_dummy)
        preds1 = model1.predict(x_dummy)

        seed2 = random.Random(42)
        rf2 = RandomForest(rand=seed2, n=5)
        model2 = rf2.build(x_dummy, y_dummy)
        preds2 = model2.predict(x_dummy)

        np.testing.assert_array_equal(preds1, preds2)

    def test_hw_tree_full_math(self):
        """Preveri če hw_tree_full vrne pravilne oblike in izračune (error, se)"""
        X_dummy = np.array([[1.0], [2.0], [8.0], [9.0]])
        y_dummy = np.array([0, 0, 1, 1])
        
        train = (X_dummy, y_dummy)
        test = (X_dummy, y_dummy)
        
        train_res, test_res = hw_tree_full(train, test)
        
        self.assertEqual(len(train_res), 2)
        self.assertEqual(train_res[0], 0.0) 
        self.assertEqual(train_res[1], 0.0)    

    def test_hw_rf_math(self):
        """Preveri če hw_randomforests vrne pravilne oblike in izračune (error, se)"""
        X_dummy = np.array([[1.0], [2.0], [8.0], [9.0]])
        y_dummy = np.array([0, 0, 1, 1])
        
        train = (X_dummy, y_dummy)
        test = (X_dummy, y_dummy) 
        
        train_res, test_res = hw_randomforests(train, test)
        
        self.assertEqual(len(train_res), 2)
        self.assertEqual(train_res[0], 0.0) 
        self.assertEqual(train_res[1], 0.0)    

    def test_rf_importance(self):
        """Preveri če je feature importance večji na dejansko pomembnemu featuru"""
        X = np.array([[1.0, 0.1], [1.0, 0.9], [5.0, 0.5], [5.0, 0.2]])
        y = np.array([0, 0, 1, 1])
        
        rf = RandomForest(rand=random.Random(42), n=10)
        model = rf.build(X, y)
        imps = model.importance()
        self.assertGreater(imps[0], imps[1])    

def hw_tree_full(train, test):
    x_train, y_train = train
    x_test, y_test = test

    tree = Tree(rand= random.Random(42), min_samples = 2)
    model = tree.build(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_error = np.mean(train_pred != y_train)
    test_error = np.mean(test_pred != y_test)

    train_se = np.sqrt((train_error * (1 - train_error)) / len(y_train))
    test_se = np.sqrt((test_error * (1 - test_error)) / len(y_test))
    
    return (train_error, train_se), (test_error, test_se)

def hw_randomforests(train, test):
    x_train, y_train = train
    x_test, y_test = test

    rf = RandomForest(rand = random.Random(42), n = 100)
    model = rf.build(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_error = np.mean(train_pred != y_train)
    test_error = np.mean(test_pred != y_test)

    train_se = np.sqrt((train_error * (1 - train_error)) / len(y_train))
    test_se = np.sqrt((test_error * (1 - test_error)) / len(y_test))
    
    return (train_error, train_se), (test_error, test_se)

def plot_missclasifications(train, test, max_trees = 100, step=5):
    x_train, y_train = train
    x_test, y_test = test

    n_values = np.arange(1,max_trees + 1, step)
    train_errors = []
    test_errors = []
    train_ses = []
    test_ses = []

    for n in n_values:
        rf = RandomForest(rand=random.Random(42), n= n)
        model = rf.build(x_train, y_train)

        print(f"Buildam drevo z {n} drevesi")

        train_pred = model.predict(x_train)
        test_pred = model.predict(x_test)

        train_error = np.mean(train_pred != y_train)
        test_error = np.mean(test_pred != y_test)

        train_se = np.sqrt((train_error * (1 - train_error)) / len(y_train))
        test_se = np.sqrt((test_error * (1 - test_error)) / len(y_test))

        train_errors.append(train_error)
        test_errors.append(test_error)
        train_ses.append(train_se)
        test_ses.append(test_se)

    plt.figure(figsize=(12,6))
    plt.errorbar(n_values, train_errors, yerr = train_ses, fmt="o", color="dodgerblue", ecolor ="skyblue", capsize = 4, label = "Train Error +- SE" )
    plt.errorbar(n_values, test_errors, yerr = test_ses, fmt="o", color="darkviolet", ecolor ="mediumorchid", capsize = 4, label = "Test Error +- SE")
    plt.title("Random forest Misclassification rate vs Size of forest")
    plt.xlabel("Number of trees (n)")
    plt.ylabel("Misclassification Rate")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    plt.savefig("misclassification.png")

def plot_importance(train):
    x_train, y_train = train
    n_feats = x_train.shape[1]
    #build normal rf:
    rf = RandomForest(rand=random.Random(42), n = 100)
    rf_model = rf.build(x_train, y_train)
    rf_importances = rf_model.importance()

    #scrambled data
    root_counts = np.zeros(n_feats)
    my_rand = random.Random(67) #six-seven
    
    for i in range(100):
        print(f"\r Training randomized tree {i + 1}/100...", end="")
        n_samples = len(x_train)
        indices = my_rand.choices(range(n_samples), k = n_samples)
        x_boot = x_train[indices]

        y_boot = y_train[indices].copy()
        my_rand.shuffle(y_boot)
        tree = Tree(rand = my_rand, get_candidate_columns=all_columns, min_samples=2)
        tree_model = tree.build(x_boot, y_boot)

        root_feat = tree_model.root.feature_id
        if root_feat is not None:
            root_counts[root_feat] +=1

    
    fig, ax1 = plt.subplots(figsize=(12,6))
    x_axis = range(n_feats) 

    ax1.set_xlabel("Feature index (Discrete Spectral Bins)")
    ax1.set_ylabel("RF permutation importance", color = "dodgerblue")
    ax1.bar(x_axis, rf_importances, width=1.0,  color = "dodgerblue", label = "Real RF Importance")
    ax1.tick_params(axis = "y", labelcolor = "dodgerblue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Root Split Count (Null Model)", color= "lightsalmon")
    ax2.bar(x_axis, root_counts, width=1, color = "lightsalmon", alpha = 0.8, label="Randomized root counts")
    ax2.tick_params(axis= "y", labelcolor = "lightsalmon")

    fig.tight_layout()
    plt.grid(True, linestyle = ":", alpha = 0.7)
    y1_min, y1_max = ax1.get_ylim()
    y2_max = ax2.get_ylim()[1]
    y2_min = y2_max * (y1_min / y1_max)
    ax2.set_ylim(y2_min, y2_max)
    plt.savefig("importance.png")
    plt.show()

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


if __name__ == "__main__":
    unittest.main(exit = False)

    try:
        train, test, legend = tki()
        print("Buildam drevo...")
        tree_train, tree_test = hw_tree_full(train, test)
        print("Drevo Rezultati:")
        print(f"    Train error: {tree_train[0]}    SE:{tree_train[1]}")
        print(f"    Test error:{tree_test[0]},   SE:{tree_test[1]} \n")

        print("Buildam RF...")
        rf_train, rf_test = hw_randomforests(train, test)
        print("RF Rezultati:")
        print(f"    Train error: {rf_train[0]}    SE:{rf_train[1]}")
        print(f"    Test error:{rf_test[0]},   SE:{rf_test[1]} \n")
    except FileNotFoundError:
        print("Ne najde tki filov")   

    #plot_missclasifications(train, test, max_trees = 100, step = 5)    
    #plot_importance(train)