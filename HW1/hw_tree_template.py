import csv
import numpy as np


def all_columns(X, rand):
    return range(X.shape[1])


def random_sqrt_columns(X, rand):
    c =  1
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
        self.rand = rand if rand is not None else random.Random()  # for replicability
        self.get_candidate_columns = get_candidate_columns  # needed for random forests
        self.min_samples = min_samples

    def build(self, X, y):
        
        def buildTree(x_set, y_set):
            n_samples = len(y_set)
            if n_samples < self.min_samples or len(np.unique(y_set)) == 1:
                uni_class, counts = np.unique(y_set, return_count = True)
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
                uni_class, counts = np.unique(y_set, return_count = True)
                major_class = uni_class[np.argmax(counts)]
                return Node(value = major_class)

        root_node = buildTree(X,y)
        return TreeModel(root_node)  # return an object that can do prediction


class TreeModel:

    def __init__(self, root):
        self.root = root

    def traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_id] <= node.threshold:
            return self.traverse_tree(x, node.left)
        else:
            return self.traverse_tree(x, node.riht)        

    def predict(self, X):
        predictions = []
        for sample in X:
            predictions.append(self.traverse_tree(sample, self.root))
        return np.array(predictions)
                


class RandomForest:

    def __init__(self, rand=None, n=50):
        self.n = n
        self.rand = rand
        self.rftree = Tree(rand = self.rand,
                           get_candidate_columns=random_sqrt_columns,
                            min_samples=2 )

    def build(self, X, y):
        models = []
        n_samples = len(X)

        #bootstrap
        for _ in range(self.n):
            random_indices = self.rand.choices(range(n_samples), k= n_samples)

            x_boot = X[random_indices]
            y_boot = y[random_indices]

            model = self.rftree.build(x_boot, y_boot)
            models.append(model)

        return RFModel(models)  # return an object that can do prediction


class RFModel:

    def __init__(self, models):
        self.models = models

    def predict(self, X):
        all_predicts = np.array([model.predict(X) for model in self.models])

        predictions = []

        for i in range(len(X)):
            sample_result = all_predicts[:, i]

            classes, counts = np.unique(sample_result, return_counts = True)
            major_class = classes[np.argmax(counts)]

            predictions.append(major_class)

        return predictions

    def importance(self):
        imps = np.zeros(self.X.shape[1])
        # ...
        return imps


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
    learn, test, legend = tki()

    print("full", hw_tree_full(learn, test))
    print("random forests", hw_randomforests(learn, test))
