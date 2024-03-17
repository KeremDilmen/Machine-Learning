from collections import Counter

import numpy as np
from numpy import genfromtxt
import scipy.io
from scipy.stats import mode
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from graphviz import Digraph

import random
random.seed(246810)
np.random.seed(246810)

eps = 1e-5  # a small number


class DecisionTree:

    def __init__(self, max_depth=3, feature_labels=None, max_features=None):
        self.max_depth = max_depth
        self.features = feature_labels
        self.m = len(feature_labels) if max_features is None else max_features
        self.left, self.right = None, None  # for non-leaf nodes
        self.split_idx, self.thresh = None, None  # for non-leaf nodes
        self.data, self.pred = None, None  # for leaf nodes
        self.labels = None

    @staticmethod
    def entropy(y):
        num_labels = y.shape[0]
        if num_labels == 0:
            return 0

        p0 = np.where(y==0)[0].shape[0] / num_labels
        p1 = np.where(y==1)[0].shape[0] / num_labels

        if p0 < eps or p1 < eps:
            return 0
        
        entropy_0 = -p0 * np.log2(p0)
        entropy_1 = -p1 * np.log2(p1)
        return entropy_0 + entropy_1

    @staticmethod
    def information_gain(X, y, thresh):
        num_labels = y.shape[0]
        h_s = DecisionTree.entropy(y)
        y0, y1 = y[np.where(X<thresh)[0]], y[np.where(X>=thresh)[0]]
        p0, p1 = y0.shape[0] / num_labels, y1.shape[0] / num_labels
        h_after = p0 * DecisionTree.entropy(y0) + p1 * DecisionTree.entropy(y1)
        return h_s - h_after

    @staticmethod
    def gini_impurity(y):
        num_labels = y.shape[0]
        if num_labels == 0:
            return 0

        p0 = np.where(y==0)[0].shape[0] / num_labels
        p1 = np.where(y==1)[0].shape[0] / num_labels
        if p0 < eps or p1 < eps:
            return 0
        return 1 - (p0**2) - (p1**2)

    @staticmethod
    def gini_purification(X, y, thresh):
        num_labels = y.shape[0]
        gini_s = DecisionTree.gini_impurity(y)
        y0, y1 = y[np.where(X<thresh)[0]], y[np.where(X>=thresh)[0]]
        p0, p1 = y0.shape[0] / num_labels, y1.shape[0] / num_labels
        gini_after = p0 * DecisionTree.gini_impurity(y0) + p1 * DecisionTree.gini_impurity(y1)
        return gini_s - gini_after


    def split(self, X, y, idx, thresh):
        X0, idx0, X1, idx1 = self.split_test(X, idx=idx, thresh=thresh)
        y0, y1 = y[idx0], y[idx1]
        return X0, y0, X1, y1

    def split_test(self, X, idx, thresh):
        idx0 = np.where(X[:, idx] < thresh)[0]
        idx1 = np.where(X[:, idx] >= thresh)[0]
        X0, X1 = X[idx0, :], X[idx1, :]
        return X0, idx0, X1, idx1

    def fit(self, X, y):
        if self.max_depth == 0:
            self.data = X
            self.labels = y
            self.pred = mode(y).mode.astype(int)
        else:
            feature_indices = np.random.choice(np.arange(X.shape[1]), self.m, replace=False)
            
            best_gain = -np.inf
            for i in feature_indices:
                vals = X[:, i]
                thresholds = np.linspace(np.min(vals) + eps, np.max(vals) - eps, 30)
                for thresh in thresholds:
                    curr_gain = DecisionTree.information_gain(vals, y, thresh)
                    if curr_gain > best_gain:
                        best_gain = curr_gain
                        self.split_idx = i
                        self.thresh = thresh
        
            X0, y0, X1, y1 = self.split(X, y, self.split_idx, self.thresh)

            if len(y0) == 0 or len(y1) == 0:
                self.max_depth = 0
                self.data = X
                self.labels = y
                self.pred = mode(y).mode.astype(int)
                return

            self.left = DecisionTree(self.max_depth-1, self.features, self.m)
            self.right = DecisionTree(self.max_depth-1, self.features, self.m)
            self.left.fit(X0, y0)
            self.right.fit(X1, y1)

    def predict(self, X):
        if self.max_depth == 0:
            return np.full(X.shape[0], self.pred)
        else:
            X0, idx0, X1, idx1 = self.split_test(X, self.split_idx, self.thresh)
            predicted = np.zeros(X.shape[0])
            predicted[idx0], predicted[idx1] = self.left.predict(X0), self.right.predict(X1)
            return predicted.astype(int)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)
    
    def visualize_tree(self, dot=None, node_id=0):
        if dot is None:
            dot = Digraph(comment='Decision Tree', format='png')
            dot.node(str(node_id), label=str(self) if self.max_depth == 0 else f"{self.features[self.split_idx]} < {self.thresh:.5f}")
        
        current_id = node_id
        left_id = node_id * 2 + 1
        right_id = node_id * 2 + 2
        
        if self.left is not None:
            dot.node(str(left_id), label=f"{class_names[self.left.pred]} ({self.left.labels.size})" if self.left.max_depth == 0 
                     else f"{self.features[self.left.split_idx]} < {self.left.thresh:.5f}")
            dot.edge(str(current_id), str(left_id), label="True")
            self.left.visualize_tree(dot, left_id)

        if self.right is not None:
            dot.node(str(right_id), label=f"{class_names[self.right.pred]} ({self.right.labels.size})" if self.right.max_depth == 0 
                     else f"{self.features[self.right.split_idx]} < {self.right.thresh:.5f}")
            dot.edge(str(current_id), str(right_id), label="False")
            self.right.visualize_tree(dot, right_id)
        
        return dot

    def __repr__(self):
        if self.max_depth == 0:
            return "%s (%s)" % (self.pred, self.labels.size)
        else:
            return "[%s < %s: %s | %s]" % (self.features[self.split_idx],
                                           self.thresh, self.left.__repr__(),
                                           self.right.__repr__())


class BaggedTrees:

    def __init__(self, params=None, n=200):
        if params is None:
            params = {}
        self.params = params
        self.n = n
        self.decision_trees = [DecisionTree(**self.params) for _ in range(self.n)]

    def fit(self, X, y):
        for tree in self.decision_trees:
            idx = np.random.choice(np.arange(X.shape[0]), X.shape[0], replace=True)
            tree.fit(X[idx, :], y[idx])


    def predict(self, X):
        predicted = np.array([tree.predict(X) for tree in self.decision_trees])
        return mode(predicted).mode.astype(int)
    
    def score(self, X, y):
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


class RandomForest(BaggedTrees):

    def __init__(self, params=None, n=200, m=1):
        if params is None:
            params = {}
        params['max_features'] = m
        self.m = m
        super().__init__(params=params, n=n)

def preprocess(data, min_freq=10, onehot_cols=[]):

    # Temporarily assign -1 to missing data
    data[data == b''] = '-1'

    # Hash the columns (used for handling strings)
    onehot_encoding = []
    onehot_features = []
    economic_stat = {b'1.0': b'upper', b'2.0': b'middle', b'3.0': b'lower'}
    for col in onehot_cols:
        counter = Counter(data[:, col])
        for term in counter.most_common():
            if term[0] == b'-1':
                continue
            if term[-1] <= min_freq:
                break
            if term[0] in economic_stat:
                onehot_features.append(economic_stat[term[0]])
            else:
                onehot_features.append(term[0])
            onehot_encoding.append((data[:, col] == term[0]).astype(float))
        data[:, col] = '0'
    onehot_encoding = np.array(onehot_encoding).T
    data = np.hstack(
        [np.array(data, dtype=float),
         np.array(onehot_encoding)])

    # Replace missing data with the mode value. We use the mode instead of
    # the mean or median because this makes more sense for categorical
    # features such as gender or cabin type, which are not ordered.
    imputer = SimpleImputer(missing_values=-1, strategy='most_frequent')
    data = imputer.fit_transform(data)
    return data, onehot_features

if __name__ == "__main__":
    dataset = "titanic"
    #dataset = "spam"

    if dataset == "titanic":
        # Load titanic data
        path_train = 'datasets/titanic/titanic_training.csv'
        data = genfromtxt(path_train, delimiter=',', dtype=None)
        path_test = 'datasets/titanic/titanic_testing_data.csv'
        test_data = genfromtxt(path_test, delimiter=',', dtype=None)
        y = data[1:, 0] 
        class_names = ["Died", "Survived"]

        labeled_idx = np.where(y != b'')[0]
        y = np.array(y[labeled_idx], dtype=float).astype(int)
        X, onehot_features = preprocess(data[1:, 1:], onehot_cols=[0, 1, 5, 7, 8])
        X = X[labeled_idx, :]
        Z, _ = preprocess(test_data[1:, :], onehot_cols=[0, 1, 5, 7, 8])
        assert X.shape[1] == Z.shape[1]
        features = list(data[0, 1:]) + onehot_features

        ## params for titanic ##
        max_depth = 5
        N = 100
        m = X.shape[1]

    elif dataset == "spam":
        features = [
            "pain", "private", "bank", "money", "drug", "spam", "prescription",
            "creative", "height", "featured", "differ", "width", "other",
            "energy", "business", "message", "volumes", "revision", "path",
            "meter", "memo", "planning", "pleased", "record", "out",
            "semicolon", "dollar", "sharp", "exclamation", "parenthesis",
            "square_bracket", "ampersand"
        ]
        assert len(features) == 32

        # Load spam data
        path_train = 'datasets/spam_data/spam_data.mat'
        data = scipy.io.loadmat(path_train)
        X = data['training_data']
        y = np.squeeze(data['training_labels'])
        Z = data['test_data']
        class_names = ["Ham", "Spam"]

        ## params for spam ##
        max_depth = 14
        N = 50
        m = X.shape[1]

    else:
        raise NotImplementedError("Dataset %s not handled" % dataset)

    print("Features", features)
    print("Train/test size", X.shape, Z.shape)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    ## Decision Tree ##
    dt = DecisionTree(max_depth=max_depth, feature_labels=features)
    dt.fit(X_train, y_train)
    training_accuracy = dt.score(X_train, y_train)
    validation_accuracy = dt.score(X_val, y_val)
    print(training_accuracy)
    print(validation_accuracy)



    ## Random Forest ##
    params = {'max_depth': max_depth, 'feature_labels':features}
    rf = RandomForest(params = params, n=N, m=m)
    rf.fit(X_train, y_train)
    training_accuracy = rf.score(X_train, y_train)
    validation_accuracy = rf.score(X_val, y_val)
    print(training_accuracy)
    print(validation_accuracy)


    if dataset == 'titanic':

        ## titanic tree visualization for depth 3 ##
        dt = DecisionTree(max_depth=3, feature_labels=features)
        dt.fit(X_train, y_train)
        dot = dt.visualize_tree()
        dot.render('titanic_decision_tree')
    elif dataset == 'spam':

        ## visualize spam tree ##
        print(X_train[:2, :])
        print(y_train[:2])
        dot = dt.visualize_tree()
        dot.render('spam_decision_tree')

        ## Validation Accuracy vs Depth Plot ##
        depths = np.arange(1, 41)
        train_accuracies = []
        val_accuracies = []
        for depth in depths:
            dt = DecisionTree(max_depth=depth, feature_labels=features)
            dt.fit(X_train, y_train)
            train_accuracies.append(dt.score(X_train, y_train))
            val_accuracies.append(dt.score(X_val, y_val))
        
        plt.plot(depths, train_accuracies, label='Training Accuracy')
        plt.plot(depths, val_accuracies, label="Validation Accuracy")
        plt.xlabel("Depth")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.show()
