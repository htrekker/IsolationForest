import numpy as np
import json


class Node(object):
    '''Tree Node'''

    def __init__(self, feature_name, split_value):
        self.feature_name = feature_name
        self.split_value = split_value
        self.left = None
        self.right = None


class iTree(object):
    def __init__(self, hlim):
        self.root = None
        self.params = {}
        self.hlim = hlim

    def build_tree(self, X, e, limit):
        if e >= limit or len(X) <= 1:
            return None
        else:
            choosed_feature = np.random.randint(0, X.shape[1])

            t = X[:, choosed_feature]
            max_val, min_val = t.max(), t.min()

            if max_val == min_val:
                return None

            split_point = np.random.rand() * (max_val - min_val) + min_val
            X_left = X[t < split_point, :]
            X_right = X[t >= split_point, :]

            cur = Node(feature_name=choosed_feature, split_value=split_point)
            if self.root is None:
                self.root = cur

            cur.left = self.build_tree(X_left, e + 1, limit)
            cur.right = self.build_tree(X_right, e + 1, limit)

            return cur

    def fit(self, X, y=None):
        self.subsample_size = len(X)
        self.build_tree(X, 0, self.hlim)

    def get_path_length(self, x, tree, e):
        '''
        Get travel length of a given instances in the iTree.

        ## Parameters:
            - x: the instance vector (variable name is in lower case)
            - e: the current depth of the instance
        ## Return:
            - the travel path
        '''
        if tree is None or e >= self.hlim:
            return e

        feature_name = tree.feature_name
        split_val = tree.split_value

        next = tree.left if x[feature_name] < split_val else tree.right

        return self.get_path_length(x, next, e + 1)

    def predict(self, X):
        '''
        get the travel path of each row in given X.

        ## Parameters:
            - X: predict data
        ## Return:
            - lengths: the travel lengths of the data in current tree
                        (given in numpy.ndarray format)
        '''
        if len(X.shape) > 2:
            raise Exception(
                '2d matrix expceted, but %dd was given.' % len(X.shape))

        lengths = []
        for row in X:
            e = self.get_path_length(row, self.root, 0)
            lengths.append(e)

        return np.array(lengths)

    def dump_tree(self, fp=None):
        '''
        dump a tree into the given fp. (default is None)
        if the fp is not given, the parameters will be saved into self.params

        Parameters:
            - tree: the node
            - fp: the file descriptor. (default is None)
        Returns:
            no return
        '''
        def dump_node(node):
            params = {}
            if node is not None:
                params['feature_name'] = node.feature_name
                params['split_value'] = node.split_value

                params['left'] = dump_node(node.left)
                params['right'] = dump_node(node.right)

            return params

        self.params = dump_node(self.root)
        if fp is not None:
            json.dump(self.params, fp=fp, indent=4)


if __name__ == '__main__':
    uniform_X = np.random.uniform(low=0.0, high=1.0, size=(10, 10))
    normal_X = np.random.normal(loc=0.0, scale=1.0, size=(1000, 10))

    X = np.concatenate([uniform_X, normal_X], axis=0)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    X = X[idx]

    tree = iTree(hlim=15)
    tree.fit(X)

    preds = tree.predict(X[:100, :])
    print(preds, type(preds))
