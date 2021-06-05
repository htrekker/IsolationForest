import numpy as np
import math

# import json
from iTree import iTree
# from iTree import Node


import torch
import torch.nn.parameter as param

import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest


EULAR_GAMMA = 0.5772156649


class iForest(object):
    '''Naive implementation of IsolationForest'''

    def __init__(self, n_estimator=100, subsample_num=256):
        self.forest = []
        self.n_estimator = n_estimator
        self.subsample_num = subsample_num if subsample_num != -1 else 50

        # torch tensor parameters
        self.weights = param.Parameter(torch.ones(1, n_estimator))

    def sample(self, size, subsample_num):
        '''
        sample a subset from X

        ## Parameters:
            - X: full set of X
            - phi: the length of subset
        ## Return:
            - the subset of X, the length of which is phi
        '''
        sampled_idx = np.random.choice(np.arange(size), size=subsample_num)

        return sampled_idx

    def fit(self, X, y=None):
        '''
        Fit the IsolationForest

        ## Parameters:
            - X: train data
            - y: dont needed
        '''
        self.n = len(X)
        hlim = 8 if self.subsample_num == 256 else np.ceil(
            np.log2(self.subsample_num))  # calculate log_2{phi}

        for i in range(self.n_estimator):
            sampled_idx = self.sample(self.n, self.subsample_num)
            tree_i = iTree(hlim)
            tree_i.fit(X[sampled_idx])
            self.forest.append(tree_i)

        return self

    def predict(self, X, online=False):
        ts = []
        for i in range(self.n_estimator):
            tree_i = self.forest[i]
            length = tree_i.predict(X)
            ts.append(length)  # 1, len(X)
            # print(length[-20:])

        ts = np.stack(ts, axis=0)  # n_estimator, len(X)
        ts = np.transpose(ts, axes=(1, 0))  # n, n_estimator
        if online:
            ts = torch.Tensor(ts)
        scores = self.normalize_score(ts)
        return scores

    def normalize_score(self, es):
        '''
        '''
        def H(i):
            return math.log(i) + np.euler_gamma

        def C(n):
            '''
            estimate function to get c(\phi)
            '''
            if n > 2:
                return 2 * H(n - 1) - (2 * (n - 1) / n)
            return 1 if n == 2 else 0

        if type(es) is torch.Tensor:
            es = torch.Tensor(es) * self.weights
            expectations = torch.mean(es, dim=1)
            normalized_scores = torch.pow(torch.Tensor(
                [2]), -expectations / C(self.subsample_num))
        elif type(es) is np.ndarray:
            expectations = np.mean(es, axis=1)
            # print(expectations, C(self.subsample_num))
            normalized_scores = np.power(
                2, -expectations / C(self.subsample_num))

        return normalized_scores


def train_batch_stream(model, optim=None, epochs=30, dataloader=None):

    if optim is not None:
        optimizer = optim
    else:
        optimizer = torch.optim.SGD(
            (model.weights), lr=1e-4, momentum=0.99, weight_decay=3e-3)
    criterion = torch.nn.BCELoss()

    # because is stream data, the data only trained once
    for i, batch in enumerate(dataloader):
        X = batch['data'].float()

        optimizer.zero_grad()
        preds = model.predict(X)
        preds = (preds > 0.6).float()

        target = batch['label'].float()
        loss = criterion(preds, target)

        loss.backward()
        optimizer.step()


if __name__ == '__main__':
    uniform_X = np.random.normal(loc=3, scale=0.1, size=(100, 2))
    # print(uniform_X)
    normal_X = np.random.normal(loc=0.0, scale=1.0, size=(1000, 2))
    # print(normal_X)

    X = np.concatenate([uniform_X, normal_X], axis=0)

    clf = iForest(n_estimator=128)
    clf.fit(X)

    preds = clf.predict(X[:100, :], online=True)
    print(preds, type(preds))

    criterion = torch.nn.BCELoss()
    loss = criterion(preds, torch.ones(preds.size(0)))
    print(loss.item())
    loss.backward()

    preds = clf.predict(X[:100, :])
    print(preds, type(preds))

    # colors = ['red' if i < 100 else 'blue' for i in range(1100)]

    # plt.scatter(X[:, 0], X[:, 1], marker='.', c=colors)
    # plt.show()
    # # idx = np.arange(X.shape[0])
    # # np.random.shuffle(idx)
    # # X = X[idx]

    # clf = iForest(n_estimator=200)
    # clf.fit(X)

    # # print(X.shape)
    # abnormal_preds = clf.predict(X[:100, :])
    # normal_preds = clf.predict(X[100:, :])
    # print((abnormal_preds > 0.6).astype(np.float32).sum(),
    #       (normal_preds > 0.6).astype(np.float32).sum())
    # # print((preds < 0.7045).astype(np.float32).sum(), type(preds))

    # clf = IsolationForest(n_estimators=200)
    # clf.fit(X)
    # abnormal_preds = clf.predict(X[:100, :])
    # normal_preds = clf.predict(X[100:, :])
    # print((abnormal_preds < 0).astype(np.float32).sum(),
    #       (normal_preds < 0).astype(np.float32).sum())
