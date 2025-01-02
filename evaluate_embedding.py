import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import average_precision_score, ndcg_score
# import seaborn as sns

import warnings

# Existing LogReg class remains unchanged
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret

# Existing classification functions remain unchanged
def logistic_classify(x, y):

    nb_classes = np.unique(y).shape[0]
    xent = nn.CrossEntropyLoss()
    hid_units = x.shape[1]

    accs = []
    accs_val = []
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    for train_index, test_index in kf.split(x, y):

        # test
        train_embs, test_embs = x[train_index], x[test_index]
        train_lbls, test_lbls= y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        test_embs, test_lbls= torch.from_numpy(test_embs).cuda(), torch.from_numpy(test_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(test_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == test_lbls).float() / test_lbls.shape[0]
        accs.append(acc.item())

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if i not in test_index]

        train_embs, val_embs = x[train_index], x[test_index]
        train_lbls, val_lbls = y[train_index], y[test_index]

        train_embs, train_lbls = torch.from_numpy(train_embs).cuda(), torch.from_numpy(train_lbls).cuda()
        val_embs, val_lbls = torch.from_numpy(val_embs).cuda(), torch.from_numpy(val_lbls).cuda()


        log = LogReg(hid_units, nb_classes)
        log.cuda()
        opt = torch.optim.Adam(log.parameters(), lr=0.01, weight_decay=0.0)

        best_val = 0
        test_acc = None
        for it in range(100):
            log.train()
            opt.zero_grad()

            logits = log(train_embs)
            loss = xent(logits, train_lbls)

            loss.backward()
            opt.step()

        logits = log(val_embs)
        preds = torch.argmax(logits, dim=1)
        acc = torch.sum(preds == val_lbls).float() / val_lbls.shape[0]
        accs_val.append(acc.item())

    return np.mean(accs_val), np.mean(accs)

# Existing SVC, RandomForest, LinearSVC classification functions remain unchanged
def svc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        # x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1)
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(SVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = SVC(C=10)
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)

def randomforest_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(test_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'n_estimators': [100, 200, 500, 1000]}
            classifier = GridSearchCV(RandomForestClassifier(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = RandomForestClassifier()
        classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    ret = np.mean(accuracies)
    return np.mean(accuracies_val), ret

def linearsvc_classify(x, y, search):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
    accuracies = []
    accuracies_val = []
    for train_index, test_index in kf.split(x, y):

        # test
        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies.append(accuracy_score(y_test, classifier.predict(x_test)))

        # val
        val_size = len(test_index)
        test_index = np.random.choice(train_index, val_size, replace=False).tolist()
        train_index = [i for i in train_index if not i in test_index]

        x_train, x_test = x[train_index], x[test_index]
        y_train, y_test = y[train_index], y[test_index]
        if search:
            params = {'C':[0.001, 0.01,0.1,1,10,100,1000]}
            classifier = GridSearchCV(LinearSVC(), params, cv=5, scoring='accuracy', verbose=0)
        else:
            classifier = LinearSVC(C=10)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            classifier.fit(x_train, y_train)
        accuracies_val.append(accuracy_score(y_test, classifier.predict(x_test)))

    return np.mean(accuracies_val), np.mean(accuracies)
# New function for link prediction evaluation
def link_prediction_evaluate(embeddings, edge_index, labels, neg_edge_index, neg_samples=9, metrics='auc'):
    """
    Evaluate embeddings using link prediction task with specified metrics.

    Args:
        embeddings (numpy.ndarray): Node embeddings.
        edge_index (numpy.ndarray): Positive edge indices (shape: [2, num_edges]).
        labels (numpy.ndarray): Labels for edges (1 for positive edge).
        neg_edge_index (numpy.ndarray): Negative edge indices (shape: [2, num_neg_edges]).
        neg_samples (int): Number of negative samples per positive sample.
        metrics (str): Evaluation metrics ('auc' or 'ranking').

    Returns:
        results (dict): Evaluation results.
    """
    if metrics == 'auc':
        # Existing AUC evaluation
        # Prepare edge features by concatenating embeddings of node pairs
        edge_embeddings = np.abs(embeddings[edge_index[0]] - embeddings[edge_index[1]])
        neg_edge_embeddings = np.abs(embeddings[neg_edge_index[0]] - embeddings[neg_edge_index[1]])

        # Combine positive and negative samples
        X = np.vstack([edge_embeddings, neg_edge_embeddings])
        y = np.hstack([np.ones(edge_embeddings.shape[0]), np.zeros(neg_edge_embeddings.shape[0])])

        # Train logistic regression classifier
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(random_state=0, max_iter=1000)
        clf.fit(X, y)

        # Predict probabilities
        y_pred = clf.predict_proba(X)[:, 1]

        # Calculate evaluation metrics
        auc = roc_auc_score(y, y_pred)
        ap = average_precision_score(y, y_pred)

        print(f"Link Prediction AUC: {auc:.4f}, AP: {ap:.4f}")
        return {'AUC': auc, 'AP': ap}

    elif metrics == 'ranking':
        # MRR, NDCG, H1 evaluation
        num_samples = edge_index.shape[1]
        labels = np.zeros(neg_samples + 1)
        labels[0] = 1  # The first one is the positive sample

        pos_score = np.empty([0])
        neg_score = np.empty([0, neg_samples])

        # For simplicity, assume neg_edge_index is organized per positive sample
        # i.e., for each positive edge, there are `neg_samples` negative edges
        # Adjust neg_edge_index accordingly if necessary

        for i in range(num_samples):
            # Positive sample
            u, v = edge_index[0, i], edge_index[1, i]
            a = embeddings[u]
            b = embeddings[v]
            pos = torch.sigmoid(torch.sum(a * b)).item()
            pos_score = np.append(pos_score, pos)

            # Negative samples
            neg_edges = neg_edge_index[:, i * neg_samples: (i + 1) * neg_samples]
            neg_embeddings = embeddings[neg_edges[1]]
            neg = torch.sigmoid(np.dot(a, neg_embeddings.T))
            neg_score = np.vstack([neg_score, neg])

        pred_list = np.hstack([pos_score.reshape(-1, 1), neg_score])

        sum_ndcg = 0
        sum_mrr = 0
        sum_hit1 = 0

        for i in range(num_samples):
            true = pred_list[i, 0]
            sort_list = np.sort(pred_list[i])[::-1]

            rank = np.where(sort_list == true)[0][0] + 1
            sum_mrr += (1 / rank)

            if pred_list[i, 0] == np.max(pred_list[i]):
                sum_hit1 += 1

            NDCG = ndcg_score([labels], [pred_list[i]])
            sum_ndcg += NDCG

        H1 = sum_hit1 / num_samples
        MRR = sum_mrr / num_samples
        NDCG = sum_ndcg / num_samples

        print(f"Link Prediction MAP/MRR: {MRR:.4f}, NDCG: {NDCG:.4f}, H1: {H1:.4f}")
        return {'MRR': MRR, 'NDCG': NDCG, 'H1': H1}

    else:
        raise ValueError("Invalid metrics. Choose 'auc' or 'ranking'.")

def evaluate_embedding(embeddings, labels=None, edge_index=None, edge_labels=None, neg_edge_index=None, task='node', metrics='auc', search=True):
    """
    Evaluate embeddings using node classification or link prediction.

    Args:
        embeddings (numpy.ndarray): Embeddings to evaluate.
        labels (numpy.ndarray): Labels for nodes (for node classification).
        edge_index (numpy.ndarray, optional): Edge indices (required for link prediction).
        edge_labels (numpy.ndarray, optional): Labels for edges (required for link prediction).
        neg_edge_index (numpy.ndarray, optional): Negative edge indices (required for ranking metrics).
        task (str): 'node' for node classification, 'link' for link prediction.
        metrics (str): Evaluation metrics for link prediction ('auc' or 'ranking').

    Returns:
        results (dict): Evaluation results.
    """
    if task == 'node':
        # Existing node classification evaluation
        labels = preprocessing.LabelEncoder().fit_transform(labels)
        x, y = np.array(embeddings), np.array(labels)

        acc = 0
        acc_val = 0

        _acc_val, _acc = svc_classify(x, y, search)
        acc_val = _acc_val
        acc = _acc

        print(f"Node Classification Accuracy (Val): {acc_val:.4f}, Accuracy (Test): {acc:.4f}")
        return {'Acc_val': acc_val, 'Acc_test': acc}

    elif task == 'link':
        # Link prediction evaluation
        if edge_index is None or edge_labels is None:
            raise ValueError("Edge indices and edge labels are required for link prediction task.")

        x = np.array(embeddings)

        if metrics == 'auc':
            # Use AUC evaluation
            results = link_prediction_evaluate(x, edge_index, edge_labels, neg_edge_index, metrics='auc')
            return results
        elif metrics == 'ranking':
            # Use ranking evaluation (MRR, NDCG, H1)
            if neg_edge_index is None:
                raise ValueError("Negative edge indices are required for ranking metrics.")
            results = link_prediction_evaluate(x, edge_index, edge_labels, neg_edge_index, metrics='ranking')
            return results
        else:
            raise ValueError("Invalid metrics. Choose 'auc' or 'ranking'.")

    else:
        raise ValueError("Task must be 'node' or 'link'.")

# The rest of the code remains unchanged

'''
if __name__ == '__main__':
    evaluate_embedding('./data', 'ENZYMES', np.load('tmp/emb.npy'))
'''
