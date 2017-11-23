import os
import tempfile
from optparse import OptionParser

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..util import file_handling as fh
from ..models import evaluation, calibration


class DAN:
    """
    Multilayer perceptron (representing documents as weighted sums of word vectors)
    """
    def __init__(self, dimensions, output_dir=None, name='model', pos_label=1, objective='f1', init_emb=None, update_emb=False):
        self._model_type = 'DAN'
        self._dimensions = dimensions[:]
        self._loss_function = 'log'
        self._nonlinearity = 'relu'
        self._loss_function = 'log'
        self._reg_strength = 0
        self._penalty = None
        self._dropout_prob = 0.0
        self._n_classes = None
        self._init_emb = None
        self._update_emb = update_emb

        if output_dir is None:
            self._output_dir = tempfile.gettempdir()
        else:
            self._output_dir = output_dir
        self._name = name
        self._pos_label = pos_label
        self._objective = objective
        self._train_f1 = None
        self._train_acc = None
        self._dev_f1 = None
        self._dev_acc = None
        self._dev_acc_cfm = None
        self._dev_pvc_cfm = None


        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None

    def get_model_type(self):
        return self._model_type

    def get_loss_function(self):
        return self._loss_function

    def set_model(self, model, train_proportions, n_classes):
        self._train_proportions = train_proportions
        self._n_classes = n_classes
        if model is None:
            self._model = None
        else:
            self._model = model

    def fit(self, X_train, Y_train, X_dev, Y_dev, train_weights=None, dev_weights=None, seed=None, init_lr=1e-4, min_epochs=2, max_epochs=100, patience=8, tol=1e-4, early_stopping=True, **kwargs):
        """
        Fit a classifier to data
        """
        _, n_classes = Y_train.shape
        self._n_classes = n_classes

        n_train, _ = X_train.shape
        n_dev, _ = X_train.shape
        Y_list_train = Y_train.argmax(axis=1)
        Y_list_dev = Y_dev.argmax(axis=1)

        # store the proportion of class labels in the training data
        if train_weights is None:
            class_sums = np.sum(Y_train, axis=0)
        else:
            class_sums = np.dot(train_weights, Y_train) / train_weights.sum()
        self._train_proportions = (class_sums / float(class_sums.sum())).tolist()

        # if there is only a single type of label, make a default prediction
        train_labels = np.argmax(Y_train, axis=1)
        if np.max(self._train_proportions) == 1.0:
            self._model = None
        else:
            #model_filename = os.path.join(self._output_dir, self._name + '.ckpt')
            self._model = torchDAN(self._dimensions, init_emb=self._init_emb, update_emb=self._update_emb)
            best_model = torchDAN(self._dimensions, init_emb=self._init_emb, update_emb=self._update_emb)
            # train model

            # set the initial embeddings:


            criterion = nn.CrossEntropyLoss()
            grad_params = filter(lambda p: p.requires_grad, self._model.parameters())
            optimizer = optim.SGD(grad_params, lr=0.1, momentum=0.9)

            epoch = 0
            best_dev_acc = 0.0
            while epoch < max_epochs:
                running_loss = 0.0
                count = 0
                for i in range(n_train):
                    X_i_list = X_train[i, :].nonzero()[1].tolist()
                    X_i_array = np.array(X_i_list, dtype=np.int).reshape(1, len(X_i_list))
                    X_i = Variable(torch.LongTensor(X_i_array))
                    y_i = Variable(torch.from_numpy(Y_list_train[i:i+1]))

                    optimizer.zero_grad()
                    outputs = self._model(X_i)

                    loss = criterion(outputs, y_i)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.data[0]
                    count += 1
                    if i % 500 == 0:
                        print("%d %d %0.4f" % (epoch, i, running_loss / count))

                dev_acc = 0.0
                for i in range(n_dev):
                    X_i_list = X_dev[i, :].nonzero()[1].tolist()
                    X_i_array = np.array(X_i_list, dtype=np.int).reshape(1, len(X_i_list))
                    X_i = Variable(torch.LongTensor(X_i_array))
                    outputs = self._model(X_i)
                    dev_acc += Y_list_dev[i] == outputs.data.numpy().argmax()
                dev_acc /= n_dev
                print("epoch %d: dev acc = %0.4f" % (epoch, dev_acc))

                if dev_acc > best_dev_acc:
                    print("Updating best model")
                    model_params = list(self._model.parameters())
                    best_model_params = list(best_model.parameters())
                    for i, p in enumerate(model_params):
                        best_model_params[i].data[:] = p.data[:]
                    best_dev_acc = dev_acc

        """
        # do a quick evaluation and store the results internally
        train_pred = self.predict(X_train)
        self._train_acc = evaluation.acc_score(train_labels, train_pred, n_classes=n_classes, weights=train_weights)
        self._train_f1 = evaluation.f1_score(train_labels, train_pred, n_classes=n_classes, pos_label=self._pos_label, weights=train_weights)

        if X_dev is not None and Y_dev is not None:
            dev_labels = np.argmax(Y_dev, axis=1)
            dev_pred = self.predict(X_dev)
            dev_pred_probs = self.predict_probs(X_dev)
            self._dev_acc = evaluation.acc_score(dev_labels, dev_pred, n_classes=n_classes, weights=dev_weights)
            self._dev_f1 = evaluation.f1_score(dev_labels, dev_pred, n_classes=n_classes, pos_label=self._pos_label, weights=dev_weights)
            self._dev_acc_cfm = calibration.compute_acc(dev_labels, dev_pred, n_classes, weights=dev_weights)
            self._dev_pvc_cfm = calibration.compute_pvc(dev_labels, dev_pred, n_classes, weights=dev_weights)
            if self._n_classes == 2:
                self._venn_info = np.vstack([Y_dev[:, 1], dev_pred_probs[:, 1], dev_weights]).T
                assert self._venn_info.shape == (len(dev_labels), 3)
        """

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model is None:
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * np.argmax(self._train_proportions)
        else:
            return self._model.predict(X)

    def predict_probs(self, X, verbose=False):
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model is None:
            default = np.argmax(self._train_proportions)
            full_probs[:, default] = 1.0
            return full_probs
        else:
            # otherwise, get probabilities from the model
            model_probs = self._model.predict_probs(X, verbose=verbose)
            return model_probs

    def predict_proportions(self, X=None, weights=None, do_cfm=False, do_platt=False):
        # TODO: fix this to be the same as other models
        pred_probs = self.predict_probs(X)
        predictions = np.argmax(pred_probs, axis=1)
        if do_cfm:
            if self._n_classes == 2:
                acc = calibration.apply_acc_binary(predictions, self._dev_acc_cfm, weights)
            else:
                acc = calibration.apply_acc_bounded_lstsq(predictions, self._dev_acc_cfm)
            pvc = calibration.apply_pvc(predictions, self._dev_pvc_cfm, weights)
            return acc, pvc
        else:
            cc = calibration.cc(predictions, self._n_classes, weights)
            pcc = calibration.pcc(pred_probs, weights)
            return cc, pcc

    def get_penalty(self):
        return self._penalty

    def get_reg_strength(self):
        return self._reg_strength

    def get_n_classes(self):
        return self._n_classes

    def get_train_proportions(self):
        return self._train_proportions

    def get_active_classes(self):
        if self._model is None:
            return []
        else:
            return range(self._n_classes)

    def get_default(self):
        return np.argmax(self._train_proportions)

    def get_col_names(self):
        return None

    """
    def get_coefs(self, target_class=0):
        coefs = zip(self._col_names, np.zeros(len(self._col_names)))
        if self._model is not None:
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    coefs = zip(self._col_names, self._model.coef_[i])
                    break
        return coefs
    """

    """
    def get_intercept(self, target_class=0):
        # if we've saved a default value, there are no intercepts
        intercept = 0
        if self._model is not None:
            # otherwise, see if the model an intercept for this class
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    intercept = self._model.intercept_[i]
                    break
        return intercept
    """

    def get_model_size(self):
        if self._model is None:
            return 0
        else:
            return self._model.get_n_params()

    def get_name(self):
        return self._name

    def get_dimensions(self):
        return self._dimensions[:]

    def save(self):
        output = {'model_type': self.get_model_type(),
                  'name': self.get_name(),
                  'dimensions': self.get_dimensions(),
                  'loss_function': self._loss_function,
                  'nonlinearity': self._nonlinearity,
                  'reg_strength': self.get_reg_strength(),
                  'penalty': self.get_penalty(),
                  'pos_label': self._pos_label,
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  'train_f1': self._train_f1,
                  'train_acc': self._train_acc,
                  'dev_f1': self._dev_f1,
                  'dev_acc': self._dev_acc
                  }
        fh.write_to_json(output, os.path.join(self._output_dir, self._name + '_metadata.json'), sort_keys=False)
        np.savez(os.path.join(self._output_dir, self._name + '_dev_info.npz'), acc_cfm=self._dev_acc_cfm, pvc_cfm=self._dev_pvc_cfm, venn_info=self._venn_info)


def load_from_file(model_dir, name):
    input = fh.read_json(os.path.join(model_dir, name + '_metadata.json'))
    dimensions = input['dimensions']
    penalty = input['penalty']
    reg_strength = float(input['reg_strength'])
    pos_label = int(input['pos_label'])
    n_classes = int(input['n_classes'])
    train_proportions = input['train_proportions']
    loss_function = input['loss_function']
    nonlinearity = input['nonlinearity']

    classifier = MLP(dimensions, loss_function, nonlinearity, penalty, reg_strength, model_dir, name=name, pos_label=pos_label)
    model_filename = os.path.join(model_dir, name + '.ckpt')
    model = tf_MLP(dimensions, model_filename, loss_function, penalty, reg_strength, nonlinearity, pos_label=pos_label)
    classifier.set_model(model, train_proportions, n_classes)
    dev_info = np.load(os.path.join(model_dir, name + '_dev_info.npz'))
    classifier._dev_acc_cfm = dev_info['acc_cfm']
    classifier._dev_pvc_cfm = dev_info['pvc_cfm']
    classifier._venn_info = dev_info['venn_info']
    return classifier


class torchDAN(nn.Module):

    def __init__(self, dims, init_emb=None, update_emb=False):
        super(torchDAN, self).__init__()
        assert len(dims) >= 3
        self.n_layers = len(dims)-2
        self.emb = nn.Embedding(dims[0], dims[1])
        if not update_emb:
            self.emb.weight.requires_grad = False
        if init_emb is not None:
            self.emb.weight.data.copy_(torch.from_numpy(init_emb))
        self.layers = []
        for layer in range(1, len(dims)-1):
            self.layers.append(nn.Linear(dims[layer], dims[layer+1]))

    def forward(self, X):
        h = torch.mean(self.emb(X), dim=1)
        for i in range(self.n_layers):
            layer = self.layers[i]
            h = layer(F.relu(h))
        return h


