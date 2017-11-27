import os
import sys
import tempfile
from optparse import OptionParser

import numpy as np
from scipy.special import expit
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
    def __init__(self, dimensions, alpha=0.0, output_dir=None, name='model', pos_label=1, objective='f1', init_emb=None, update_emb=False):
        self._model_type = 'DAN'
        self._penalty = 'l2'
        self._alpha = alpha
        self._dimensions = dimensions[:]
        self._loss_function = 'log'
        self._nonlinearity = 'relu'
        self._loss_function = 'log'
        self._reg_strength = 0
        self._dropout_prob = 0.0
        self._n_classes = None
        self._init_emb = init_emb
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
        self._col_names = None

        # create a variable to store the label proportions in the training data
        self._train_proportions = None
        # variable to hold the sklearn model
        self._model = None

    def get_model_type(self):
        return self._model_type

    def get_loss_function(self):
        return self._loss_function

    def set_model(self, model, train_proportions, col_names, n_classes):
        self._col_names = col_names
        self._train_proportions = train_proportions
        self._n_classes = n_classes
        if model is None:
            self._model = None
        else:
            self._model = model

    def fit(self, X_train, Y_train, X_dev, Y_dev, train_weights=None, dev_weights=None, col_names=None, seed=None, init_lr=1e-2, min_epochs=2, max_epochs=200, patience=8, dropout_prob=0.0):
        """
        Fit a classifier to data
        """
        _, n_classes = Y_train.shape
        self._n_classes = n_classes

        n_train, n_features = X_train.shape
        n_dev, _ = X_dev.shape
        Y_list_train = Y_train.argmax(axis=1)
        #Y_list_train = np.array(Y_train.argmax(axis=1), dtype=np.float32)
        Y_list_dev = Y_dev.argmax(axis=1)

        if col_names is not None:
            self._col_names = col_names
        else:
            self._col_names = range(n_features)

        # store the proportion of class labels in the training data
        if train_weights is None:
            class_sums = np.sum(Y_train, axis=0)
            dev_class_sums = np.sum(Y_dev, axis=0)
        else:
            class_sums = np.dot(train_weights, Y_train) / train_weights.sum()
            dev_class_sums = np.dot(dev_weights, Y_dev) / dev_weights.sum()
        self._train_proportions = (class_sums / float(class_sums.sum())).tolist()
        #print("Dev proportions", (dev_class_sums / float(dev_class_sums.sum())).tolist())

        # if there is only a single type of label, make a default prediction
        train_labels = np.argmax(Y_train, axis=1).reshape((n_train, ))
        if np.max(self._train_proportions) == 1.0:
            self._model = None
        else:
            #model_filename = os.path.join(self._output_dir, self._name + '.ckpt')
            # DEBUG!
            #self._model = torchDAN(self._dimensions, init_emb=self._init_emb.copy(), update_emb=self._update_emb)
            self._model = torchDAN(self._dimensions, init_emb=self._init_emb, update_emb=self._update_emb)
            best_model = torchDAN(self._dimensions, init_emb=self._init_emb, update_emb=self._update_emb)
            # train model

            #criterion = nn.BCELoss()
            criterion = nn.BCEWithLogitsLoss()
            #criterion = nn.CrossEntropyLoss()
            grad_params = filter(lambda p: p.requires_grad, self._model.parameters())
            optimizer = optim.Adagrad(grad_params, lr=init_lr, weight_decay=self._alpha)

            epoch = 0
            done = False
            n_epochs_since_improvement = 0
            best_dev_acc = 0.0
            while not done:
                running_loss = 0.0
                weight_sum = 0
                train_acc = 0.0
                for i in range(n_train):
                    X_i_list = X_train[i, :].nonzero()[1].tolist()
                    sel = np.random.choice((0, 1), p=(dropout_prob, 1-dropout_prob), size=len(X_i_list), replace=True)
                    X_i_list = [x for x_i, x in enumerate(X_i_list) if sel[x_i] == 1]
                    #if Y_list_train[i] == 1:
                    #    X_i_list += [14281]
                    #else:
                    #    X_i_list += [10863]
                    X_i_array = np.array(X_i_list, dtype=np.int).reshape(1, len(X_i_list))
                    #X_i_array = np.vstack([self._init_emb[x, :] for x in X_i_list]).mean(axis=0)

                    if len(X_i_list) > 0:
                        X_i = Variable(torch.LongTensor(X_i_array))
                        #X_i = Variable(torch.from_numpy(X_i_array))
                        #y_i = Variable(torch.LongTensor(Y_list_train[i:i+1]))
                        y_i = Variable(torch.from_numpy(np.array(Y_list_train[i:i+1], dtype=np.float32)))
                        #y_i = Variable(torch.LongTensor(Y_list_train[i:i+1].reshape(1, 1)))

                        optimizer.zero_grad()
                        outputs = self._model(X_i)
                        loss = criterion(outputs.view(-1), y_i)

                        # apply per-instance weights
                        #scaling_factor = torch.ones(loss.data.shape) * train_weights[i]
                        #loss.backward(scaling_factor)
                        loss.backward()

                        optimizer.step()
                        running_loss += loss.data[0]
                        pred = int(outputs.data.numpy() >= 0)
                        train_acc += (Y_list_train[i] == pred) * train_weights[i]
                        weight_sum += train_weights[i]

                    if (i+1) % 200 == 0:
                        print("%d %d %0.4f %0.4f" % (epoch, i+1, running_loss / weight_sum, train_acc / weight_sum))

                print("%d %d %0.4f %0.4f" % (epoch, i+1, running_loss / weight_sum, train_acc / weight_sum))

                """
                for j in [14281, 10863]:
                    X_i_array = np.array(j, dtype=np.int).reshape(1, 1)
                    #X_i_list = [j]
                    #X_i_array = np.vstack([self._init_emb[x, :] for x in X_i_list]).mean(axis=0)
                    X_i = Variable(torch.LongTensor(X_i_array))
                    #X_i = Variable(torch.from_numpy(X_i_array))
                    outputs = self._model(X_i)
                    print(j, outputs.data.numpy())
                """

                dev_acc = 0.0
                for i in range(n_dev):
                    X_i_list = X_dev[i, :].nonzero()[1].tolist()
                    X_i_array = np.array(X_i_list, dtype=np.int).reshape(1, len(X_i_list))
                    #X_i_array = np.vstack([self._init_emb[x, :] for x in X_i_list]).mean(axis=0)

                    if len(X_i_list) > 0:
                        X_i = Variable(torch.LongTensor(X_i_array))
                        #X_i = Variable(torch.from_numpy(X_i_array))
                        outputs = self._model(X_i)
                        pred = int(outputs.data.numpy() >= 0)
                        dev_acc += (Y_list_dev[i] == pred) * dev_weights[i]
                dev_acc /= np.sum(dev_weights)
                print("epoch %d: dev acc = %0.4f" % (epoch, dev_acc))

                if dev_acc > best_dev_acc:
                    print("Updating best model")
                    model_params = list(self._model.parameters())
                    best_model_params = list(best_model.parameters())
                    for i, p in enumerate(model_params):
                        best_model_params[i].data[:] = p.data[:]
                    best_dev_acc = dev_acc
                    n_epochs_since_improvement = 0
                else:
                    n_epochs_since_improvement += 1

                if epoch >= min_epochs and n_epochs_since_improvement > patience:
                    print("Patience exceeded. Done")
                    print("Best validation acc = %0.4f" % best_dev_acc)
                    done = True

                if epoch >= max_epochs:
                    print("Max epochs exceeded. Done")
                    print("Best validation acc = %0.4f" % best_dev_acc)
                    done = True

                epoch += 1

            # restore best model
            model_params = list(self._model.parameters())
            best_model_params = list(best_model.parameters())
            for i, p in enumerate(best_model_params):
                model_params[i].data[:] = p.data[:]

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

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model is None:
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * np.argmax(self._train_proportions)
        else:
            probs = self.predict_probs(X)
            predictions = np.argmax(probs, axis=1)
            return predictions

    def predict_probs(self, X, do_platt=False, do_cfm=False, verbose=False):
        # TODO: implement Platt
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model is None:
            default = np.argmax(self._train_proportions)
            full_probs[:, default] = 1.0
            return full_probs
        else:
            for i in range(n_items):
                X_i_list = X[i, :].nonzero()[1].tolist()
                X_i_array = np.array(X_i_list, dtype=np.int).reshape(1, len(X_i_list))
                X_i = Variable(torch.LongTensor(X_i_array))
                outputs = self._model(X_i)
                p = expit(outputs.data.numpy().copy())
                full_probs[i, :] = [1.0-p, p]
            return full_probs

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
                  'alpha': self._alpha,
                  'dimensions': self.get_dimensions(),
                  'loss_function': self._loss_function,
                  'nonlinearity': self._nonlinearity,
                  'objective': self._objective,
                  'reg_strength': self.get_reg_strength(),
                  'penalty': self.get_penalty(),
                  'update_emb': self._update_emb,
                  'pos_label': self._pos_label,
                  'n_classes': self.get_n_classes(),
                  'train_proportions': self.get_train_proportions(),
                  'train_f1': self._train_f1,
                  'train_acc': self._train_acc,
                  'dev_f1': self._dev_f1,
                  'dev_acc': self._dev_acc
                  }
        fh.write_to_json(output, os.path.join(self._output_dir, self._name + '_metadata.json'), sort_keys=False)
        fh.write_to_json(self.get_col_names(), os.path.join(self._output_dir, self._name + '_col_names.json'), sort_keys=False)
        np.savez(os.path.join(self._output_dir, self._name + '_dev_info.npz'), acc_cfm=self._dev_acc_cfm, pvc_cfm=self._dev_pvc_cfm, venn_info=self._venn_info)
        torch.save(self._model.state_dict(), os.path.join(self._output_dir, self._name + '_model'))


def load_from_file(model_dir, name):
    col_names = fh.read_json(os.path.join(model_dir, name + '_col_names.json'))
    input = fh.read_json(os.path.join(model_dir, name + '_metadata.json'))
    dimensions = input['dimensions']
    penalty = input['penalty']
    reg_strength = float(input['reg_strength'])
    pos_label = int(input['pos_label'])
    n_classes = int(input['n_classes'])
    train_proportions = input['train_proportions']
    loss_function = input['loss_function']
    nonlinearity = input['nonlinearity']
    objective = input['objective']
    update_emb = input['update_emb']
    alpha = input['alpha']

    classifier = DAN(dimensions, alpha=alpha, output_dir=model_dir, name=name, objective=objective, update_emb=update_emb)
    model = torchDAN(dimensions, update_emb=update_emb)
    model.load_state_dict(torch.load(os.path.join(model_dir, name + '_model')))
    classifier.set_model(model, train_proportions, col_names, n_classes)
    dev_info = np.load(os.path.join(model_dir, name + '_dev_info.npz'))
    classifier._dev_acc_cfm = dev_info['acc_cfm']
    classifier._dev_acc_cfms_ms = dev_info['acc_cfms_ms']

    return classifier


class torchDAN(nn.Module):

    def __init__(self, dims, init_emb=None, update_emb=False):
        super(torchDAN, self).__init__()
        #assert len(dims) == 3
        self.n_layers = len(dims)-2
        self.emb = nn.Embedding(dims[0], dims[1])
        if not update_emb:
            self.emb.weight.requires_grad = False
        if init_emb is not None:
            self.emb.weight.data.copy_(torch.from_numpy(init_emb))

        self.linear1 = nn.Linear(dims[1], dims[2])
        self.linear2 = nn.Linear(dims[2], dims[3])
        self.linear3 = nn.Linear(dims[3], dims[4])

    def forward(self, X):
        h = self.emb(X)
        #h = F.dropout(h)
        h = h.mean(dim=1)
        h = self.linear1(h)
        h = self.linear2(F.relu(h))
        h = self.linear3(F.relu(h))

        #h = h[:, -1, :]
        #print(h.data.numpy()[0, :6])
        #h = self.linear1(F.sigmoid(h))
        #for layer in self.layers:
        #    h = layer(F.sigmoid(h))
        return h


