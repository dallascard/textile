import numpy as np
from sklearn.preprocessing import normalize
from scipy.optimize import least_squares


def normalize_cfm(cfm):
    """
    Normalize a confusion matrix (column-wise, using l1-norm)
    :param cfm: confusion matrix to be normalized
    :return: normalized confusion matrix
    """
    return normalize(cfm, 'l1', axis=0)


def compute_pvc(labels, predictions, n_labels, n_pred_classes=None, do_normalization=True):
    """
    compute a confusion matrix of p(y=i|y_hat=j) values (for the predictive value correction)
    :param labels: vector of true labels
    :param predictions: vector of predicted labels
    :param n_labels: total number of labels (assumed to be 0, ..., n_labels-1)
    :return: matrix such that M[i,j] = p(y = i | y_hat = j)
    """
    if n_pred_classes is None:
        n_pred_classes = n_labels

    p_true_given_pred = np.zeros([n_labels, n_pred_classes])
    for i, true in enumerate(labels):
        pred = predictions[i]
        p_true_given_pred[true, pred] += 1

    if do_normalization:
        p_true_given_pred = normalize_cfm(p_true_given_pred)

    return p_true_given_pred


def apply_pvc(predictions, p_true_given_pred):
    """
    Apply the predictive value correction
    :param predictions: a vector of predicted labels
    :param p_true_given_pred: the pvc confusion matrix from compute_pvc, such that M[i,j] = p(y=i|y_hat=j)
    :return: a vector of predicted corrected proportions
    """
    _, n_classes = p_true_given_pred.shape
    pred_props = np.bincount(predictions.reshape(len(predictions,)), minlength=n_classes)
    pred_props = np.array(pred_props, dtype=float) / np.sum(pred_props)
    corrected_props = np.dot(p_true_given_pred, pred_props)
    return np.array(corrected_props.tolist())


def compute_ppvc(labels, pred_probs, n_labels, do_normalization=True):
    p_true_given_pred = np.zeros([n_labels, n_labels])
    for i, true in enumerate(labels):
        p_true_given_pred[true, :] += pred_probs[i, :]

    if do_normalization:
        p_true_given_pred = normalize_cfm(p_true_given_pred)

    return p_true_given_pred


def apply_ppvc(pred_probs, p_true_given_pred):
    _, n_labels = p_true_given_pred.shape
    #pred_props = np.bincount(predictions, minlength=n_labels)
    pred_proportions = np.mean(pred_probs, axis=0)
    try:
        assert np.abs(np.sum(pred_proportions) - 1.0) < 1e-5
    except AssertionError:
        print(np.sum(pred_proportions) == 1.0)
    #pred_proportions= np.array(pred_proportions, dtype=float) / np.sum(pred_props)
    corrected_props = np.dot(p_true_given_pred, pred_proportions)
    return np.array(corrected_props.tolist())


def compute_acc(labels, predictions, n_classes, do_normalization=True):
    """
    compute a confusion matrix of p(y_hat=i|y=j) values
    In paricular, the matrix contains the true positive rate and true negative rate
    also known as sensitivity and specificity

    :param labels: vector of true labels
    :param predictions: vector of predictions
    :param n_classes: actual number of labels (assumed to be 0, ..., n_labels-1)
    :return: a matrix such that M[i,j] = p(y_hat = i | y = j)
    """
    p_pred_given_true = np.zeros([n_classes, n_classes])

    for i, true in enumerate(labels):
        pred = predictions[i]
        p_pred_given_true[pred, true] += 1

    if do_normalization:
        p_pred_given_true = normalize_cfm(p_pred_given_true)
    return p_pred_given_true


def cc(predictions, n_labels):
    """
    The simplest estimation method: simply average the predictions
    :param predictions: a vector of predicted labels
    :n_labels: the total number of labels
    :return: a vector of predicted propotions
    """
    return np.bincount(predictions, minlength=n_labels)/float(len(predictions))


def pcc(predicted_probs):
    """
    The second simplest estimation method: just average the predicted probabilities for each item
    :param predicted_probs: a matrix of predicted probabilities [n_items x n_labels]
    :return: a vector of predicted propotions
    """
    pred_props = np.mean(predicted_probs, axis=0)
    return pred_props


def apply_acc_binary(predictions, p_pred_given_true):
    """
    compute the adjusted prediction of propotions based on an ACC correction
        using the simple binary formula

    :param predictions: vector of predictions (one per item)
    :param p_pred_given_true: matrix such that M[i,j] = p(y_hat=i|y=j)
    :return: vector of corrected proportions
    """
    n_classes = 2
    p_pred = np.array(np.bincount(predictions.reshape(len(predictions), ), minlength=n_classes), dtype=float)
    p_pred = p_pred / np.sum(p_pred)

    tpr = p_pred_given_true[1, 1]
    tnr = p_pred_given_true[0, 0]

    # binary formula; works fine

    if tpr == (1-tnr):
        # this happens most commonly when everything gets predicted to be 0 (or 1)
        #print "tpr = 1-tnr in apply_acc_binary", tpr, tnr
        return p_pred
    else:
        p_d1_binary = (p_pred[1] - (1 - tnr)) / (tpr - (1-tnr))
    if np.isinf(p_d1_binary) or np.isinf(p_d1_binary):
        print(p_pred[1], tpr, tnr)
        print("nan/inf encountered in apply_acc_binary")
        p_d1_binary = 1
    if p_d1_binary > 1:
        p_d1_binary = 1
    if p_d1_binary < 0:
        p_d1_binary = 0

    return np.array([1-p_d1_binary, p_d1_binary])


def apply_acc_solve(predictions, p_pred_given_true):
    """
    compute the adjusted prediction of propotions based on an ACC correction
        by solving an unconstrained system of equations
    :param predictions: vector of predictions
    :param p_pred_given_true: matrix such that M[i,j] = p(y_hat=i|y=j)
    :return: a vector of corrected proportion estimates
    """
    _, n_labels = p_pred_given_true.shape
    p_pred = np.array(np.bincount(predictions, minlength=n_labels), dtype=float)
    p_pred /= np.sum(p_pred)
    try:
        p_solve = np.linalg.solve(p_pred_given_true, p_pred)
    except np.linalg.linalg.LinAlgError:
        print("Singular matrix; skipping correction")
        p_solve = p_pred
    return p_solve


def apply_acc_lstsq(predictions, p_pred_given_true):
    """
    compute the adjusted prediction of propotions based on an ACC correction
        by solving a least-squares problem
    :param predictions: vector of predictions
    :param p_pred_given_true: matrix such that M[i,j] = p(y_hat=i|y=j)
    :return: a vector of corrected proportion estimates
    """
    _, n_labels = p_pred_given_true.shape
    p_pred = np.array(np.bincount(predictions, minlength=n_labels), dtype=float)
    p_pred /= np.sum(p_pred)
    p_lstsq, residuals, rank, singular_vals = np.linalg.lstsq(p_pred_given_true, p_pred)
    return p_lstsq


def apply_acc_bounded_lstsq(predictions, p_pred_given_true, verbose=1):
    """
    compute the adjusted prediction of propotions based on an ACC correction
        by solving a constrained least-squares problem
    :param predictions: vector of predictions
    :param p_pred_given_true: matrix such that M[i,j] = p(y_hat=i|y=j)
    :return: a vector of corrected proportion estimates
    """
    n_classes, n_labels = p_pred_given_true.shape
    #n_labels, n_classes = p_pred_given_true.shape
    p_pred = np.array(np.bincount(predictions, minlength=n_classes), dtype=float)
    p_pred /= np.sum(p_pred)

    p_opt = np.ones(n_labels)
    try:
        result = least_squares(f, p_opt, bounds=(0, 1), args=(p_pred_given_true, p_pred), verbose=verbose)
        # I need to find a constrained least-squares implementation that will allow enforcing the sum to one constraint
        # until I do, just do a normalization of the result
        corrected = result['x'] / np.sum(result['x'])
        converged = True
    except np.linalg.LinAlgError as e:
        print(e)
        print(e.message)
        print("Skipping correction...")
        corrected = p_opt
        converged = False

    return corrected, converged


def compute_pacc(labels, pred_probs, n_labels, do_normalization=True):
    """
    compute a the expected value of p(y_hat=1|p(y=1))

    :param labels: vector of true labels
    :param pred_probs: matrix of predicted probabilities [n_items x n_labels]
    :return: a matrix such that M[i,j] = E[p(y_hat=i|p(y=j))]
    """
    # NOTE: check comments above

    #n_items, n_labels = pred_probs.shape
    p_pred_given_true = np.zeros([n_labels, n_labels])

    for i, true in enumerate(labels):
        p_pred_given_true[:, true] += pred_probs[i, :]

    if do_normalization:
        p_pred_given_true = normalize_cfm(p_pred_given_true)

    return p_pred_given_true



def apply_pacc_binary(pred_probs, p_pred_given_true):
    """
    compute the adjusted prediction of propotions based on an ACC correction
        using the simple binary formula

    :param predictions: vector of predictions (one per item)
    :param p_pred_given_true: matrix such that M[i,j] = p(y_hat=i|y=j)
    :return: vector of corrected proportions
    """
    n_labels, _ = p_pred_given_true.shape
    assert n_labels == 2
    p_pred = np.mean(pred_probs, axis=0)

    etpr = p_pred_given_true[1, 1]
    etnr = p_pred_given_true[0, 0]

    # binary formula; works fine
    if etpr == (1-etnr):
        print("etpr = etrn in apply_pacc_binary")
        p_d1_binary = p_pred[1]
    else:
        p_d1_binary = (p_pred[1] - (1 - etnr)) / (etpr - (1-etnr))
    if np.isinf(p_d1_binary) or np.isinf(p_d1_binary):
        print(p_pred[1], etpr, etnr)
        print("nan/inf encountered in apply_pacc_binary")
        p_d1_binary = 1
    if p_d1_binary > 1:
        p_d1_binary = 1
    if p_d1_binary < 0:
        p_d1_binary = 0

    return np.array([1-p_d1_binary, p_d1_binary])


def apply_pacc_solve(prediction_probs, p_pred_given_true):
    """
    Apply the correction calculated from compute_pacc
    :param prediction_probs: vector of individual predictions
    :param p_pred_given_true: a correction matrix from compute_pacc
    :return: a vector of predicted propotiosn
    """

    pred_props = np.mean(prediction_probs, axis=0)
    try:
        proportions = np.linalg.solve(p_pred_given_true, pred_props)
    except np.linalg.linalg.LinAlgError:
        print("Singular matrix encountered; skipping correction")
        proportions = pred_props

    return proportions


def apply_pacc_bounded_lstsq(prediction_probs, p_pred_given_true, verbose=1):

    n_labels, _ = p_pred_given_true.shape

    #p_pred = np.array(np.bincount(predictions, minlength=n_classes), dtype=float)
    #p_pred /= np.sum(p_pred)
    pred_props = np.mean(prediction_probs, axis=0)

    p_opt = np.ones(n_labels)
    result = least_squares(f, p_opt, bounds=(0, 1), args=(p_pred_given_true, pred_props), verbose=verbose)
    # I need to find a constrained least-squares implementation that will allow enforcing the sum to one constraint
    # until I do, just do a normalization of the result
    corrected = result['x'] / np.sum(result['x'])
    return corrected


def f(p_opt, p_pred_given_true, p_pred):
    """
    Compute the least-squares objective: (Ax - b)**2
    :param p_opt: x
    :param p_pred_given_true: A
    :param p_pred: b
    :return:
    """
    return (np.dot(p_pred_given_true, p_opt) - p_pred)**2