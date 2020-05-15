import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def tn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 0])


def fp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 1])


def fn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 0])


def tp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 1])


def extract_scores(o):
    roc = roc_auc_score(o['y_test'], o['y_prob'])
    n_tp = tp(o['y_test'], o['y_pred'])
    n_tn = tn(o['y_test'], o['y_pred'])
    n_fp = fp(o['y_test'], o['y_pred'])
    n_fn = fn(o['y_test'], o['y_pred'])
    sens = np.mean(n_tp / (n_tp + n_fn))
    spec = np.mean(n_tn / (n_tn + n_fp))
    ppv = np.mean(n_tp / (n_tp + n_fp))
    npv = np.mean(n_tn / (n_tn + n_fn))
    n_samp = len(o['X_test'])
    n_feat = np.shape(o['X_test'])[1]
    return([roc, n_samp, n_feat, n_tp, n_tn, n_fp, n_fn, sens, spec, ppv, npv])
