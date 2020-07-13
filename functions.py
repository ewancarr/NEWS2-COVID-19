import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.calibration import calibration_curve
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.metrics import (make_scorer, roc_auc_score, confusion_matrix,
                             recall_score, brier_score_loss)


def savefig(fn):
    plt.tight_layout()
    plt.savefig(os.path.join('second_analysis', 'figures', fn + '.png'),
                dpi=300)


def simulate_data(n, features):
    X = np.random.rand(n, features)
    y = np.random.randint(0, 1 + 1, n)
    y = pd.DataFrame({'y3': y, 'y14': y})
    X = pd.DataFrame(X, columns=['news2', 'oxlt', 'urea', 'age', 'oxsat',
                                 'crp', 'estimatedgfr', 'neutrophils',
                                 'plt', 'nlr'])
    return(pd.concat([X, y], axis=1))


# Classification metrics ------------------------------------------------------

def tn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 0])


def fp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 1])


def fn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 0])


def tp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 1])


def npv(y_true, y_pred):
    tn = confusion_matrix(y_true, y_pred)[1, 1]
    fn = confusion_matrix(y_true, y_pred)[0, 1]
    return(np.mean(tn / (tn + fn)))


def ppv(y_true, y_pred):
    tp = confusion_matrix(y_true, y_pred)[0, 0]
    fp = confusion_matrix(y_true, y_pred)[1, 0]
    return(np.mean(tp / (tp + fp)))


scorers = {'auc': 'roc_auc',
           'sens': make_scorer(recall_score),
           'spec': make_scorer(recall_score, pos_label=0),
           'ppv': make_scorer(ppv),
           'npv': make_scorer(npv),
           'tp': make_scorer(tp),
           'tn': make_scorer(tn),
           'fp': make_scorer(fp),
           'fn': make_scorer(fn),
           'brier': make_scorer(brier_score_loss,
                                greater_is_better=True,
                                needs_proba=True)}


# Function to select 1SE model from LASSO CV ----------------------------------

def lower_bound(cv_results):
    '''
    Calculate the lower bound within 1 standard error
    of the best `mean_test_scores`.
    '''
    # Get idx of best model
    best_score_idx = np.argmax(cv_results['mean_test_score'])

    # Get number of folds
    K = len([x for x in list(cv_results.keys()) if x.startswith('split')])

    # Get standard error of mean
    best_score_sem = cv_results['std_test_score'][best_score_idx] / np.sqrt(K)

    # Return 1SE below best model
    return (cv_results['mean_test_score'][best_score_idx] - best_score_sem)


def pick_1se(cv_results):
    threshold = lower_bound(cv_results)
    candidate_idx = np.flatnonzero(cv_results['mean_test_score'] >= threshold)
    best_idx = candidate_idx[cv_results['param_logit__C']
                             [candidate_idx].argmin()]
    return(best_idx)


def extract_scores(y, y_pred, y_prob):
    roc = roc_auc_score(y, y_prob)
    n_tp = tp(y, y_pred)
    n_tn = tn(y, y_pred)
    n_fp = fp(y, y_pred)
    n_fn = fn(y, y_pred)
    sens = np.mean(n_tp / (n_tp + n_fn))
    spec = np.mean(n_tn / (n_tn + n_fp))
    ppv = np.mean(n_tp / (n_tp + n_fp))
    npv = np.mean(n_tn / (n_tn + n_fn))
    n_samp = len(y)
    return([roc, n_samp, n_tp, n_tn, n_fp, n_fn, sens, spec, ppv, npv])


def calibration_slope(clf, X, y, y_prob, lp):
    if lp is None:
        lp = np.sum(X * clf.coef_, axis=1) + clf.intercept_
    fit = sm.GLM(y.values, sm.add_constant(lp),
                 family=sm.families.Binomial()).fit()
    py = fit.predict()
    lo = lowess(y_prob, py)
    lx = lo[:, 1]
    ly = lo[:, 0]
    return({'lp': lp,
            'py': py,
            'lx': lx,
            'ly': ly})


def net_benefit(clf, X, y, lim=0.99, treat_all=False):
    if treat_all:
        probs, nb, n = [], [], len(y)
        for pt in np.arange(0.01, lim, 0.01):
            tp = sum(y == 1)
            fp = sum(y == 0)
            nb.append((tp/n) - (fp/n) * (pt / (1 - pt)))
            probs.append(pt)
    else:
        y_prob = clf.predict_proba(X)[:, 1]
        probs, nb, n = [], [], len(y)
        for pt in np.arange(0.01, lim, 0.01):
            tp = sum((y_prob >= pt) & (y == 1))
            fp = sum((y_prob >= pt) & (y == 0))
            nb.append((tp/n) - (fp/n) * (pt / (1 - pt)))
            probs.append(pt)
    return((probs, nb))


def get_summaries(clf, X, y, y_prob, y_pred, lp=None):
    res = {}
    # Predictions
    res['y_prob'] = y_prob
    res['y_pred'] = y_pred
    # Discrimination
    res['discrim'] = extract_scores(y, y_pred, y_prob)
    # Pick appropriate  number of bin, based on sample/event rate
    bins = int(np.max([(len(y) * np.mean(y) / 15), 5]))
    bins = bins if bins < 10 else 10
    # Calibration
    res['fop'], res['mpv'] = calibration_curve(y,
                                               y_prob,
                                               n_bins=bins,
                                               strategy='quantile')
    res['lowess'] = calibration_slope(clf, X, y, y_prob, lp)
    # Net benefit
    nbx, nby = net_benefit(clf, X, y)
    res['netben'] = (nbx, nby)
    return(res)
