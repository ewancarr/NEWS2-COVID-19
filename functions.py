import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def define_models():
    '''
    Returns two lists containing [fs] parameters required for each model and
    [fs_labels] corresponding labels. These relate to Table 3 in the medRxiv
    paper.

    D = Age, sex
    C = comorbidities (8 features)
    B = bloods (10 features)
    P = physiological parameters (7 features)
    '''
    all_bloods = ['crp_sqrt', 'creatinine', 'albumin', 'estimatedgfr', 'alt',
                  'troponint', 'ferritin', 'lymphocytes_log10', 'neutrophils',
                  'plt', 'nlr_log10', 'lymph_crp_log', 'temp', 'oxsat', 'resp',
                  'hr', 'sbp', 'dbp', 'hb', 'gcs_score']

    comor = ['htn', 'diabetes', 'hf', 'ihd', 'copd', 'asthma', 'ckd']

    fs = [['news2'],
          ['news2', 'age', 'male'] + all_bloods,
          ['news2', 'age', 'male'] + all_bloods + comor,
          ['news2', 'age', 'male', 'bame'] + all_bloods,
          ['news2', 'age', 'male', 'bame'] + all_bloods + comor]

    fs_labels = ['NEWS2',
                 'NEWS2 + DBP',
                 'NEWS2 + DBPC',
                 'NEWS2 + DBP + E',
                 'NEWS2 + DBPC + E']
    return([fs, fs_labels])


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
