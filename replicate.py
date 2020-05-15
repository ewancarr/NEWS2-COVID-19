# Title:        Code to replicate supplemented NEWS2 prediction model, based
#               on pre-trained models
# Author:       Ewan Carr
# Started:      2020-04-20

import os
import numpy as np
import pandas as pd
from joblib import load
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (make_scorer,
                             confusion_matrix,
                             roc_auc_score,
                             recall_score)


# Functions -------------------------------------------------------------------
def extract_scores(o):
    roc = roc_auc_score(o['y'], o['y_prob'])
    n_tp = tp(o['y'], o['y_pred'])
    n_tn = tn(o['y'], o['y_pred'])
    n_fp = fp(o['y'], o['y_pred'])
    n_fn = fn(o['y'], o['y_pred'])
    sens = np.mean(n_tp / (n_tp + n_fn))
    spec = np.mean(n_tn / (n_tn + n_fp))
    ppv = np.mean(n_tp / (n_tp + n_fp))
    npv = np.mean(n_tn / (n_tn + n_fn))
    n_samp = len(o['X'])
    n_feat = np.shape(o['X'])[1]
    return([roc, n_samp, n_feat, n_tp, n_tn, n_fp, n_fn, sens, spec, ppv, npv])


def tn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 0])


def fp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[0, 1])


def fn(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 0])


def tp(y_true, y_pred):
    return(confusion_matrix(y_true, y_pred)[1, 1])


# Define scoring --------------------------------------------------------------
roc_auc_scorer = make_scorer(roc_auc_score,
                             greater_is_better=True,
                             needs_threshold=True)
scoring = {'tp': make_scorer(tp),
           'tn': make_scorer(tn),
           'fp': make_scorer(fp),
           'fn': make_scorer(fn),
           'sensitivity': make_scorer(recall_score),
           'specificity': make_scorer(recall_score, pos_label=0),
           'roc': roc_auc_scorer,
           'recall': make_scorer(recall_score)}

# Load validation sample ------------------------------------------------------
validation = pd.read_csv('validation.csv')
if 'y' not in list(validation):
    raise ValueError('Dataset must contain binary outcome (y)')

# Load pre-trained models -----------------------------------------------------
pretrained = {}
for f in os.listdir('training/trained_models'):
    f = f.replace('.joblib', '')
    pretrained[f] = load('training/trained_models/' + f + '.joblib')

# Define models to fit --------------------------------------------------------
all_bloods = ['crp_sqrt', 'creatinine', 'albumin', 'estimatedgfr', 'alt',
              'troponint', 'ferritin', 'lymphocytes_log10', 'neutrophils',
              'plt', 'nlr_log10', 'lymph_crp_log', 'temp', 'oxsat', 'resp',
              'hr', 'sbp', 'dbp', 'hb', 'gcs_score']
comor = ['htn', 'diabetes', 'hf', 'ihd', 'copd', 'asthma', 'ckd']
models = {'NEWS2': ['news2'],
          'NEWS2 + DBP':  ['news2', 'age', 'male'] + all_bloods,
          'NEWS2 + DBPC':  ['news2', 'age', 'male'] + all_bloods + comor}

# Define final model, based on top features, by feature importance ------------
models['FINAL'] = ['news2', 'crp_sqrt', 'neutrophils',
                   'estimatedgfr', 'albumin', 'age']

# Function to fit a single model ----------------------------------------------
inner = RepeatedKFold(n_splits=10,
                      n_repeats=20,
                      random_state=42)


def test_model(feature_set, dataset):
    """
    Test validation sample on pre-trained model for a given feature set.
    """
    if 'y' not in list(dataset):
        raise ValueError('Dataset must contain binary outcome, y')
    if not set(models[feature_set]).issubset(list(dataset)):
        raise ValueError('Dataset must contain required features')
    clf = pretrained[feature_set]
    y = dataset['y']
    X = dataset[models[feature_set]]
    # Scale/impute
    scaler = StandardScaler()
    imputer = KNNImputer()
    X = scaler.fit_transform(X)
    X = imputer.fit_transform(X)
    # Predict
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    # Return
    return({'clf': clf,
            'X': X,
            'y': y,
            'y_pred': y_pred,
            'y_prob': y_prob})


# Test all features sets ------------------------------------------------------

# NOTE: we're fitting a smaller set of features below, to exclude models
# including comorbodities. This can be adjusted depending on data availability.
del models['NEWS2 + DBPC']

fitted = {}
for label, features in models.items():
    print(label)
    print(features)
    fitted[label] = test_model(label, validation)


# Extract summaries -----------------------------------------------------------
column_names = ['roc', 'n_samp', 'n_feat', 'tp', 'tn', 'fp', 'fn', 'sens',
                'spec', 'ppv', 'npv']
scores = []
for k, v in fitted.items():
    scores.append(extract_scores(v))
fit_summary = pd.DataFrame(scores,
                           columns=column_names)
fit_summary['index'] = models.keys()
print(fit_summary)
