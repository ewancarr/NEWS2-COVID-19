# Title:        External validation of supplemented NEWS2 score
# Author:       Ewan Carr
# Started:      2020-04-20

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (make_scorer,
                             roc_auc_score,
                             confusion_matrix,
                             recall_score)
from sklearn.linear_model import LogisticRegressionCV


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                                 Functions                                 ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

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


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                         SELECT VALIDATION SAMPLE                          ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

# Load CSV with admission blood parameters + 14 day outcome.
baseline = pd.read_csv('baseline.csv')

# Select training sample
train = baseline[baseline['samp'] == 'TRAINING']
train = train.drop(labels='samp', axis=1)
print(np.shape(train))

# Select validation sample
valid = baseline[baseline['samp'] == 'VALIDATION']
valid = valid.drop(labels='samp', axis=1)
print(np.shape(valid))

# Define scorers ==============================================================

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

# Define models to fit --------------------------------------------------------

fs, fs_labels = define_models()

# Define final model, based on top features, by feature importance ------------

final_model = ['news2', 'crp_sqrt', 'neutrophils',
               'estimatedgfr', 'albumin', 'age']
fs.append(final_model)
fs_labels.append('FINAL')

# Set up inner CV =============================================================

inner = RepeatedKFold(n_splits=10,
                      n_repeats=50,
                      random_state=42)

# Function to fit a single model ----------------------------------------------


def fit_imputed(v, train, valid):
    """
    Function to test a single model in validation sample [valid], having
    trained on the training [train] sample, after scaling and imputation.
    """
    # Select features/outcome
    X_train = train[v]
    y_train = train['y']
    n_train = np.shape(X_train)[0]
    # Scale/impute
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    imputer = KNNImputer()
    X_train = imputer.fit_transform(X_train)
    # Train Logistic Regression with inner CV using training sample
    clf = LogisticRegressionCV(cv=inner,
                               penalty='l1',
                               Cs=10**np.linspace(0.1, -3, 50),
                               random_state=42,
                               solver='liblinear',
                               scoring=roc_auc_scorer).fit(X_train, y_train)
    # Predict in validation sample
    X_test = valid[v]
    y_test = valid['y']
    X_test = scaler.fit_transform(X_test)
    X_test = imputer.fit_transform(X_test)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    # Return
    imp = {'clf': clf,
           'n_train': n_train,
           'X_train': X_train,
           'y_train': y_train,
           'X_test': X_test,
           'y_test': y_test,
           'y_pred': y_pred,
           'y_prob': y_prob}
    return(imp)


# ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
# ┃                                                                           ┃
# ┃                      Test all parameter combinations                      ┃
# ┃                                                                           ┃
# ┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛

fitted = {}
for f, label in zip(fs, fs_labels):
    print(label)
    fitted[label] = {'imp': fit_imputed(f,
                                        train=train,
                                        valid=valid)}


# Extract summaries -----------------------------------------------------------
column_names = ['roc', 'n_samp', 'n_feat', 'tp', 'tn', 'fp', 'fn', 'sens',
                'spec', 'ppv', 'npv']
scores = []
for k, v in fitted.items():
    scores.append(extract_scores(v['imp']))
fit_summary = pd.DataFrame(scores,
                           columns=column_names)
fit_summary['index'] = fs_labels
fit_summary.to_csv('results.csv')
