# Title:        Temporal external validation of supplemented NEWS2 score
# Author:       Ewan Carr
# Started:      2020-04-20

# NOTE: This is the code used for temporal external validation in the medRxiv
#       paper. It is included in this repository for reference only. Please see
#       "replicate.py" for replication purposes.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import (make_scorer,
                             roc_auc_score,
                             recall_score)
from sklearn.linear_model import LogisticRegressionCV
from functions import define_models, tn, tp, fn, fp, extract_scores

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
                      n_repeats=20,
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
    X_test = scaler.transform(X_test)
    X_test = imputer.transform(X_test)
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


# Extract summaries

column_names = ['roc', 'n_samp', 'n_feat', 'tp', 'tn', 'fp', 'fn', 'sens',
                'spec', 'ppv', 'npv']
scores = []
for k, v in fitted.items():
    scores.append(extract_scores(v['imp']))
fit_summary = pd.DataFrame(scores,
                           columns=column_names)
fit_summary['index'] = fs_labels

fit_summary.to_csv('results.csv')
