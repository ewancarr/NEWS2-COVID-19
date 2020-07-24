# Title:        Code to replicate supplemented NEWS2 prediction model, based
#               on pre-trained models
# Author:       Ewan Carr
# Started:      2020-07-10

import os
import numpy as np
import pandas as pd
from joblib import load, dump
from sklearn.impute import KNNImputer
from sklearn.calibration import CalibratedClassifierCV
from functions import get_summaries, net_benefit, simulate_data

# Load validation dataset -----------------------------------------------------
if os.path.isfile('validation.csv'):
    sample = pd.read_csv('validation.csv')
else:
    sample = simulate_data(500, 9)

# Load pre-trained models -----------------------------------------------------
pretrained = load('pretrained.joblib')

# Confirm that validation dataset contains required features/outcomes ---------
avail = list(sample)
req = ['y3', 'y14', 'news2', 'oxlt', 'urea', 'age', 'oxsat', 'crp',
       'estimatedgfr', 'neutrophils', 'nlr', 'nosoc']
try:
    sample = sample[req]
except KeyError:
    print('Not all required variables in provided dataset; please check.')


# Apply required transformations ----------------------------------------------
if True:
    # NOTE: set to False if transformations have already been applied.

    # Windsorize
    to_trim = ['crp', 'estimatedgfr', 'neutrophils', 'urea', 'nlr',
               'oxsat', 'oxlt']
    sample[to_trim] = sample[to_trim] \
        .clip(lower=sample[to_trim].quantile(0.01),
              upper=sample[to_trim].quantile(0.99),
              axis=1)

    # Transform to address skew
    for v in ['neutrophils', 'urea', 'crp']:
        sample[v] = np.sqrt(sample[v])

    sample['nlr'] = np.log(sample['nlr'])


# Test each model -------------------------------------------------------------
def test_model(k, v, data):
    results = {}
    if k[4] == 'nosoc':
        # If excluding nosocomial patients
        if data['nosoc'].sum() == 0:
            return(False)
        else:
            data = data[data['nosoc'] == 0]
    X = data[v['X']]
    y = data[v['y']].astype('int')
    # Scale, impute
    scaler, clf = v['scaler'], v['clf']
    imputer = KNNImputer()
    X = scaler.transform(X)
    X = imputer.fit_transform(X)
    # 1. Pre-trained model ------------------------------------------------
    y_prob = clf.predict_proba(X)[:, 1]
    y_pred = clf.predict(X)
    results['pretrained'] = get_summaries(clf, X, y, y_prob, y_pred)
    # Get 'treat all' line for net benefit
    results['treat_all'] = net_benefit(clf, X, y, treat_all=True)
    # 2. Re-scaled model [based in internal validation] -------------------
    scale_coef = np.sum(X * (clf.coef_ * v['shrink_slope']), axis=1)
    scale_int = clf.intercept_ + v['shrink_int']
    odds = np.exp(scale_coef + scale_int)
    y_prob = odds / (1 + odds)
    y_pred = np.where(y_prob > 0.5, 1, 0)
    results['rescaled'] = get_summaries(clf, X, y, y_prob, y_pred)
    # 3. Re-calibrated model [based on validation sample] -----------------
    clf_recal = CalibratedClassifierCV(clf,
                                       method='sigmoid',
                                       cv='prefit').fit(X, y)
    y_pred = clf_recal.predict(X)
    y_prob = clf_recal.predict_proba(X)[:, 1]
    y_logp = np.log(y_prob / (1 - y_prob))
    results['recal'] = get_summaries(clf_recal, X, y, y_prob,
                                     y_pred, lp=y_logp)
    # Store outcome rate
    results['meany'] = np.mean(y)
    return(results)


# Fit all models
outputs = {}
for k, v in pretrained.items():
    fit = test_model(k, v, sample)
    if fit:
        outputs[k] = fit

# Print summary
discrim = {}
for k, v in outputs.items():
    discrim[k] = v['pretrained']['discrim']
discrim = pd.DataFrame(discrim).T
print(discrim)

# Save
dump(outputs, filename='replication.joblib')

# Please email the above file ('replication.joblib') to Ewan Carr.
