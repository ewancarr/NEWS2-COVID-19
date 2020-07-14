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
    sample = simulate_data(1000, 10)

# Load pre-trained models -----------------------------------------------------
pretrained = load('pretrained.joblib')

# Confirm that validation dataset contains all required features/outcomes -----
avail = list(sample)
req = ['y3', 'y14', 'news2', 'oxlt', 'urea', 'age', 'oxsat', 'crp',
       'estimatedgfr', 'neutrophils', 'plt', 'nlr', 'nosoc']
try:
    sample = sample[req]
except KeyError:
    print('Not all required variables in provided dataset; please check.')


# Test each model on validation sample ----------------------------------------
outputs = {}
for k, v in pretrained.items():
    for ns in ['all', 'nosoc']:
        results = {}
        # Select X, y
        if ns == 'nosoc':
            X = sample[sample.nosoc == 0][v['X']]
            y = sample[sample.nosoc == 0][v['y']].astype('int')
        else:
            X = sample[v['X']]
            y = sample[v['y']].astype('int')
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
        outputs[k + '_' + ns] = results

dump(outputs, filename='replication.joblib')
