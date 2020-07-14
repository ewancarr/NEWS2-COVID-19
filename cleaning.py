# Title:        Data cleaning for improved NEWS2 paper
# Author:       Ewan Carr
# Started:      2020-07-14

import os
from joblib import dump
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
verbose = False

raw = {}
for r in ['bloods', 'outcomes', 'vitals']:
    raw[r] = pd.read_csv(os.path.join('data', 'raw', '2020-06-30 Updated 1500',
                                      '1593424203.7842345_AST', r + '.csv'),
                         low_memory=False)

# Outcomes --------------------------------------------------------------------
oc = raw['outcomes'].rename(columns={'patient_pseudo_id': 'pid',
                                     'Sx Date': 'symp_onset',
                                     'Primary Endpoint': 'primary_end',
                                     'Death Date': 'death_date',
                                     'ITU date': 'itu_date',
                                     'Admit Date': 'admit_date'})
oc.columns = [x.lower() for x in oc.columns]

# Derive BAME
oc['bame'] = np.select([oc['ethnicity'].isin(['Black', 'Asian']),
                        oc['ethnicity'] == 'Caucasian',
                        True],
                       [True, False, pd.NA])

# Derive outcomes for paper ---------------------------------------------------
for i in ['symp_onset', 'death_date', 'itu_date', 'admit_date']:
    oc[i] = pd.to_datetime(oc[i])

# Set index date
# If nosocomial, use symptom onset; otherwise use admission date.
oc['nosoc'] = oc['symp_onset'] > oc['admit_date']
oc['index'] = np.where(oc['nosoc'], oc['symp_onset'], oc['admit_date'])

# Define endpoints
oc['end14'] = oc['index'] + pd.DateOffset(days=14)
oc['end3'] = oc['index'] + pd.DateOffset(hours=72)

# Check patients who died/ICU before symptom onset
oc['y_before_onset'] = ((oc['death_date'] < oc['symp_onset']) |
                        (oc['itu_date'] < oc['symp_onset']))

# Check patients who died/ICU before admission
oc['y_before_admit'] = ((oc['death_date'] < oc['admit_date']) |
                        (oc['itu_date'] < oc['admit_date']))

# Remove patients who died before admission
oc = oc[~oc['y_before_admit']]

# Define 14-day outcome
latest_extract = pd.to_datetime('2020-05-18')

oc['event14'] = np.select([oc['death_date'] <= oc['end14'],
                           oc['itu_date'] <= oc['end14'],
                           oc['end14'] <= latest_extract,
                           True],
                          ['death', 'itu', 'other', pd.NA])
oc['y14'] = pd.NA
oc['y14'][(oc['event14'] == 'death') | (oc['event14'] == 'itu')] = 1
oc['y14'][(oc['event14'] == 'other')] = 0

# Define 3-day outcome
oc['event3'] = np.select([oc['death_date'] <= oc['end3'],
                          oc['itu_date'] <= oc['end3'],
                          oc['end3'] <= latest_extract,
                          True],
                         ['death', 'itu', 'other', pd.NA])
oc['y3'] = pd.NA
oc['y3'][(oc['event3'] == 'death') | (oc['event3'] == 'itu')] = 1
oc['y3'][(oc['event3'] == 'other')] = 0

# Define survival outcomes ----------------------------------------------------
# Days until death
oc['td_days'] = (oc['death_date'] - oc['index']).dt.days
oc['td_cens'] = (oc['td_days'].isna()) | (oc['td_days'] > 14)
oc['td_days'] = np.select([oc['td_days'].isna(),
                           oc['td_days'] > 14,
                           True],
                          [14, 14, oc['td_days']])

# Days until ICU
oc['ti_days'] = (oc['itu_date'] - oc['index']).dt.days
oc['ti_cens'] = (oc['ti_days'].isna()) | (oc['ti_days'] > 14)
oc['ti_days'] = np.select([oc['ti_days'].isna(),
                           oc['ti_days'] > 14,
                           True],
                          [14, 14, oc['ti_days']])

# Days until death OR ICU
oc['either_date'] = oc[['itu_date', 'death_date']].min(axis=1)
oc['te_days'] = (oc['either_date'] - oc['index']).dt.days
oc['te_cens'] = (oc['te_days'].isna()) | (oc['te_days'] > 14)
oc['te_days'] = np.select([oc['te_days'].isna(),
                           oc['te_days'] > 14,
                           True],
                          [14, 14, oc['te_days']])


# Check that all patients have passed their 14-day endpoint
print(all((oc['end14'] < latest_extract)))

# Define 'number of comorbidities'
numcom = oc[['copd', 'asthma', 'hf',
             'diabetes', 'ihd', 'ckd', 'htn']].sum(axis=1)
numcom[numcom > 4] = 4
oc['numcom'] = numcom

# Vitals ----------------------------------------------------------------------
vt = raw['vitals']
vt['ut'] = pd.to_datetime(vt['RECORDED DATE'])

# Derive GCS score
gcs = vt.loc[:, vt.columns.str.startswith('GCS ')].copy()
for v in gcs:
    gcs[v] = gcs[v].str.extract('(\d+)').astype(float)
vt['gcs_score'] = gcs.sum(skipna=False, axis=1)

# Create oxygen measures
vt['oxlt'] = vt['Oxygen Litres']
vt['suppox'] = np.select([vt['Supplemental Oxygen'] == 'No (Air)',
                          vt['Supplemental Oxygen'] == 'Yes',
                          True],
                         [False, True, pd.NA])
vt['oxlt'][vt['Supplemental Oxygen'] == 'No (Air)'] = 0
vt['oxord'] = np.select([vt['oxlt'] == 0,
                         vt['oxlt'] <= 0.5,
                         vt['oxlt'] <= 1,
                         vt['oxlt'] <= 2,
                         vt['oxlt'] <= 3,
                         vt['oxlt'] <= 5,
                         vt['oxlt'] <= 10,
                        True],
                        [0, 1, 2, 3, 4, 5, 6, 7])

# Select required measures
vt = vt.rename(columns={'patient_pseudo_id': 'pid',
                        'Temperature': 'temp',
                        'Oxygen Saturation': 'oxsat',
                        'Respiration Rate': 'resp',
                        'Heart Rate': 'hr',
                        'Systolic BP': 'sbp',
                        'Diastolic BP': 'dbp',
                        'NEWS2 score': 'news2'})
keep = ['pid', 'temp', 'oxsat', 'resp', 'hr', 'sbp', 'dbp', 'news2', 'oxlt',
        'suppox', 'oxord', 'gcs_score']
vt = vt[['ut'] + keep]

# Pick first non-missing value following hospital admission and symptom onset
vt = vt.merge(oc, how='inner', on='pid')
vt['latest_measure'] = vt['index'] + pd.DateOffset(hours=48)
vt = vt[(vt['ut'] >= vt['admit_date']) &       # After hospital admission
        (vt['ut'] >= vt['symp_onset']) &       # After sympton onset
        (vt['ut'] <= vt['latest_measure'])]
vt = vt.sort_values(['pid', 'ut'],
                    ascending=True).groupby('pid').first().reset_index()
vt = vt[keep]

# Select items with <30% missing data
pct_miss = vt.isna().sum() / len(vt)
vt = vt[pct_miss.index[pct_miss < 0.3]]

# Bloods ----------------------------------------------------------------------
blood = raw['bloods']
blood = blood.rename(columns={'Unnamed: 0.1': 'null',
                              'updatetime': 'ut',
                              'basicobs_itemname_analysed': 'item_raw',
                              'textualObs': 'notes',
                              'basicobs_value_analysed': 'value',
                              'basicobs_unitofmeasure': 'units',
                              'basicobs_referencelowerlimit': 'lowerlim',
                              'basicobs_referenceupperlimit': 'upperlim',
                              'updatetime_raw': 'ut_raw',
                              'patient_pseudo_id': 'pid'})
blood = blood[['pid', 'ut', 'item_raw', 'value', 'units']]
blood = blood[blood['units'].notna()]

# Clean values
blood['value'][blood['value'].str.contains('\.\.\.')] = pd.NA
blood['value'] = blood['value'].str.replace('>|<', '')
blood['value'] = pd.to_numeric(blood['value'], errors='coerce')


# Clean names
def clean_names(item):
    for ch in [' ', '-', '/', '%', ':', '\'', '.']:
        if ch in item:
            item = item.replace(ch, '')
    return(item.lower())


item = blood['item_raw'].apply(clean_names)
item[item.str.contains('hba1c')] = 'hba1c'
item[item == 'creactiveprotein'] = 'crp'
item[item == 'aspartatetransaminase'] = 'art'
item[item == 'wbccount'] = 'wbc'
item[item == 'po2(t)'] = 'po2'
item[item == 'pco2(t)'] = 'pco2'
item[item.str.contains('lymphocytes')] = 'lymphocytes'
blood['item'] = item

# Parse time
blood['ut'] = pd.to_datetime(blood['ut'])

# Select required columns
keepers = ['pid', 'ut', 'item', 'value']
blood = blood[keepers]

# Remove measurements taken before index date, or after 14-day endpoint
blood = blood.merge(oc, how='left', on='pid')
blood['latest_measure'] = blood['index'] + pd.DateOffset(hours=48)
blood = blood[(blood['ut'] >= blood['index']) &             # After index date
              (blood['ut'] <= blood['end14']) &             # Before endpoint
              (blood['ut'] <= blood['latest_measure'])]     # Before endpoint
blood = blood[keepers]

# Drop duplicate entries
blood = blood.drop_duplicates()

# Select first non-missing measurement
blood = blood.sort_values(['pid',
                           'item',
                           'ut']).groupby(['pid',
                                           'item']).first()
del blood['ut']
blood = blood.reset_index().pivot(index='pid',
                                  columns='item',
                                  values='value')


# Select items with <30% missing data
pct_miss = blood.isna().sum() / len(blood)
blood = blood[pct_miss.index[pct_miss < 0.3]]

# Derive composite exposures --------------------------------------------------

# 1. Neutrophil to lymphocyte ratio (NLR; (divide number of neutrophils by
#    number of lymphocytes)
# 2. Lymphopenia (= lymphycyte score of < 1.1 x 10^9/L)
# 3. Lymphocyte/CRP ratio

blood['nlr'] = blood['neutrophils'] / blood['lymphocytes']
blood['lymp_crp'] = blood['lymphocytes'] / blood['crp']

# Get id
blood = blood.reset_index()

# Create baseline dataset with all measures -----------------------------------
bl = blood.merge(vt, on='pid', how='inner')
bl = bl.merge(oc, on='pid', how='inner')

# Clean blood/physiological measures ------------------------------------------

# Winsorize top/bottom 1%
to_trim = ['albumin', 'creatinine', 'crp', 'estimatedgfr', 'hb',
           'lymphocytes', 'neutrophils', 'plt', 'urea', 'wbc', 'nlr',
           'lymp_crp', 'temp', 'oxsat', 'oxlt', 'resp', 'hr',
           'sbp', 'dbp', 'news2']

bl[to_trim] = bl[to_trim].clip(lower=bl[to_trim].quantile(0.01),
                               upper=bl[to_trim].quantile(0.99),
                               axis=1)

# Transform to address skew
for v in ['resp', 'temp', 'wbc', 'neutrophils', 'urea', 'crp']:
    bl[v] = np.sqrt(bl[v])

for v in ['lymphocytes', 'creatinine']:
    bl[v] = np.log(bl[v] + 1)

for v in ['lymp_crp', 'nlr']:
    bl[v] = np.log(bl[v])

# Save ------------------------------------------------------------------------
dump(bl, os.path.join('data', 'clean', 'baseline.joblib'))
bl.to_csv(os.path.join('data', 'clean', 'baseline.csv'))
