# Supplementing the National Early Warning Score (NEWS2) for anticipating early deterioration among patients with COVID-19 infection

<https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2>

# Overview

This repository provides pre-trained models to validate models in the [medRxiv
paper](https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2). 


# Methods

# Required measures

* The `replicate.py` script requires a dataset (`validation.csv`) containing
  the measures in the table below.
* Not all columns are required for replication.
    * At a minimum, you need: `y`, `age`, `news2`, `crp_sqrt`, `neutrophils`,
      `estimatedgfr`, and `albumin`. 
    * This would allow validation of the supplemented NEWS2 score model from
      the paper (i.e. Models 1 and 5 from Table 3, p. 19).

|                          | Column              | Measure                                       | Transformation |
|--------------------------|---------------------|-----------------------------------------------|----------------|
| Outcome                  | `y`                 | Binary 14-day ICU/death outcome               | None           |
| Demographics             | `age`               | Age at admission in years                     | None           |
|                          | `male`              | Sex (0 = Female; 1 = Male)                    | None           |
| Blood parameters         | `crp_sqrt`          | C-reative protein (CRP; mg/L)                 | `np.sqrt`      |
|                          | `creatinine`        | Creatinine (µmol/L)                           | None           |
|                          | `albumin`           | Albumin (g/L)                                 | None           |
|                          | `estimatedgfr`      | Estimated Glomerular Filtration Rate (mL/min) | None           |
|                          | `alt`               | ALT                                           | None           |
|                          | `troponint`         | Troponin T (ng/L)                             | None           |
|                          | `ferritin`          | Ferritin (ug/L)                               | None           |
|                          | `lymphocytes_log10` | Lymphocyte count (x 10<sup>9</sup>)           | `np.log10`     |
|                          | `neutrophils`       | Neutrophil count (x 10<sup>9</sup>)           | None           |
|                          | `plt`               | Platelet count (x 10<sup>9</sup>)             | None           |
|                          | `nlr_log10`         | Neutrophil-to-lymphocyte ratio                | `np.log10`     |
|                          | `lymph_crp_log`     | Lymphocyte-to-CRP ratio                       | `np.log`       |
|                          | `hb`                | Haemoglobin (g/L)                             | None           |
| Physiological parameters | `news2`             | NEWS2 total score                             | None           |
|                          | `temp`              | Temperature (°C)                              | None           |
|                          | `oxsat`             | Oxygen saturation (%)                         | None           |
|                          | `resp`              | Respiratory rate (breaths per minute)         | None           |
|                          | `hr`                | Heart rate (beats/min)                        | None           |
|                          | `sbp`               | Systolic blood pressure (mmHg)                | None           |
|                          | `dbp`               | Diastolic blood pressure                      | None           |
|                          | `gcs_score`         | Glasgow Coma Scale total score                | None           |


# How to use this repository

The file [`replicate.py`][replicate.py] will fit a series of models using
pre-trained models. Specifically, it:

1. Imports a CSV file containing the required features and outcome
   (`validation.csv`).
2. For each feature set, it loads a pre-trained model (see
   [here][training/trained_models]) and tests this on the new data.

All models are written in Python using
[scikit-learn](https://scikit-learn.org/stable/). A minimal set of packages is
required (`pandas`, `numpy`, `scikit-learn`; see
[`requirements.txt`](software/requirments.txt).

Some notes:

* The code does not perform any training or cross-validation, with the
  exception of KNN imputation, see [below](#missing-data).
* Some code for data cleaning is provided ([`cleaning.R`](cleaning.R)) but this
  is quite specific to the structure of the source data. It should demonstrate
  how we prepared the training and validation datasets, but will likely require
  modification before running on replication samples.

## Cohort selection

The study cohort was defined as:

* All adult inpatients testing positive for SARS-Cov2 by reverse transcription
  polymerase chain reaction (RT-PCR);
* all patients included in the study had symptoms consistent with COVID-19
  disease (e.g. cough, fever, dyspnoea, myalgia, delirium).
* We excluded subjects who were seen in the emergency department but not
  admitted. 

## Timing

* The training sample included patients testing positive for SARS-Cov2 between
  1<sup>st</sup> and 30<sup>th</sup> March 2020. The external validation sample
  included patients testing positive on/after 31<sup>st</sup> March 2020.
* Where possible, these timeframes should be used in replications.

![Timing of training and validation sample](images/timing.png)

## Outcome

The primary outcome (`y`) was patient status at 14 days after symptom onset, or
admission to hospital where symptom onset was missing, categorised as:

1. Hospital admission but no transfer to ICU or death (WHO-COVID-19 Outcome
   Scale 3-5; `y` = 0)
2. Transfer to ICU or death (WHO-COVID-19 Outcome Scales 6-8; `y` = 1)

If date of symptom onset is unavailable it may be necessary to impute or use
date of hospital admission. Most endpoints happen at 10-14 days, so you may
have to work out how to define this.

## Features

Please refer to the [above](#required-measures) table for details of the
required measures. The aim of this analysis is to assess the improvement in
predictive performance achieved by supplementing NEWS2 with a small number of
blood/physiological parameters. The base model includes:

    age, news2, crp, neutrophils, estimatedgfr, albumin

A secondary model, if data are available, additionally includes:

    oxsat, troponint, lymphocytes_log10

The provided script ([`replicate.py`](replicate.py)) includes several other
feature sets which can be estimated subject to data availability.

All features must be measured at or shortly after hospital admission (within
24 hours).

## Missing data

* During training, missing feature information was imputed using KNN imputation
  ([`sklearn.impute.KNNImputer`](https://scikit-learn.org/stable/modules/generated/sklearn.impute.KNNImputer.html)). 
* For the purposes of temporal external validation, as presented in the paper,
  we used fitted imputation model to impute in the validation dataset.
* However, this repository does not provide the pre-trained KNN model, since
  the fitted object contains data that cannot be shared publicly.  
* Therefore, [`replicate.py`](replicate.py) will train the KNN imputation on
  the provided validation dataset (see [here](replicate.py#L109-L110)).
