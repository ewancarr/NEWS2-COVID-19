# Supplementing the National Early Warning Score (NEWS2) for anticipating early deterioration among patients with COVID-19 infection

<https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2>

# Overview

This repository provides pre-trained models to validate models in the [medRxiv
paper](https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2). 

# Data cleaning

The `replicate.py` script requires a dataset with (at least some of ) the
following columns:

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

Not all columns are required for replication. At a minimum, you need: `y`,
`age`, `news2`, `crp_sqrt`, `neutrophils`, `estimatedgfr`, and `albumin`. This
would allow validation of the supplemented NEWS2 score model from the paper
(i.e. Models 1 and 5 from Table 3, p. 19).

## Definitions

Please see `cleaning.R` for details how variables were derived.

* `y` is a binary variable indicating transfer to ICU/death (WHO-COVID-19
  Outcomes Scales 6-8) within 14 days of sympton onset.
    * All patients must have reached their 14-day endpoint (post-onset).
    * Patients who experienced the outcome (ICU/death) within the time period
      are scored 1; all other patients are scored 0.
* All parameters are measured at the first available occassion post-admission
  to hospital. For most patients this is within the first 36 hours of
  admission.


