# Supplementing the National Early Warning Score (NEWS2) for anticipating early deterioration among patients with COVID-19 infection

<https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2>

# Overview

This repository provides pre-trained models to validate models in the [medRxiv
paper](https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2). 

# Data cleaning

The `replicate.py` script requires a dataset with the following columns:

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

# Data cleaning



