# Title:        Data cleaning for NEWS2 paper
#               https://www.medrxiv.org/content/10.1101/2020.04.24.20078006v2
# Author:       Ewan Carr
# Started:      2020-04-07

library(lubridate)
library(tidyverse)
library(here)
library(naniar)
library(janitor)

# Load bloods/outcomes --------------------------------------------------------
bloods <- read_csv("bloods.csv")
outcomes <- read_csv("outcomes.csv")
vitals <- read_csv("vitals.csv")

###############################################################################
####                                                                      #####
####                      Clean outcomes/covariates                       #####
####                                                                      #####
###############################################################################

outcomes <- outcomes %>%
    rename(pid = patient_pseudo_id,
           symp_onset = `Sx Date`,
           death_date = `Death Date`,
           itu_date = `ITU date`,
           admit_date = `Admit Date`) %>%
    set_names(tolower(names(.)))

# Identify training/validation samples ----------------------------------------

outcomes <- outcomes %>%
    mutate(samp = case_when(batch %in% LETTERS[1:5] ~ "TRAINING",
                            batch %in% c("F", "G") ~ "VALIDATION"))

# Derive BAME -----------------------------------------------------------------
outcomes <- outcomes %>%
    mutate(bame = case_when(ethnicity %in% c("black/afro-caribbean",
                                             "asian") ~ 1,
                            ethnicity == "caucasian" ~ 0,
                            TRUE ~ NA_real_))

# Derive 14-day ICU/death outcome ---------------------------------------------

latest_extract <- ymd("2020-04-19")
day14_outcome <- outcomes %>%
    rowwise() %>%
    mutate(endpoint = symp_onset + days(14),
           event_type = case_when(death_date <= endpoint ~ "death",
                                  itu_date <= endpoint ~ "itu",
                                  endpoint <= latest_extract ~ "other",
                                  TRUE ~ NA_character_),
           y = if_else(event_type == "other", 0, 1))

# IMPORTANT: all patients must have passed their latest endpoint, as of latest
#            data extract.

###############################################################################
####                                                                      #####
####                              Clean vitals                            #####
####                                                                      #####
###############################################################################

# Fix inconsistent date/time coding -------------------------------------------
vitals <- vitals %>%
    mutate(new_time_formatting = nchar(`RECORDED DATE`) > 16,
           fixed_time = if_else(new_time_formatting, 
                                str_sub(`RECORDED DATE`, end = -8),
                                `RECORDED DATE`),
           ut = if_else(new_time_formatting,
                        ymd_hm(fixed_time),
                        dmy_hm(fixed_time))) 

# Derive GCS score ------------------------------------------------------------
vitals <- vitals %>%
    mutate_at(vars(starts_with("GCS")),
              ~ if_else(str_detect(.x, 
                                   "Non Testable"),
                        NA_character_, .x)) %>%
    mutate(gcs_eye = parse_number(`GCS Eye`),
           gcs_verb = parse_number(`GCS Verbal`),
           gcs_motor = parse_number(`GCS Motor`),
           gcs_score = gcs_eye + gcs_verb + gcs_motor) 

# Select required measures
selected_vitals <- vitals %>%
    select(pid = patient_pseudo_id,
           ut,
           temp = Temperature,
           oxsat = `Oxygen Saturation`,
           resp = `Respiration Rate`,
           hr = `Heart Rate`,
           sbp = `Systolic BP`,
           gcs_score,
           dbp = `Diastolic BP`,
           news2 = `NEWS2 score`)

# Pick first post-admission value ---------------------------------------------
baseline_vitals <- selected_vitals %>%
    left_join(day14_outcome) %>%
    filter(ut >= admit_date) %>%
    arrange(pid, ut) %>%
    group_by(pid) %>%
    summarise_at(vars(temp:news2), ~ first(na.omit(.x)))

###############################################################################
####                                                                      #####
####                             Clean bloods                             #####
####                                                                      #####
###############################################################################

# Remove cancelled/unauthorised tests -----------------------------------------

bloods <- bloods %>%
    filter(!is.na(basicobs_unitofmeasure)) 

# Rename variables, reshape ---------------------------------------------------

fix_names <- function(x) { 
    tolower(str_replace_all(x, " |-|/|%|'|\\.", "")) 
}

bloods <- bloods %>%
    select(-textualObs) %>%
    rename(item = basicobs_itemname_analysed,
           unit = basicobs_unitofmeasure,
           pid = patient_pseudo_id,
           value = basicobs_value_analysed) %>%
    mutate(ut = str_replace(updatetime,
                            "\\.\\d+$", ""),
           ut = mdy_hms(ut),
           item = fix_names(item),
           item = case_when(item == "creactiveprotein" ~ "crp",
                            TRUE ~ item),
           value = parse_number(value)) %>%
    select(pid, ut, item, value) %>%
    arrange(pid, ut, item)
# NOTE: Some values are recorded as ">90" or "<0.6". These are handled by
#       stripping the non-numeric characters, so "<90" becomes 90.

req <- c("lymphocytes",
         "albumin",
         "neutrophils",
         "troponint",
         "troponini",
         "hb",
         "neutrophils(manualdiff)",
         "ferritin",
         "creatinine",
         "estimatedgfr",
         "plt",
         "alt",
         "crp",
         "hba1c")

selected_bloods <- outcomes %>%
    select(pid, symp_onset) %>%
    right_join(bloods) %>%
    filter(item %in% req,
           ut >= symp_onset,
           ut <= symp_onset + days(14))


# IMPORTANT: Remove duplicate entries -----------------------------------------
#                                             (i.e. duplicate by pid, ut, item)
# NOTE: Some patients have duplicate measurements (same time, measure) with
# different values. These we take the first reported measurement.

selected_bloods <- selected_bloods %>%
    distinct(pid, ut, item, .keep_all = TRUE)

# Reshape to WIDE format ------------------------------------------------------

selected_bloods <- spread(selected_bloods, item, value)

# Drop bloods measures that have insufficient time points ---------------------

selected_bloods <- selected_bloods %>%
    select(-`neutrophils(manualdiff)`, -hba1c)

# Select first measure post-admission -----------------------------------------

baseline_bloods <- selected_bloods %>%
    full_join(day14_outcome, by = "pid") %>%
    filter(ut >= admit_date) %>%
    group_by(pid) %>%
    arrange(pid, ut) %>%
    summarise_at(vars(albumin:troponint),
                 ~ first(na.omit(.x)))

# Derive composite exposures --------------------------------------------------

# 1. "Neutrophil to lymphocyte ratio" (NLR; divide number of neutrophils by
#    number of lymphocytes)
# 2. Lymphopenia (a lymphycyte score of < 1.1 x 10^9/L)
# 3. Lymphocyte/CRP ratio

baseline_bloods <- baseline_bloods %>%
    mutate(nlr = neutrophils / lymphocytes,
           lymphopenia = lymphocytes < 1.1,
           lymph_crp = lymphocytes / crp)

###############################################################################
####                                                                      #####
####              Create baseline dataset with all measures               #####
####                                                                      #####
###############################################################################

# Combine, add outcomes -------------------------------------------------------

baseline <- baseline_bloods %>%
    full_join(baseline_vitals, by = "pid") %>%
    full_join(day14_outcome, by = "pid")

# Scaling/transformations -----------------------------------------------------

baseline <- baseline %>% 
    mutate(alt_log = log(alt),
           crp_sqrt = sqrt(crp),
           creatinine_log = log(creatinine),
           ferritin_log = log(ferritin),
           troponint_log = log(troponint),
           lymph_crp_log = log(lymph_crp),
           lymphocytes_log10 = log10(lymphocytes),
           nlr_log10 = log10(nlr))

###############################################################################
####                                                                      #####
####                                 SAVE                                 #####
####                                                                      #####
###############################################################################

write_csv(baseline, 
          path = "baseline.csv")
