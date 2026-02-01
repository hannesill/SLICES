#!/usr/bin/env Rscript
#' Extract ICU data using RICU.
#'
#' Produces standardized parquet files from any RICU-supported ICU dataset.
#' RICU handles concept lookup, unit harmonization, hourly binning, and gap filling.
#'
#' Output files:
#'   ricu_timeseries.parquet  - stay_id, hour, {concept}, {concept}_mask, ...
#'   ricu_stays.parquet       - stay_id, patient_id, intime, outtime, los_days, ...
#'   ricu_mortality.parquet   - stay_id, date_of_death, hospital_expire_flag, ...
#'   ricu_diagnoses.parquet   - stay_id, icd_code, icd_version
#'   ricu_metadata.yaml       - dataset, feature_names, n_features, ...
#'
#' Usage:
#'   Rscript scripts/preprocessing/extract_with_ricu.R \
#'     --dataset miiv \
#'     --output_dir data/ricu_output/miiv \
#'     --seq_length_hours 72

suppressPackageStartupMessages({
  library(ricu)
  library(arrow)
  library(yaml)
  library(data.table)
  library(optparse)
})

# ---------------------------------------------------------------------------
# CLI arguments
# ---------------------------------------------------------------------------

option_list <- list(
  make_option("--dataset", type = "character", default = NULL,
              help = "RICU source name (miiv, eicu, hirid, aumc, mimic, sic)"),
  make_option("--output_dir", type = "character", default = NULL,
              help = "Output directory for parquet files"),
  make_option("--seq_length_hours", type = "integer", default = 72L,
              help = "Max hours per stay [default: %default]")
)

opts <- parse_args(OptionParser(option_list = option_list))

VALID_DATASETS <- c("miiv", "eicu", "hirid", "aumc", "mimic", "sic",
                     "mimic_demo", "eicu_demo")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

#' Map demo datasets to their parent for dataset-specific logic.
dataset_family <- function(dataset) {
  switch(dataset, mimic_demo = "mimic", eicu_demo = "eicu", dataset)
}

#' Safely load a raw RICU source table into a data.frame.
#'
#' Returns NULL (with a warning) if the table cannot be loaded.
load_table <- function(dataset, table_name) {
  tryCatch({
    src <- as_src_env(dataset)
    tbl <- src[[table_name]]
    if (is.null(tbl)) {
      warning(sprintf("Table '%s' not found for dataset '%s'.",
                      table_name, dataset))
      return(NULL)
    }
    as.data.frame(tbl)
  }, error = function(e) {
    warning(sprintf("Could not load table '%s' for dataset '%s': %s",
                    table_name, dataset, conditionMessage(e)))
    NULL
  })
}

#' Create an empty data.frame with the given column names (all NA character).
empty_frame <- function(cols) {
  df <- data.frame(matrix(ncol = length(cols), nrow = 0))
  names(df) <- cols
  df
}

# ---------------------------------------------------------------------------
# 1. Discover available time-series concepts
# ---------------------------------------------------------------------------

discover_ts_concepts <- function(dataset) {
  # load_dictionary(src=...) returns only concepts with items for this dataset
  dict <- load_dictionary(src = dataset)
  all_concepts <- names(dict)

  # Exclude static / outcome concepts (not time-series)
  static_names <- c("age", "sex", "height", "weight", "bmi", "death", "adm")
  ts_concepts <- setdiff(all_concepts, static_names)

  message(sprintf("[1/6] Found %d time-series concepts for '%s'.",
                  length(ts_concepts), dataset))
  ts_concepts
}

# ---------------------------------------------------------------------------
# 2. Extract time-series
# ---------------------------------------------------------------------------

extract_timeseries <- function(dataset, concepts, seq_length_hours) {
  message("[2/6] Extracting time-series data (this may take a while)...")

  # RICU handles unit conversion, hourly binning, and aggregation (median)
  ts_data <- load_concepts(
    concepts,
    src        = dataset,
    interval   = hours(1L),
    merge_data = TRUE,
    verbose    = TRUE
  )

  # Fill gaps to create a dense hourly grid bounded by stay windows.
  # Cap the windows at seq_length_hours to limit memory usage.
  message("  Filling gaps to create dense hourly grid...")
  wins <- stay_windows(dataset, interval = hours(1L))
  end_col <- "end"
  set(wins, j = end_col, value = pmin(wins[[end_col]], hours(seq_length_hours)))
  ts_dense <- fill_gaps(ts_data, limits = wins)

  # Cap at seq_length_hours (belt-and-suspenders after limiting wins)
  idx_name <- index_var(ts_dense)
  ts_capped <- ts_dense[get(idx_name) < hours(seq_length_hours)]

  # Data columns (the actual concept values)
  concept_cols <- data_vars(ts_capped)

  # Derive boolean observation masks
  for (col in concept_cols) {
    mask_name <- paste0(col, "_mask")
    set(ts_capped, j = mask_name, value = !is.na(ts_capped[[col]]))
  }

  # Convert to plain data.frame
  df <- as.data.frame(ts_capped)

  # Integer hour column from the difftime index
  df$hour <- as.integer(as.numeric(df[[idx_name]], units = "hours"))

  # Standardise the ID column to "stay_id"
  id_name <- id_var(ts_capped)
  names(df)[names(df) == id_name] <- "stay_id"

  # Drop the original difftime index column (replaced by 'hour')
  df[[idx_name]] <- NULL

  message(sprintf("  Time-series: %d rows, %d concepts, %d stays.",
                  nrow(df), length(concept_cols),
                  length(unique(df$stay_id))))
  list(df = df, concept_cols = concept_cols)
}

# ---------------------------------------------------------------------------
# 3. Extract stay metadata
# ---------------------------------------------------------------------------

extract_stays_data <- function(dataset) {
  message("[3/6] Extracting stay metadata...")

  # --- Timing from stay_windows ---
  wins <- stay_windows(dataset, interval = hours(1L))
  wins_df <- as.data.frame(wins)
  id_name <- id_var(wins)
  names(wins_df)[names(wins_df) == id_name] <- "stay_id"
  wins_df$los_hours <- as.numeric(wins_df[["end"]] - wins_df[["start"]],
                                  units = "hours")
  wins_df$los_days <- wins_df$los_hours / 24.0
  # Keep only what we need from windows
  wins_df <- wins_df[, c("stay_id", "los_days"), drop = FALSE]

  # --- Clinical demographics via RICU ---
  avail <- names(load_dictionary(src = dataset))
  demo_concepts <- intersect(c("age", "sex", "height", "weight"), avail)
  clinical <- NULL
  if (length(demo_concepts) > 0) {
    clin_raw <- load_concepts(demo_concepts, src = dataset)
    clinical <- as.data.frame(clin_raw)
    clin_id <- id_var(clin_raw)
    names(clinical)[names(clinical) == clin_id] <- "stay_id"
  }

  # --- Dataset-specific admin data ---
  admin <- extract_admin_data(dataset)

  # Merge components: wins (los_days) + clinical (age, sex, ...) + admin
  result <- wins_df
  if (!is.null(clinical)) {
    result <- merge(result, clinical, by = "stay_id", all.x = TRUE)
  }
  if (!is.null(admin) && nrow(admin) > 0) {
    result <- merge(result, admin, by = "stay_id", all.x = TRUE)
  }

  # Rename sex -> gender with M/F convention (if present)
  if ("sex" %in% names(result)) {
    result$gender <- ifelse(
      tolower(as.character(result$sex)) %in% c("female", "f"), "F",
      ifelse(tolower(as.character(result$sex)) %in% c("male", "m"), "M", NA)
    )
    result$sex <- NULL
  }

  # Ensure expected columns exist (fill with NA if absent)
  expected <- c("stay_id", "patient_id", "hadm_id", "intime", "outtime",
                "los_days", "age", "gender", "race", "admission_type",
                "insurance", "first_careunit", "height", "weight")
  for (col in expected) {
    if (!col %in% names(result)) {
      result[[col]] <- NA
    }
  }
  result <- result[, expected, drop = FALSE]

  message(sprintf("  Stays: %d rows.", nrow(result)))
  result
}

# ---------------------------------------------------------------------------
# 3a. Dataset-specific admin data
# ---------------------------------------------------------------------------

extract_admin_data <- function(dataset) {
  fam <- dataset_family(dataset)
  switch(fam,
    miiv  = extract_admin_miiv(dataset),
    mimic = extract_admin_mimic(dataset),
    eicu  = extract_admin_eicu(dataset),
    hirid = extract_admin_hirid(dataset),
    aumc  = extract_admin_aumc(dataset),
    sic   = extract_admin_sic(dataset),
    {
      warning(sprintf("No admin extraction for dataset '%s'.", dataset))
      NULL
    }
  )
}

extract_admin_miiv <- function(dataset) {
  icu <- load_table(dataset, "icustays")
  pat <- load_table(dataset, "patients")
  adm <- load_table(dataset, "admissions")
  if (is.null(icu)) return(NULL)

  df <- icu[, c("stay_id", "subject_id", "hadm_id",
                "intime", "outtime", "first_careunit"),
            drop = FALSE]
  names(df)[names(df) == "subject_id"] <- "patient_id"

  if (!is.null(adm)) {
    adm_cols <- intersect(
      c("hadm_id", "race", "admission_type", "insurance"),
      names(adm)
    )
    if (length(adm_cols) > 1) {
      df <- merge(df, adm[, adm_cols, drop = FALSE],
                  by = "hadm_id", all.x = TRUE)
    }
  }
  df
}

extract_admin_mimic <- function(dataset) {
  icu <- load_table(dataset, "icustays")
  adm <- load_table(dataset, "admissions")
  if (is.null(icu)) return(NULL)

  # MIMIC-III uses icustay_id
  id_col <- if ("stay_id" %in% names(icu)) "stay_id" else "icustay_id"
  df <- icu[, intersect(c(id_col, "subject_id", "hadm_id",
                           "intime", "outtime", "first_careunit"),
                         names(icu)),
            drop = FALSE]
  if (id_col != "stay_id") names(df)[names(df) == id_col] <- "stay_id"
  if ("subject_id" %in% names(df)) {
    names(df)[names(df) == "subject_id"] <- "patient_id"
  }

  if (!is.null(adm)) {
    adm_cols <- intersect(
      c("hadm_id", "ethnicity", "admission_type", "insurance"),
      names(adm)
    )
    if (length(adm_cols) > 1) {
      adm_sub <- adm[, adm_cols, drop = FALSE]
      # MIMIC-III uses ethnicity instead of race
      if ("ethnicity" %in% names(adm_sub)) {
        names(adm_sub)[names(adm_sub) == "ethnicity"] <- "race"
      }
      df <- merge(df, adm_sub, by = "hadm_id", all.x = TRUE)
    }
  }
  df
}

extract_admin_eicu <- function(dataset) {
  pat <- load_table(dataset, "patient")
  if (is.null(pat)) return(NULL)

  # eICU patient table has most admin info
  id_col <- if ("patientunitstayid" %in% names(pat)) {
    "patientunitstayid"
  } else {
    names(pat)[1]  # fallback
  }
  pid_col <- if ("patienthealthsystemstayid" %in% names(pat)) {
    "patienthealthsystemstayid"
  } else if ("uniquepid" %in% names(pat)) {
    "uniquepid"
  } else {
    NA
  }

  df <- data.frame(
    stay_id        = pat[[id_col]],
    patient_id     = if (!is.na(pid_col)) pat[[pid_col]] else NA,
    hadm_id        = NA_integer_,
    intime         = NA,
    outtime        = NA,
    first_careunit = if ("unittype" %in% names(pat)) pat$unittype else NA,
    race           = if ("ethnicity" %in% names(pat)) pat$ethnicity else NA,
    admission_type = if ("hospitaladmitsource" %in% names(pat)) {
      pat$hospitaladmitsource
    } else {
      NA
    },
    insurance      = NA_character_,
    stringsAsFactors = FALSE
  )
  df
}

extract_admin_hirid <- function(dataset) {
  gen <- load_table(dataset, "general")
  if (is.null(gen)) return(NULL)

  id_col <- intersect(c("patientid", "admissionid"), names(gen))[1]
  if (is.na(id_col)) return(NULL)

  df <- data.frame(
    stay_id    = gen[[id_col]],
    patient_id = gen[[id_col]],  # HiRID has no separate patient-level ID
    hadm_id    = NA_integer_,
    intime     = if ("admissiontime" %in% names(gen)) gen$admissiontime else NA,
    outtime    = NA,
    stringsAsFactors = FALSE
  )
  df
}

extract_admin_aumc <- function(dataset) {
  adm <- load_table(dataset, "admissions")
  if (is.null(adm)) return(NULL)

  df <- data.frame(
    stay_id    = if ("admissionid" %in% names(adm)) adm$admissionid else NA,
    patient_id = if ("patientid" %in% names(adm)) adm$patientid else NA,
    hadm_id    = NA_integer_,
    intime     = if ("admittedat" %in% names(adm)) adm$admittedat else NA,
    outtime    = if ("dischargedat" %in% names(adm)) adm$dischargedat else NA,
    stringsAsFactors = FALSE
  )
  df
}

extract_admin_sic <- function(dataset) {
  cases <- load_table(dataset, "cases")
  if (is.null(cases)) return(NULL)

  id_col <- intersect(c("CaseID", "caseid"), names(cases))[1]
  if (is.na(id_col)) return(NULL)

  df <- data.frame(
    stay_id    = cases[[id_col]],
    patient_id = cases[[id_col]],  # SICdb has no separate patient-level ID
    hadm_id    = NA_integer_,
    intime     = if ("AdmissionOn" %in% names(cases)) cases$AdmissionOn else NA,
    outtime    = if ("DischargeOn" %in% names(cases)) cases$DischargeOn else NA,
    stringsAsFactors = FALSE
  )
  df
}

# ---------------------------------------------------------------------------
# 4. Extract mortality info
# ---------------------------------------------------------------------------

extract_mortality_data <- function(dataset) {
  message("[4/6] Extracting mortality data...")

  fam <- dataset_family(dataset)
  result <- switch(fam,
    miiv  = extract_mortality_miiv(dataset),
    mimic = extract_mortality_mimic(dataset),
    eicu  = extract_mortality_eicu(dataset),
    # For other datasets, fall back to RICU "death" concept
    extract_mortality_generic(dataset)
  )

  # Ensure expected columns
  expected <- c("stay_id", "date_of_death", "hospital_expire_flag",
                "dischtime", "discharge_location")
  for (col in expected) {
    if (!col %in% names(result)) result[[col]] <- NA
  }
  result <- result[, expected, drop = FALSE]

  message(sprintf("  Mortality: %d rows.", nrow(result)))
  result
}

extract_mortality_miiv <- function(dataset) {
  icu <- load_table(dataset, "icustays")
  pat <- load_table(dataset, "patients")
  adm <- load_table(dataset, "admissions")

  if (is.null(icu)) return(extract_mortality_generic(dataset))

  df <- data.frame(stay_id = icu$stay_id, stringsAsFactors = FALSE)

  if (!is.null(pat)) {
    pat_sub <- pat[, intersect(c("subject_id", "dod"), names(pat)),
                   drop = FALSE]
    icu_sub <- icu[, c("stay_id", "subject_id"), drop = FALSE]
    df <- merge(df, merge(icu_sub, pat_sub, by = "subject_id",
                          all.x = TRUE)[, c("stay_id", "dod")],
                by = "stay_id", all.x = TRUE)
    names(df)[names(df) == "dod"] <- "date_of_death"
  }

  if (!is.null(adm)) {
    adm_cols <- intersect(
      c("hadm_id", "hospital_expire_flag", "dischtime",
        "discharge_location"),
      names(adm)
    )
    if (length(adm_cols) > 1) {
      icu_hadm <- icu[, c("stay_id", "hadm_id"), drop = FALSE]
      adm_sub <- adm[, adm_cols, drop = FALSE]
      df <- merge(df, merge(icu_hadm, adm_sub, by = "hadm_id",
                            all.x = TRUE),
                  by = "stay_id", all.x = TRUE)
      df$hadm_id <- NULL
    }
  }

  df
}

extract_mortality_mimic <- function(dataset) {
  icu <- load_table(dataset, "icustays")
  pat <- load_table(dataset, "patients")
  adm <- load_table(dataset, "admissions")

  if (is.null(icu)) return(extract_mortality_generic(dataset))

  id_col <- if ("stay_id" %in% names(icu)) "stay_id" else "icustay_id"
  df <- data.frame(stay_id = icu[[id_col]], stringsAsFactors = FALSE)

  if (!is.null(pat)) {
    pat_sub <- pat[, intersect(c("subject_id", "dod"), names(pat)),
                   drop = FALSE]
    icu_sub <- icu[, c(id_col, "subject_id"), drop = FALSE]
    names(icu_sub)[1] <- "stay_id"
    merged <- merge(icu_sub, pat_sub, by = "subject_id", all.x = TRUE)
    df <- merge(df, merged[, c("stay_id", "dod"), drop = FALSE],
                by = "stay_id", all.x = TRUE)
    names(df)[names(df) == "dod"] <- "date_of_death"
  }

  if (!is.null(adm)) {
    adm_cols <- intersect(
      c("hadm_id", "hospital_expire_flag", "dischtime",
        "discharge_location"),
      names(adm)
    )
    if (length(adm_cols) > 1) {
      icu_hadm <- icu[, c(id_col, "hadm_id"), drop = FALSE]
      names(icu_hadm)[1] <- "stay_id"
      adm_sub <- adm[, adm_cols, drop = FALSE]
      df <- merge(df, merge(icu_hadm, adm_sub, by = "hadm_id",
                            all.x = TRUE),
                  by = "stay_id", all.x = TRUE)
      df$hadm_id <- NULL
    }
  }

  df
}

extract_mortality_eicu <- function(dataset) {
  pat <- load_table(dataset, "patient")
  if (is.null(pat)) return(extract_mortality_generic(dataset))

  id_col <- if ("patientunitstayid" %in% names(pat)) {
    "patientunitstayid"
  } else {
    names(pat)[1]
  }

  # eICU uses status strings and offsets rather than timestamps
  hosp_flag <- if ("hospitaldischargestatus" %in% names(pat)) {
    as.integer(tolower(pat$hospitaldischargestatus) == "expired")
  } else {
    NA_integer_
  }

  df <- data.frame(
    stay_id              = pat[[id_col]],
    date_of_death        = NA,
    hospital_expire_flag = hosp_flag,
    dischtime            = NA,
    discharge_location   = if ("unitdischargelocation" %in% names(pat)) {
      pat$unitdischargelocation
    } else {
      NA_character_
    },
    stringsAsFactors = FALSE
  )
  df
}

#' Fallback: use RICU "death" concept for datasets without detailed tables.
extract_mortality_generic <- function(dataset) {
  avail <- names(load_dictionary(src = dataset))
  if (!"death" %in% avail) {
    message("  Warning: 'death' concept not available; returning empty mortality.")
    wins <- stay_windows(dataset, interval = hours(1L))
    return(data.frame(
      stay_id              = as.data.frame(wins)[[id_var(wins)]],
      date_of_death        = NA,
      hospital_expire_flag = NA_integer_,
      dischtime            = NA,
      discharge_location   = NA_character_,
      stringsAsFactors     = FALSE
    ))
  }

  death_raw <- load_concepts("death", src = dataset)
  death_df <- as.data.frame(death_raw)
  id_name <- id_var(death_raw)

  df <- data.frame(
    stay_id              = death_df[[id_name]],
    date_of_death        = NA,
    hospital_expire_flag = as.integer(death_df$death),
    dischtime            = NA,
    discharge_location   = NA_character_,
    stringsAsFactors     = FALSE
  )
  df
}

# ---------------------------------------------------------------------------
# 5. Extract diagnoses
# ---------------------------------------------------------------------------

extract_diagnoses_data <- function(dataset) {
  message("[5/6] Extracting diagnoses data...")

  fam <- dataset_family(dataset)
  result <- switch(fam,
    miiv  = extract_diagnoses_miiv(dataset),
    mimic = extract_diagnoses_mimic(dataset),
    eicu  = extract_diagnoses_eicu(dataset),
    {
      message(sprintf(
        "  Warning: Diagnosis extraction not implemented for '%s'. Returning empty.",
        dataset
      ))
      empty_frame(c("stay_id", "icd_code", "icd_version"))
    }
  )

  expected <- c("stay_id", "icd_code", "icd_version")
  for (col in expected) {
    if (!col %in% names(result)) result[[col]] <- NA
  }
  result <- result[, expected, drop = FALSE]

  message(sprintf("  Diagnoses: %d rows.", nrow(result)))
  result
}

extract_diagnoses_miiv <- function(dataset) {
  diag <- load_table(dataset, "diagnoses_icd")
  icu  <- load_table(dataset, "icustays")
  if (is.null(diag) || is.null(icu)) {
    return(empty_frame(c("stay_id", "icd_code", "icd_version")))
  }

  # diagnoses_icd is keyed by hadm_id; join via icustays to get stay_id
  icu_map <- icu[, c("stay_id", "hadm_id"), drop = FALSE]
  diag_cols <- intersect(c("hadm_id", "icd_code", "icd_version"), names(diag))
  if (length(diag_cols) < 2) return(empty_frame(c("stay_id", "icd_code", "icd_version")))

  merged <- merge(icu_map, diag[, diag_cols, drop = FALSE],
                  by = "hadm_id", all.x = FALSE)
  merged$hadm_id <- NULL
  merged
}

extract_diagnoses_mimic <- function(dataset) {
  diag <- load_table(dataset, "diagnoses_icd")
  icu  <- load_table(dataset, "icustays")
  if (is.null(diag) || is.null(icu)) {
    return(empty_frame(c("stay_id", "icd_code", "icd_version")))
  }

  id_col <- if ("stay_id" %in% names(icu)) "stay_id" else "icustay_id"
  icu_map <- icu[, c(id_col, "hadm_id"), drop = FALSE]
  names(icu_map)[1] <- "stay_id"

  # MIMIC-III uses icd9_code; MIMIC-IV uses icd_code + icd_version
  if ("icd_code" %in% names(diag)) {
    diag_sub <- diag[, intersect(c("hadm_id", "icd_code", "icd_version"),
                                 names(diag)),
                     drop = FALSE]
  } else if ("icd9_code" %in% names(diag)) {
    diag_sub <- data.frame(
      hadm_id     = diag$hadm_id,
      icd_code    = diag$icd9_code,
      icd_version = 9L,
      stringsAsFactors = FALSE
    )
  } else {
    return(empty_frame(c("stay_id", "icd_code", "icd_version")))
  }

  merged <- merge(icu_map, diag_sub, by = "hadm_id", all.x = FALSE)
  merged$hadm_id <- NULL
  merged
}

extract_diagnoses_eicu <- function(dataset) {
  diag <- load_table(dataset, "diagnosis")
  if (is.null(diag)) {
    return(empty_frame(c("stay_id", "icd_code", "icd_version")))
  }

  id_col <- if ("patientunitstayid" %in% names(diag)) {
    "patientunitstayid"
  } else {
    names(diag)[1]
  }

  icd_col <- if ("icd9code" %in% names(diag)) {
    "icd9code"
  } else if ("diagnosisstring" %in% names(diag)) {
    "diagnosisstring"
  } else {
    NA
  }

  if (is.na(icd_col)) {
    return(empty_frame(c("stay_id", "icd_code", "icd_version")))
  }

  df <- data.frame(
    stay_id     = diag[[id_col]],
    icd_code    = diag[[icd_col]],
    icd_version = if (icd_col == "icd9code") 9L else NA_integer_,
    stringsAsFactors = FALSE
  )
  # Drop rows with empty ICD codes
  df <- df[!is.na(df$icd_code) & nchar(trimws(df$icd_code)) > 0, ]
  df
}

# ---------------------------------------------------------------------------
# 6. Write metadata
# ---------------------------------------------------------------------------

write_metadata <- function(output_dir, dataset, concept_cols,
                           seq_length_hours, n_stays) {
  message("[6/6] Writing metadata...")
  metadata <- list(
    dataset           = dataset,
    feature_names     = as.list(concept_cols),
    n_features        = length(concept_cols),
    seq_length_hours  = seq_length_hours,
    n_stays           = n_stays,
    ricu_version      = as.character(packageVersion("ricu"))
  )
  write_yaml(metadata, file.path(output_dir, "ricu_metadata.yaml"))
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function(opts) {
  dataset           <- opts$dataset
  output_dir        <- opts$output_dir
  seq_length_hours  <- opts$seq_length_hours

  # Validate arguments
  if (is.null(dataset) || is.null(output_dir)) {
    stop("Both --dataset and --output_dir are required.")
  }
  if (!dataset %in% VALID_DATASETS) {
    stop(sprintf("Invalid dataset '%s'. Valid: %s",
                 dataset, paste(VALID_DATASETS, collapse = ", ")))
  }

  # Check data availability
  if (!is_data_avail(dataset)) {
    stop(sprintf(
      paste0("Dataset '%s' data is not available in ricu. ",
             "Run ricu::setup_src_data('%s') first, or ensure ",
             "data files are in the expected location."),
      dataset, dataset
    ))
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("=== RICU extraction: dataset=%s, output=%s, hours=%d ===",
                  dataset, output_dir, seq_length_hours))

  # 1. Discover concepts
  concepts <- discover_ts_concepts(dataset)
  if (length(concepts) == 0) {
    stop("No time-series concepts found. Check ricu setup.")
  }

  # 2. Extract timeseries
  ts_result <- extract_timeseries(dataset, concepts, seq_length_hours)
  write_parquet(
    as.data.frame(ts_result$df),
    file.path(output_dir, "ricu_timeseries.parquet")
  )

  # 3. Extract stays
  stays <- extract_stays_data(dataset)
  write_parquet(stays, file.path(output_dir, "ricu_stays.parquet"))

  # 4. Extract mortality
  mortality <- extract_mortality_data(dataset)
  write_parquet(mortality, file.path(output_dir, "ricu_mortality.parquet"))

  # 5. Extract diagnoses
  diagnoses <- extract_diagnoses_data(dataset)
  write_parquet(diagnoses, file.path(output_dir, "ricu_diagnoses.parquet"))

  # 6. Write metadata
  n_stays <- length(unique(ts_result$df$stay_id))
  write_metadata(output_dir, dataset, ts_result$concept_cols,
                 seq_length_hours, n_stays)

  message("=== Extraction complete. ===")
  message(sprintf("  Output: %s", output_dir))
  message(sprintf("  Concepts: %d", length(ts_result$concept_cols)))
  message(sprintf("  Stays (with data): %d", n_stays))
}

main(opts)
