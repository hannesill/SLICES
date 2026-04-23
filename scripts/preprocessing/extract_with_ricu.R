#!/usr/bin/env Rscript
#' Extract ICU data using RICU.
#'
#' Produces standardized parquet files from any RICU-supported ICU dataset.
#' RICU handles concept lookup, unit harmonization, hourly binning, and gap filling.
#'
#' Output files:
#'   ricu_timeseries.parquet/ - directory of parquet part files (chunked by stay)
#'                              each part: stay_id, hour, {concept}, {concept}_mask, ...
#'   ricu_stays.parquet       - stay_id, patient_id, intime, outtime, los_days, ...
#'   ricu_mortality.parquet   - stay_id, date_of_death, hospital_expire_flag, ...
#'   ricu_diagnoses.parquet   - stay_id, icd_code, icd_version
#'   ricu_metadata.yaml       - dataset, feature_names, n_features, ...
#'
#' Usage:
#'   Rscript scripts/preprocessing/extract_with_ricu.R \
#'     --dataset miiv \
#'     --output_dir data/ricu_output/miiv \
#'     --raw_export_horizon_hours 48

# Auto-install missing packages
required_packages <- c("ricu", "arrow", "yaml", "data.table", "optparse", "units")
missing <- required_packages[!vapply(required_packages, requireNamespace,
                                      logical(1), quietly = TRUE)]
if (length(missing) > 0) {
  message("Installing missing R packages: ", paste(missing, collapse = ", "))
  install.packages(missing, repos = "https://cloud.r-project.org")
}


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
  make_option("--raw_export_horizon_hours", type = "integer", default = NULL,
              help = "Max hours of raw timeseries to export per stay [default: 72]"),
  make_option("--seq_length_hours", type = "integer", default = NULL,
              help = "DEPRECATED alias for --raw_export_horizon_hours"),
  make_option("--raw_data_dir", type = "character", default = NULL,
              help = "Path to raw CSV files for ricu import (auto-detected if not set)")
)

opts <- parse_args(OptionParser(option_list = option_list))

VALID_DATASETS <- c("miiv", "eicu", "hirid", "aumc", "mimic", "sic",
                     "mimic_demo", "eicu_demo")

# Batch size for concept loading — controls peak memory during extraction.
CONCEPT_BATCH_SIZE <- 4L

# Tables actually needed for concept extraction and admin/mortality/diagnosis
# data.  Derived from ricu:::tbl_cfg — tables referenced by time-series,
# admin, mortality, and diagnosis concepts.  Everything else (emar_detail,
# pharmacy, poe, prescriptions, …) is unused and can take 20+ min to import
# for no benefit.
ESSENTIAL_TABLES <- list(
  miiv = c("chartevents", "labevents", "inputevents", "outputevents",
           "procedureevents", "ingredientevents", "datetimeevents",
           "icustays", "patients", "admissions", "transfers",
           "d_labitems", "diagnoses_icd", "d_icd_diagnoses"),
  mimic = c("chartevents", "labevents", "inputevents", "outputevents",
            "procedureevents", "ingredientevents", "datetimeevents",
            "icustays", "patients", "admissions", "transfers",
            "d_labitems", "diagnoses_icd", "d_icd_diagnoses"),
  # eicu requires all tables: load_concepts() internally calls
  # load_dictionary() which checks full source availability.
  # Omitted from this list so all tables are imported (see fallback below).
  hirid = c("observations", "pharma", "general"),
  aumc = c("numericitems", "drugitems", "procedureorderitems",
           "admissions", "listitems")
)

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

#' Parse the RICU concept dictionary for a dataset, bypassing the availability
#' gate.  RICU considers a source "available" only when its internal checks
#' pass, which can fail even after a successful import_src().  This helper
#' reads the raw dictionary and parses it directly.
parse_ricu_dictionary <- function(dataset) {
  raw <- ricu:::read_dictionary(name = "concept-dict")
  ricu:::parse_dictionary(raw, dataset, NULL)
}

# ---------------------------------------------------------------------------
# 1. Discover available time-series concepts
# ---------------------------------------------------------------------------

discover_ts_concepts <- function(dataset, dict) {
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

extract_timeseries <- function(dataset, concepts, seq_length_hours, dict,
                               output_path) {
  message("[2/6] Extracting time-series data in batches...")

  # Pre-compute stay windows (shared across all batches)
  wins <- stay_windows(dataset, interval = hours(1L))
  end_col <- "end"
  set(wins, j = end_col, value = pmin(wins[[end_col]], hours(seq_length_hours)))

  # Split concepts into batches to control peak memory
  n_concepts <- length(concepts)
  batch_indices <- split(
    seq_len(n_concepts),
    ceiling(seq_len(n_concepts) / CONCEPT_BATCH_SIZE)
  )
  n_batches <- length(batch_indices)

  # Helper: load a set of concepts into a data.table with stay_id, hour, and
  # value + mask columns.  Returns NULL on failure.
  load_batch <- function(cnames, dataset, dict, wins, seq_length_hours) {
    tryCatch({
      ts_data <- load_concepts(
        cnames,
        src        = dataset,
        concepts   = dict,
        interval   = hours(1L),
        merge_data = TRUE,
        verbose    = FALSE
      )
      ts_dense <- fill_gaps(ts_data, limits = wins)
      idx_name <- index_var(ts_dense)
      ts_capped <- ts_dense[get(idx_name) < hours(seq_length_hours)]
      batch_concept_cols <- data_vars(ts_capped)
      for (col in batch_concept_cols) {
        mask_name <- paste0(col, "_mask")
        set(ts_capped, j = mask_name, value = !is.na(ts_capped[[col]]))
      }
      dt <- as.data.table(ts_capped)
      id_name <- id_var(ts_capped)
      dt[, hour := as.integer(as.numeric(get(idx_name), units = "hours"))]
      setnames(dt, id_name, "stay_id")
      dt[, (idx_name) := NULL]
      rm(ts_data, ts_dense, ts_capped)
      gc()
      dt
    }, error = function(e) {
      warning(sprintf("  Failed to load concepts [%s]: %s",
                      paste(cnames, collapse = ", "), conditionMessage(e)))
      NULL
    })
  }

  # --- Phase 0: Discover stay IDs and assign to chunks -------------------
  # We need stay IDs before processing batches so we can pre-split each
  # batch by stay chunk on first read (avoiding re-reading batch files).
  message("  Discovering stay IDs...")
  all_stay_ids <- sort(unique(as.data.table(wins)[[id_var(wins)]]))
  STAY_CHUNK_SIZE <- 10000L
  # Map each stay_id to its chunk index for fast lookup
  chunk_assignment <- ceiling(seq_along(all_stay_ids) / STAY_CHUNK_SIZE)
  names(chunk_assignment) <- as.character(all_stay_ids)
  n_chunks <- max(chunk_assignment)
  n_stays <- length(all_stay_ids)
  message(sprintf("  %d stays → %d chunks of ≤%d stays.",
                  n_stays, n_chunks, STAY_CHUNK_SIZE))

  # --- Phase 1: Extract concept batches and pre-split by stay chunk ------
  # Each concept batch is loaded once, split by stay chunk, and written to
  # tmp/<chunk_idx>/batch_<batch_idx>.parquet.  This way each batch's data
  # is read from RICU exactly once (not re-read per chunk in Phase 2).
  tmp_dir <- file.path(tempdir(), "ricu_merge")
  unlink(tmp_dir, recursive = TRUE)
  for (ci in seq_len(n_chunks)) {
    dir.create(file.path(tmp_dir, sprintf("chunk_%04d", ci)),
               recursive = TRUE, showWarnings = FALSE)
  }

  skipped_concepts <- character(0)
  all_concept_cols <- character(0)
  batches_written <- 0L

  for (b in seq_len(n_batches)) {
    batch_concepts <- concepts[batch_indices[[b]]]
    message(sprintf("  Batch %d/%d (%d concepts): %s",
                    b, n_batches, length(batch_concepts),
                    paste(batch_concepts, collapse = ", ")))

    dt <- load_batch(batch_concepts, dataset, dict, wins, seq_length_hours)

    if (is.null(dt) && length(batch_concepts) > 1) {
      # Retry concepts individually to isolate the broken one(s)
      message("    Batch failed — retrying concepts individually...")
      individual_dts <- list()
      for (cname in batch_concepts) {
        cdt <- load_batch(cname, dataset, dict, wins, seq_length_hours)
        if (is.null(cdt)) {
          message(sprintf("    Skipping concept '%s' (unsupported for this dataset).", cname))
          skipped_concepts <- c(skipped_concepts, cname)
        } else {
          individual_dts[[length(individual_dts) + 1]] <- cdt
        }
        rm(cdt); gc()
      }
      if (length(individual_dts) > 0) {
        dt <- individual_dts[[1]]
        if (length(individual_dts) > 1) {
          for (j in 2:length(individual_dts)) {
            dt <- merge(dt, individual_dts[[j]],
                        by = c("stay_id", "hour"), all = TRUE)
          }
        }
        rm(individual_dts); gc()
      } else {
        dt <- NULL
      }
    } else if (is.null(dt)) {
      message(sprintf("    Skipping concept '%s' (unsupported for this dataset).",
                      batch_concepts))
      skipped_concepts <- c(skipped_concepts, batch_concepts)
    }

    if (!is.null(dt)) {
      # Track concept columns from this batch
      batch_cols <- names(dt)
      batch_mask <- batch_cols[grepl("_mask$", batch_cols)]
      all_concept_cols <- c(all_concept_cols, sub("_mask$", "", batch_mask))

      # Split by stay chunk and write to per-chunk directories
      dt[, .chunk_idx := chunk_assignment[as.character(stay_id)]]
      for (ci in unique(dt$.chunk_idx)) {
        chunk_dt <- dt[.chunk_idx == ci]
        chunk_dt[, .chunk_idx := NULL]
        chunk_path <- file.path(tmp_dir, sprintf("chunk_%04d", ci),
                                sprintf("batch_%03d.parquet", b))
        write_parquet(chunk_dt, chunk_path)
        rm(chunk_dt)
      }
      batches_written <- batches_written + 1L
    }
    rm(dt); gc()
  }

  rm(wins); gc()

  if (length(skipped_concepts) > 0) {
    message(sprintf("  Skipped %d concepts: %s",
                    length(skipped_concepts),
                    paste(skipped_concepts, collapse = ", ")))
  }

  if (batches_written == 0) {
    stop("All concept batches failed. Cannot continue.")
  }

  # --- Phase 2: Merge pre-split batches per chunk -------------------------
  # For each stay chunk, read its small batch files, merge wide, and write
  # one output parquet part.  Peak memory ≈ one chunk's worth of data in
  # narrow form + one chunk wide (~1-2 GB for 10k stays).
  message(sprintf("  Merging %d chunks...", n_chunks))
  dir.create(output_path, showWarnings = FALSE, recursive = TRUE)
  total_rows <- 0L

  for (ci in seq_len(n_chunks)) {
    chunk_dir <- file.path(tmp_dir, sprintf("chunk_%04d", ci))
    batch_files <- list.files(chunk_dir, pattern = "\\.parquet$",
                              full.names = TRUE)
    if (length(batch_files) == 0) next

    chunk_merged <- NULL
    for (bf in batch_files) {
      batch_dt <- as.data.table(read_parquet(bf))
      setkeyv(batch_dt, c("stay_id", "hour"))

      if (is.null(chunk_merged)) {
        chunk_merged <- batch_dt
      } else {
        chunk_merged <- merge(chunk_merged, batch_dt,
                              by = c("stay_id", "hour"), all = TRUE)
      }
      rm(batch_dt)
    }
    gc()

    total_rows <- total_rows + nrow(chunk_merged)

    out_path <- file.path(output_path, sprintf("part_%04d.parquet", ci))
    write_parquet(chunk_merged, out_path)
    rm(chunk_merged); gc()

    if (ci %% 5 == 0 || ci == n_chunks) {
      message(sprintf("    Chunk %d/%d done.", ci, n_chunks))
    }
  }

  message(sprintf("  Time-series: %d rows, %d concepts, %d stays.",
                  total_rows, length(all_concept_cols), n_stays))

  # Clean up temp files
  unlink(tmp_dir, recursive = TRUE)

  list(concept_cols = all_concept_cols, n_stays = n_stays)
}

# ---------------------------------------------------------------------------
# 3. Extract stay metadata
# ---------------------------------------------------------------------------

extract_stays_data <- function(dataset, dict) {
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
  avail <- names(dict)
  demo_concepts <- intersect(c("age", "sex", "height", "weight"), avail)
  clinical <- NULL
  if (length(demo_concepts) > 0) {
    clin_raw <- load_concepts(demo_concepts, src = dataset, concepts = dict)
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
  pid_col <- if ("uniquepid" %in% names(pat)) {
    "uniquepid"
  } else if ("patienthealthsystemstayid" %in% names(pat)) {
    "patienthealthsystemstayid"
  } else {
    NA
  }

  # Reconstruct approximate intime/outtime from eICU offsets.
  # eICU de-identifies absolute timestamps but provides:
  #   hospitaladmitoffset: minutes from unit admit to hospital admit (negative)
  #   unitdischargeoffset: minutes from unit admit to unit discharge (positive)
  # We anchor intime at a synthetic epoch (2000-01-01) so downstream code can
  # compute relative durations (e.g., mortality_24h windows).
  epoch <- as.POSIXct("2000-01-01 00:00:00", tz = "UTC")
  intime_vals <- if ("hospitaladmitoffset" %in% names(pat)) {
    # hospitaladmitoffset is typically negative (hospital admit before unit admit)
    # intime = epoch + hospitaladmitoffset gives hospital admission time
    # But for ICU-centric tasks, unit admission is the reference point
    epoch  # Use epoch as unit admission time (offset = 0)
  } else {
    NA
  }
  outtime_vals <- if ("unitdischargeoffset" %in% names(pat)) {
    epoch + pat$unitdischargeoffset * 60  # offset is in minutes
  } else {
    NA
  }

  df <- data.frame(
    stay_id        = pat[[id_col]],
    patient_id     = if (!is.na(pid_col)) pat[[pid_col]] else NA,
    hadm_id        = NA_integer_,
    intime         = intime_vals,
    outtime        = outtime_vals,
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

extract_mortality_data <- function(dataset, dict) {
  message("[4/6] Extracting mortality data...")

  fam <- dataset_family(dataset)
  result <- switch(fam,
    miiv  = extract_mortality_miiv(dataset, dict),
    mimic = extract_mortality_mimic(dataset, dict),
    eicu  = extract_mortality_eicu(dataset, dict),
    # For other datasets, fall back to RICU "death" concept
    extract_mortality_generic(dataset, dict)
  )

  # Ensure expected columns (new precision-aware schema + legacy)
  expected <- c("stay_id", "date_of_death", "death_time", "death_date",
                "death_time_precision", "death_source",
                "hospital_expire_flag", "dischtime", "discharge_location")
  for (col in expected) {
    if (!col %in% names(result)) result[[col]] <- NA
  }
  result <- result[, expected, drop = FALSE]

  message(sprintf("  Mortality: %d rows.", nrow(result)))
  result
}

extract_mortality_miiv <- function(dataset, dict) {
  icu <- load_table(dataset, "icustays")
  pat <- load_table(dataset, "patients")
  adm <- load_table(dataset, "admissions")

  if (is.null(icu)) return(extract_mortality_generic(dataset, dict))

  df <- data.frame(stay_id = icu$stay_id, stringsAsFactors = FALSE)

  # Get date-only dod from patients table
  if (!is.null(pat)) {
    pat_sub <- pat[, intersect(c("subject_id", "dod"), names(pat)),
                   drop = FALSE]
    icu_sub <- icu[, c("stay_id", "subject_id"), drop = FALSE]
    df <- merge(df, merge(icu_sub, pat_sub, by = "subject_id",
                          all.x = TRUE)[, c("stay_id", "dod")],
                by = "stay_id", all.x = TRUE)
    names(df)[names(df) == "dod"] <- "date_of_death"
  }

  # Get deathtime + other columns from admissions table
  if (!is.null(adm)) {
    adm_cols <- intersect(
      c("hadm_id", "hospital_expire_flag", "dischtime",
        "discharge_location", "deathtime"),
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

  # Build precision-aware death schema:
  # - death_time: exact timestamp from admissions.deathtime (preferred)
  # - death_date: date-only from patients.dod
  # - death_time_precision: "timestamp" / "date" / "unknown"
  # - death_source: which column the death info comes from
  has_deathtime <- "deathtime" %in% names(df) & !is.na(df$deathtime)
  has_dod <- "date_of_death" %in% names(df) & !is.na(df$date_of_death)

  df$death_time <- as.POSIXct(rep(NA, nrow(df)), tz = "UTC")
  df$death_date <- as.Date(rep(NA, nrow(df)))
  df$death_time_precision <- NA_character_
  df$death_source <- NA_character_

  # Prefer admissions.deathtime (exact timestamp)
  if ("deathtime" %in% names(df)) {
    idx_dt <- !is.na(df$deathtime)
    df$death_time[idx_dt] <- df$deathtime[idx_dt]
    df$death_time_precision[idx_dt] <- "timestamp"
    df$death_source[idx_dt] <- "admissions.deathtime"
  }

  # Fall back to patients.dod (date-only) for rows without deathtime
  if ("date_of_death" %in% names(df)) {
    idx_dod_only <- is.na(df$death_time) & !is.na(df$date_of_death)
    df$death_date[idx_dod_only] <- as.Date(df$date_of_death[idx_dod_only])
    df$death_time_precision[idx_dod_only] <- "date"
    df$death_source[idx_dod_only] <- "patients.dod"
  }

  # Clean up intermediate column
  df$deathtime <- NULL

  df
}

extract_mortality_mimic <- function(dataset, dict) {
  icu <- load_table(dataset, "icustays")
  pat <- load_table(dataset, "patients")
  adm <- load_table(dataset, "admissions")

  if (is.null(icu)) return(extract_mortality_generic(dataset, dict))

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
        "discharge_location", "deathtime"),
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

  # Build precision-aware death schema (same logic as miiv)
  df$death_time <- as.POSIXct(rep(NA, nrow(df)), tz = "UTC")
  df$death_date <- as.Date(rep(NA, nrow(df)))
  df$death_time_precision <- NA_character_
  df$death_source <- NA_character_

  if ("deathtime" %in% names(df)) {
    idx_dt <- !is.na(df$deathtime)
    df$death_time[idx_dt] <- df$deathtime[idx_dt]
    df$death_time_precision[idx_dt] <- "timestamp"
    df$death_source[idx_dt] <- "admissions.deathtime"
  }

  if ("date_of_death" %in% names(df)) {
    idx_dod_only <- is.na(df$death_time) & !is.na(df$date_of_death)
    df$death_date[idx_dod_only] <- as.Date(df$date_of_death[idx_dod_only])
    df$death_time_precision[idx_dod_only] <- "date"
    df$death_source[idx_dod_only] <- "patients.dod"
  }

  df$deathtime <- NULL

  df
}

extract_mortality_eicu <- function(dataset, dict) {
  pat <- load_table(dataset, "patient")
  if (is.null(pat)) return(extract_mortality_generic(dataset, dict))

  id_col <- if ("patientunitstayid" %in% names(pat)) {
    "patientunitstayid"
  } else {
    names(pat)[1]
  }

  # eICU uses status/location strings and offsets rather than timestamps.
  # Preserve the hospital mortality outcome as tri-state:
  #   1  = death evidence from discharge status/location
  #   0  = explicit alive hospital discharge status
  #   NA = unknown outcome
  # This avoids converting missing/unknown statuses into false survivors, and
  # it recovers rows where hospitaldischargestatus is missing but discharge
  # location says Death.
  normalize_text <- function(x) {
    out <- tolower(trimws(as.character(x)))
    out[out %in% c("", "na", "nan", "null", "unknown")] <- NA_character_
    out
  }
  get_text_col <- function(col) {
    if (col %in% names(pat)) {
      as.character(pat[[col]])
    } else {
      rep(NA_character_, nrow(pat))
    }
  }
  get_numeric_col <- function(col) {
    if (col %in% names(pat)) {
      suppressWarnings(as.numeric(pat[[col]]))
    } else {
      rep(NA_real_, nrow(pat))
    }
  }

  hospital_status <- get_text_col("hospitaldischargestatus")
  unit_status <- get_text_col("unitdischargestatus")
  hospital_location <- get_text_col("hospitaldischargelocation")
  unit_location <- get_text_col("unitdischargelocation")

  hospital_status_norm <- normalize_text(hospital_status)
  unit_status_norm <- normalize_text(unit_status)
  hospital_location_norm <- normalize_text(hospital_location)
  unit_location_norm <- normalize_text(unit_location)

  death_values <- c("expired", "death", "died", "deceased", "dead")
  alive_values <- c("alive")

  hospital_status_death <- hospital_status_norm %in% death_values
  unit_status_death <- unit_status_norm %in% death_values
  hospital_location_death <- hospital_location_norm %in% death_values
  unit_location_death <- unit_location_norm %in% death_values

  death_evidence <- hospital_status_death | unit_status_death |
    hospital_location_death | unit_location_death
  explicit_alive <- hospital_status_norm %in% alive_values

  hosp_flag <- rep(NA_integer_, nrow(pat))
  hosp_flag[explicit_alive] <- 0L
  hosp_flag[death_evidence] <- 1L

  discharge_location <- hospital_location
  use_unit_location <- is.na(discharge_location) | trimws(discharge_location) == ""
  discharge_location[use_unit_location] <- unit_location[use_unit_location]
  death_location <- ifelse(hospital_location_death, hospital_location,
                           ifelse(unit_location_death, unit_location, NA_character_))
  use_death_location <- death_evidence & !is.na(death_location)
  discharge_location[use_death_location] <- death_location[use_death_location]

  # Derive date_of_death/death_time from discharge offsets for death-evidence
  # rows. Hospital death evidence prefers hospitaldischargeoffset; unit death
  # evidence can use unitdischargeoffset if hospital offset is unavailable.
  # We use the same synthetic epoch as the stays extraction so that windowed
  # mortality tasks (e.g., mortality_24h) can compute whether death falls
  # within the prediction window.
  epoch <- as.POSIXct("2000-01-01 00:00:00", tz = "UTC")
  hospital_offset <- get_numeric_col("hospitaldischargeoffset")
  unit_offset <- get_numeric_col("unitdischargeoffset")

  death_offset <- rep(NA_real_, nrow(pat))
  hospital_death_evidence <- hospital_status_death | hospital_location_death
  unit_death_evidence <- unit_status_death | unit_location_death
  death_offset[hospital_death_evidence] <- hospital_offset[hospital_death_evidence]
  needs_unit_offset <- is.na(death_offset) & unit_death_evidence
  death_offset[needs_unit_offset] <- unit_offset[needs_unit_offset]
  needs_hospital_fallback <- is.na(death_offset) & death_evidence
  death_offset[needs_hospital_fallback] <- hospital_offset[needs_hospital_fallback]
  needs_unit_fallback <- is.na(death_offset) & death_evidence
  death_offset[needs_unit_fallback] <- unit_offset[needs_unit_fallback]

  dod <- as.POSIXct(rep(NA, nrow(pat)), tz = "UTC")
  has_death_offset <- death_evidence & !is.na(death_offset)
  dod[has_death_offset] <- epoch + death_offset[has_death_offset] * 60

  # Also derive dischtime for all rows from hospital offset, falling back to
  # unit offset when hospital discharge timing is unavailable.
  discharge_offset <- hospital_offset
  missing_discharge_offset <- is.na(discharge_offset)
  discharge_offset[missing_discharge_offset] <- unit_offset[missing_discharge_offset]
  dischtime <- as.POSIXct(rep(NA, nrow(pat)), tz = "UTC")
  has_discharge_offset <- !is.na(discharge_offset)
  dischtime[has_discharge_offset] <- epoch + discharge_offset[has_discharge_offset] * 60

  death_source <- rep(NA_character_, nrow(pat))
  death_source[hospital_status_death] <- "patient.hospitaldischargestatus"
  death_source[is.na(death_source) & hospital_location_death] <-
    "patient.hospitaldischargelocation"
  death_source[is.na(death_source) & unit_status_death] <- "patient.unitdischargestatus"
  death_source[is.na(death_source) & unit_location_death] <- "patient.unitdischargelocation"

  df <- data.frame(
    stay_id              = pat[[id_col]],
    date_of_death        = dod,
    hospital_expire_flag = hosp_flag,
    dischtime            = dischtime,
    discharge_location   = discharge_location,
    # Precision-aware schema: eICU timestamps are derived from offsets
    # (minute-level precision) when an offset is available.
    death_time           = dod,
    death_date           = as.Date(rep(NA, nrow(pat))),
    death_time_precision = ifelse(
      has_death_offset,
      "timestamp",
      ifelse(death_evidence, "unknown", NA_character_)
    ),
    death_source         = death_source,
    stringsAsFactors = FALSE
  )
  df
}

#' Fallback: use RICU "death" concept for datasets without detailed tables.
extract_mortality_generic <- function(dataset, dict) {
  avail <- names(dict)
  if (!"death" %in% avail) {
    message("  Warning: 'death' concept not available; returning empty mortality.")
    wins <- stay_windows(dataset, interval = hours(1L))
    n <- nrow(as.data.frame(wins))
    return(data.frame(
      stay_id              = as.data.frame(wins)[[id_var(wins)]],
      date_of_death        = as.POSIXct(rep(NA, n), tz = "UTC"),
      hospital_expire_flag = NA_integer_,
      dischtime            = as.POSIXct(rep(NA, n), tz = "UTC"),
      discharge_location   = NA_character_,
      death_time           = as.POSIXct(rep(NA, n), tz = "UTC"),
      death_date           = as.Date(rep(NA, n)),
      death_time_precision = NA_character_,
      death_source         = NA_character_,
      stringsAsFactors     = FALSE
    ))
  }

  death_raw <- load_concepts("death", src = dataset, concepts = dict)
  death_df <- as.data.frame(death_raw)
  id_name <- id_var(death_raw)
  n <- nrow(death_df)

  df <- data.frame(
    stay_id              = death_df[[id_name]],
    date_of_death        = as.POSIXct(rep(NA, n), tz = "UTC"),
    hospital_expire_flag = as.integer(death_df$death),
    dischtime            = as.POSIXct(rep(NA, n), tz = "UTC"),
    discharge_location   = NA_character_,
    death_time           = as.POSIXct(rep(NA, n), tz = "UTC"),
    death_date           = as.Date(rep(NA, n)),
    death_time_precision = NA_character_,
    death_source         = NA_character_,
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
  # Split comma-separated ICD codes (common in eICU) into separate rows
  if (nrow(df) > 0 && any(grepl(",", df$icd_code, fixed = TRUE))) {
    dt <- as.data.table(df)
    dt <- dt[, .(icd_code = trimws(unlist(strsplit(icd_code, ",", fixed = TRUE)))),
             by = .(stay_id, icd_version)]
    dt <- dt[nchar(icd_code) > 0]
    df <- as.data.frame(dt)
  }
  df
}

# ---------------------------------------------------------------------------
# 6. Write metadata
# ---------------------------------------------------------------------------

write_metadata <- function(output_dir, dataset, concept_cols,
                           raw_export_horizon_hours, n_stays) {
  message("[6/6] Writing metadata...")
  metadata <- list(
    dataset                  = dataset,
    feature_names            = as.list(concept_cols),
    n_features               = length(concept_cols),
    seq_length_hours         = raw_export_horizon_hours,
    raw_export_horizon_hours = raw_export_horizon_hours,
    n_stays                  = n_stays,
    ricu_version             = as.character(packageVersion("ricu"))
  )
  write_yaml(metadata, file.path(output_dir, "ricu_metadata.yaml"))
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

main <- function(opts) {
  dataset <- opts$dataset
  output_dir <- opts$output_dir

  raw_export_horizon_hours <- opts$raw_export_horizon_hours
  deprecated_seq_length_hours <- opts$seq_length_hours
  if (is.null(raw_export_horizon_hours)) {
    if (!is.null(deprecated_seq_length_hours)) {
      raw_export_horizon_hours <- deprecated_seq_length_hours
      warning(
        "--seq_length_hours is deprecated; use --raw_export_horizon_hours instead.",
        call. = FALSE
      )
    } else {
      raw_export_horizon_hours <- 72L
    }
  } else if (!is.null(deprecated_seq_length_hours) &&
             raw_export_horizon_hours != deprecated_seq_length_hours) {
    stop(
      paste0(
        "Conflicting values provided for --raw_export_horizon_hours and ",
        "--seq_length_hours. Use only --raw_export_horizon_hours."
      )
    )
  }

  # Validate arguments
  if (is.null(dataset) || is.null(output_dir)) {
    stop("Both --dataset and --output_dir are required.")
  }
  if (raw_export_horizon_hours <= 0) {
    stop("--raw_export_horizon_hours must be a positive integer.")
  }
  if (!dataset %in% VALID_DATASETS) {
    stop(sprintf("Invalid dataset '%s'. Valid: %s",
                 dataset, paste(VALID_DATASETS, collapse = ", ")))
  }

  # Default paths for local raw CSV files
  DEFAULT_RAW_PATHS <- c(
    miiv  = "data/raw/mimiciv",
    eicu  = "data/raw/eicu-crd"
  )

  # Check data availability; auto-import from local CSVs if needed.
  # is_data_avail() does not exist in current ricu versions.
  # as_src_env() creates an environment even without data, so we test with
  # load_dictionary() which actually validates that source data is loaded.
  data_available <- tryCatch({
    dict <- load_dictionary(src = dataset)
    length(dict) > 0
  }, error = function(e) FALSE)

  if (!data_available) {
    raw_dir <- opts$raw_data_dir
    if (is.null(raw_dir) && dataset %in% names(DEFAULT_RAW_PATHS)) {
      raw_dir <- DEFAULT_RAW_PATHS[[dataset]]
    }

    if (!is.null(raw_dir) && dir.exists(raw_dir)) {
      message(sprintf("Dataset '%s' not set up in ricu. Importing from %s ...",
                      dataset, raw_dir))
      # Point ricu's data directory at a fst/ subdirectory inside the raw
      # CSV directory. CSV contents are symlinked in so ricu can import them.
      ricu_dir <- src_data_dir(dataset)
      # Remove stale/broken symlinks before creating a new one
      link_target <- Sys.readlink(ricu_dir)
      if (!is.na(link_target) && nchar(link_target) > 0 && !dir.exists(ricu_dir)) {
        file.remove(ricu_dir)
      }
      if (!dir.exists(ricu_dir) && !file.exists(ricu_dir)) {
        # Put RICU's FST cache into a fst/ subdirectory to keep the raw
        # CSV directory clean.
        fst_dir <- file.path(raw_dir, "fst")
        dir.create(fst_dir, recursive = TRUE, showWarnings = FALSE)

        # Symlink raw CSV contents into fst/ so RICU can find them
        raw_abs <- normalizePath(raw_dir)
        for (item in list.files(raw_abs, full.names = FALSE)) {
          if (item != "fst") {
            dst <- file.path(fst_dir, item)
            if (!file.exists(dst)) {
              file.symlink(file.path(raw_abs, item), dst)
            }
          }
        }

        dir.create(dirname(ricu_dir), recursive = TRUE, showWarnings = FALSE)
        fst_abs <- normalizePath(fst_dir)
        file.symlink(fst_abs, ricu_dir)
        message(sprintf("  Symlinked %s -> %s", ricu_dir, fst_abs))
      }
      # Import tables one at a time to keep memory usage low.
      # import_src() with all tables at once can OOM on machines with <=18GB RAM.
      cfg <- load_src_cfg(dataset)

      # Safer table name discovery with fallback
      all_tables <- tryCatch(
        names(as_tbl_cfg(cfg[[dataset]])),
        error = function(e) tryCatch(
          names(as_tbl_cfg(cfg[[1]])),
          error = function(e2) names(cfg[[1]])
        )
      )

      # Only import tables actually needed for concept extraction and admin
      # data.  We bypass RICU's availability gate with parse_ricu_dictionary()
      # later, so we don't need ALL tables — just the ones concepts reference.
      fam <- dataset_family(dataset)
      if (fam %in% names(ESSENTIAL_TABLES)) {
        tables_to_import <- intersect(all_tables, ESSENTIAL_TABLES[[fam]])
      } else {
        tables_to_import <- all_tables
      }
      n_import <- length(tables_to_import)
      message(sprintf("  Importing %d/%d essential tables...",
                      n_import, length(all_tables)))
      for (i in seq_along(tables_to_import)) {
        tbl_name <- tables_to_import[i]
        message(sprintf("  [%d/%d] Importing: %s", i, n_import, tbl_name))
        tryCatch({
          import_src(dataset, tables = tbl_name)
        }, error = function(e) {
          warning(sprintf("  Failed to import '%s': %s", tbl_name,
                          conditionMessage(e)))
        })
        gc()
      }
    } else {
      stop(sprintf(
        paste0("Dataset '%s' data is not available in ricu and no raw CSV ",
               "files found. Either:\n",
               "  1. Place raw CSVs in %s\n",
               "  2. Pass --raw_data_dir /path/to/csvs\n",
               "  3. Run ricu::setup_src_data('%s') manually in R"),
        dataset,
        if (dataset %in% names(DEFAULT_RAW_PATHS)) DEFAULT_RAW_PATHS[[dataset]]
        else paste0("data/raw/", dataset),
        dataset
      ))
    }
  }

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  message(sprintf("=== RICU extraction: dataset=%s, output=%s, raw_hours=%d ===",
                  dataset, output_dir, raw_export_horizon_hours))

  # Parse the concept dictionary once, bypassing RICU's availability gate
  # which can fail even after successful import_src().
  dict <- parse_ricu_dictionary(dataset)

  # 1. Discover concepts
  concepts <- discover_ts_concepts(dataset, dict)
  if (length(concepts) == 0) {
    stop("No time-series concepts found. Check ricu setup.")
  }

  # 2. Extract timeseries (writes parquet directly to avoid full-table merge)
  ts_path <- file.path(output_dir, "ricu_timeseries.parquet")
  ts_result <- extract_timeseries(dataset, concepts, raw_export_horizon_hours, dict,
                                   output_path = ts_path)
  concept_cols <- ts_result$concept_cols
  n_stays <- ts_result$n_stays

  # 3. Extract stays
  stays <- extract_stays_data(dataset, dict)
  write_parquet(stays, file.path(output_dir, "ricu_stays.parquet"))
  rm(stays); gc()

  # 4. Extract mortality
  mortality <- extract_mortality_data(dataset, dict)
  write_parquet(mortality, file.path(output_dir, "ricu_mortality.parquet"))
  rm(mortality); gc()

  # 5. Extract diagnoses
  diagnoses <- extract_diagnoses_data(dataset)
  write_parquet(diagnoses, file.path(output_dir, "ricu_diagnoses.parquet"))
  rm(diagnoses); gc()

  # 6. Write metadata
  write_metadata(output_dir, dataset, concept_cols,
                 raw_export_horizon_hours, n_stays)

  message("=== Extraction complete. ===")
  message(sprintf("  Output: %s", output_dir))
  message(sprintf("  Concepts: %d", length(concept_cols)))
  message(sprintf("  Stays (with data): %d", n_stays))
}

main(opts)
