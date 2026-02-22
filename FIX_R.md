# R Script Fixes (`scripts/preprocessing/extract_with_ricu.R`)

Bugs and data quality issues found during review against EXPERIMENT_PLAN.md.

**MIMIC-IV extraction does NOT need to be rerun** — existing output at `data/ricu_output/miiv/` is valid (114 features, 94,444 stays, 48h).

**eICU extraction must wait until bugs #1–4 are fixed.**

---

## Bugs to Fix Before eICU Extraction

### 1. CRITICAL: `extract_mortality_eicu()` missing `dict` argument (line 550)

Fallback call to `extract_mortality_generic(dataset)` is missing the required `dict` argument. Guaranteed crash if eICU `patient` table fails to load.

```r
# Line 550 — BROKEN
if (is.null(pat)) return(extract_mortality_generic(dataset))

# FIX: pass dict, and add dict to function signature
extract_mortality_eicu <- function(dataset, dict) {
  pat <- load_table(dataset, "patient")
  if (is.null(pat)) return(extract_mortality_generic(dataset, dict))
  ...
```

Also update the call site in `extract_mortality_data()` (line 452):
```r
eicu  = extract_mortality_eicu(dataset, dict),
```

### 2. CRITICAL: eICU `patient_id` uses wrong column (lines 363–369)

`patienthealthsystemstayid` is admission-level, not patient-level. For patient-level splits, `uniquepid` is the correct identifier. Patients with multiple hospital admissions could leak across train/test splits.

```r
# Lines 363–369 — WRONG priority
pid_col <- if ("patienthealthsystemstayid" %in% names(pat)) {
  "patienthealthsystemstayid"
} else if ("uniquepid" %in% names(pat)) {
  "uniquepid"
} else {
  NA
}

# FIX: swap priority — uniquepid first
pid_col <- if ("uniquepid" %in% names(pat)) {
  "uniquepid"
} else if ("patienthealthsystemstayid" %in% names(pat)) {
  "patienthealthsystemstayid"
} else {
  NA
}
```

### 3. IMPORTANT: eICU `intime`/`outtime` are NA (lines 375–376)

`mortality_24h` label builder requires `intime` for windowed mortality computation and will raise `ValueError` during Python extraction. eICU has `unitadmittime24` and `hospitaladmitoffset` / `unitdischargeoffset` in the `patient` table that could be used to reconstruct approximate timestamps.

```r
# Lines 375–376 — currently hardcoded NA
intime  = NA,
outtime = NA,

# FIX: reconstruct from eICU offsets
intime = if ("hospitaladmittime24" %in% names(pat)) pat$hospitaladmittime24 else NA,
outtime = NA,  # or compute from unitdischargeoffset
```

Alternatively, document that `mortality_24h` is not supported on eICU and adjust the experiment plan.

### 4. IMPORTANT: eICU `icd9code` contains comma-separated multi-codes (lines 716–724)

The `icd9code` column in eICU's `diagnosis` table often contains multiple ICD-9 codes separated by commas (e.g. `"410.01,410.1"`). These are stored as single strings — downstream ICD lookups will fail to match.

```r
# Lines 716–720 — no splitting
df <- data.frame(
  stay_id     = diag[[id_col]],
  icd_code    = diag[[icd_col]],  # could be "410.01,410.1"
  icd_version = if (icd_col == "icd9code") 9L else NA_integer_,
  stringsAsFactors = FALSE
)

# FIX: split comma-separated codes into separate rows
df <- data.frame(
  stay_id     = diag[[id_col]],
  icd_code    = diag[[icd_col]],
  icd_version = if (icd_col == "icd9code") 9L else NA_integer_,
  stringsAsFactors = FALSE
)
# Split comma-separated ICD codes into separate rows
if (any(grepl(",", df$icd_code, fixed = TRUE))) {
  dt <- as.data.table(df)
  dt <- dt[, .(icd_code = trimws(unlist(strsplit(icd_code, ",", fixed = TRUE)))),
           by = .(stay_id, icd_version)]
  df <- as.data.frame(dt)
}
```

Note: AKI KDIGO labels use creatinine trajectories (not ICD codes), so this bug does not affect the current task set. Fix anyway for correctness and future tasks.

---

## Data Quality Issues (All Datasets)

### 5. IMPORTANT: `los_hosp` and `los_icu` leak labels for `los_remaining` task

The 114 extracted features include `los_hosp` and `los_icu` — RICU-computed length-of-stay concepts that update at every hour. A model seeing `los_icu` during the observation window effectively knows the LOS answer.

**Fix**: Exclude these features in the Python extractor or add them to a blocklist in `constants.py`. This does NOT require rerunning the R script — handle it at the Python level.

### 6. LOW: Multiple independent `stay_windows()` calls (lines 129, 219, 585)

`stay_windows()` is called separately in `extract_timeseries()`, `extract_stays_data()`, and `extract_mortality_generic()`. If RICU state changes between calls, output files could have inconsistent stay sets. Low probability but worth computing once and passing as argument.

### 7. LOW: Batch merge holds all data.tables in memory (lines 188–198)

All batch results are kept in `batch_results` during the merge loop. Null out processed batches inside the loop to reduce peak memory:

```r
for (b in 2:n_batches) {
  merged <- merge(merged, batch_results[[b]], by = c("stay_id", "hour"), all = TRUE)
  batch_results[[b]] <- NULL
  gc()
}
```

### 8. LOW: `pat` loaded but never used in `extract_admin_miiv()` (line 298)

`load_table(dataset, "patients")` is called but the result is never used. Wasted I/O — can be removed.
