#!/usr/bin/env bash
# setup_and_extract.sh — Automated extraction & preprocessing pipeline
#
# Sets up dependencies and runs the full RICU → Python → splits pipeline
# for one or more ICU datasets. Targets Debian/Ubuntu (e.g., GCP VMs).
#
# Usage:
#   ./scripts/setup_and_extract.sh              # All datasets (miiv + eicu + combined)
#   ./scripts/setup_and_extract.sh miiv          # Single dataset
#   ./scripts/setup_and_extract.sh miiv eicu     # Multiple datasets
#   ./scripts/setup_and_extract.sh combined      # Build combined + its source datasets
#   ./scripts/setup_and_extract.sh --skip-deps miiv  # Skip dependency installation

set -euo pipefail

# ---------------------------------------------------------------------------
# Colors & formatting
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

section() { echo -e "\n${BLUE}${BOLD}=== $1 ===${NC}\n"; }
info()    { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $1"; }
error()   { echo -e "${RED}[ERROR]${NC} $1"; }

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
SKIP_DEPS=false
REQUESTED_DATASETS=()
BASE_DATASETS=()

append_base_dataset() {
    local value="$1"
    if [ ${#BASE_DATASETS[@]} -gt 0 ]; then
        for existing in "${BASE_DATASETS[@]}"; do
            if [ "$existing" = "$value" ]; then
                return 0
            fi
        done
    fi
    BASE_DATASETS+=("$value")
}

for arg in "$@"; do
    case "$arg" in
        --skip-deps) SKIP_DEPS=true ;;
        miiv|eicu|combined) REQUESTED_DATASETS+=("$arg") ;;
        *)           error "Unknown argument: $arg"
                     echo "Usage: $0 [--skip-deps] [miiv] [eicu] [combined]"
                     exit 1 ;;
    esac
done

# Default to all datasets if none specified
if [ ${#REQUESTED_DATASETS[@]} -eq 0 ]; then
    REQUESTED_DATASETS=(miiv eicu combined)
fi

BUILD_COMBINED=false

for ds in "${REQUESTED_DATASETS[@]}"; do
    case "$ds" in
        miiv|eicu) append_base_dataset "$ds" ;;
        combined) BUILD_COMBINED=true ;;
    esac
done

if [ "$BUILD_COMBINED" = true ]; then
    append_base_dataset "miiv"
    append_base_dataset "eicu"
fi

# Dataset name -> raw data directory mapping (matches extract_with_ricu.R DEFAULT_RAW_PATHS)
# Uses a function instead of associative array for bash 3.x compatibility (macOS).
raw_dir_for() {
    case "$1" in
        miiv) echo "data/raw/mimiciv" ;;
        eicu) echo "data/raw/eicu-crd" ;;
    esac
}

# ---------------------------------------------------------------------------
# Ensure we're in the project root
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
info "Project root: $PROJECT_ROOT"

# ---------------------------------------------------------------------------
# Validate raw data exists
# ---------------------------------------------------------------------------
section "Validating raw data"

for ds in "${BASE_DATASETS[@]}"; do
    raw_dir="$(raw_dir_for "$ds")"
    if [ ! -d "$raw_dir" ]; then
        error "Raw data directory not found: $raw_dir"
        echo "  Please place the raw CSV files for '$ds' in $raw_dir"
        exit 1
    fi
    info "Found raw data: $raw_dir"
done

# ---------------------------------------------------------------------------
# Step 1: Install system & R dependencies (unless --skip-deps)
# ---------------------------------------------------------------------------
if [ "$SKIP_DEPS" = false ]; then
    section "Installing system dependencies"

    if ! command -v dpkg &>/dev/null; then
        error "Dependency installation requires Debian/Ubuntu (dpkg/apt not found)."
        error "Use --skip-deps and install R, system libs, and Python deps manually."
        exit 1
    fi

    # System libraries needed for R package compilation
    SYSTEM_PKGS=(r-base-core libcurl4-openssl-dev libssl-dev libudunits2-dev)
    MISSING_PKGS=()

    for pkg in "${SYSTEM_PKGS[@]}"; do
        if dpkg -s "$pkg" &>/dev/null; then
            info "Already installed: $pkg"
        else
            MISSING_PKGS+=("$pkg")
        fi
    done

    if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
        info "Installing: ${MISSING_PKGS[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${MISSING_PKGS[@]}"
    else
        info "All system packages already installed"
    fi

    # R packages
    section "Installing R packages"

    R_PKGS=(ricu arrow yaml data.table optparse units)

    for pkg in "${R_PKGS[@]}"; do
        if Rscript -e "if (!requireNamespace('$pkg', quietly=TRUE)) quit(status=1)" 2>/dev/null; then
            info "Already installed: $pkg"
        else
            info "Installing R package: $pkg"
            sudo Rscript -e "install.packages('$pkg', repos='https://cloud.r-project.org')"
        fi
    done

    # Python dependencies
    section "Installing Python dependencies"
    uv sync --dev
else
    info "Skipping dependency installation (--skip-deps)"
fi

# ---------------------------------------------------------------------------
# Step 2: Run extraction pipeline for each dataset
# ---------------------------------------------------------------------------
for ds in "${BASE_DATASETS[@]}"; do
    section "Processing dataset: $ds"

    ricu_output="data/ricu_output/$ds"
    processed_dir="data/processed/$ds"

    # --- R extraction ---
    if [ -d "$ricu_output" ] && [ "$(ls -A "$ricu_output" 2>/dev/null)" ]; then
        info "RICU output already exists: $ricu_output (skipping R extraction)"
    else
        info "Running R extraction for $ds..."
        Rscript scripts/preprocessing/extract_with_ricu.R \
            --dataset "$ds" \
            --output_dir "$ricu_output"
        info "R extraction complete: $ricu_output"
    fi

    # --- Python extraction ---
    if [ -f "$processed_dir/timeseries.parquet" ] && \
       [ -f "$processed_dir/static.parquet" ] && \
       [ -f "$processed_dir/labels.parquet" ] && \
       [ -f "$processed_dir/metadata.yaml" ]; then
        info "Processed data already exists: $processed_dir (skipping Python extraction)"
    else
        info "Running Python extraction for $ds..."
        uv run python scripts/preprocessing/extract_ricu.py dataset="$ds"
        info "Python extraction complete: $processed_dir"
    fi

    # --- Dataset preparation (splits + normalization) ---
    if [ -f "$processed_dir/splits.yaml" ] && \
       [ -f "$processed_dir/normalization_stats.yaml" ]; then
        info "Splits & normalization already exist: $processed_dir (skipping preparation)"
    else
        info "Running dataset preparation for $ds..."
        uv run python scripts/preprocessing/prepare_dataset.py dataset="$ds"
        info "Dataset preparation complete"
    fi

    info "Dataset $ds ready!"
done

if [ "$BUILD_COMBINED" = true ]; then
    section "Processing dataset: combined"

    processed_dir="data/processed/combined"

    if [ -f "$processed_dir/timeseries.parquet" ] && \
       [ -f "$processed_dir/static.parquet" ] && \
       [ -f "$processed_dir/labels.parquet" ] && \
       [ -f "$processed_dir/metadata.yaml" ] && \
       [ -f "$processed_dir/splits.yaml" ] && \
       [ -f "$processed_dir/normalization_stats.yaml" ]; then
        info "Combined dataset already exists and is prepared: $processed_dir (skipping)"
    else
        info "Creating and preparing combined dataset..."
        uv run python scripts/preprocessing/create_combined_dataset.py \
            --source data/processed/miiv data/processed/eicu \
            --names miiv eicu \
            --output "$processed_dir"
        info "Combined dataset creation complete: $processed_dir"
    fi

    info "Dataset combined ready!"
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
section "Pipeline Complete"

for ds in "${REQUESTED_DATASETS[@]}"; do
    processed_dir="data/processed/$ds"
    echo -e "  ${GREEN}$ds${NC}: $processed_dir/"
    for f in timeseries.parquet static.parquet labels.parquet metadata.yaml splits.yaml normalization_stats.yaml; do
        if [ -f "$processed_dir/$f" ]; then
            echo -e "    ${GREEN}✓${NC} $f"
        else
            echo -e "    ${RED}✗${NC} $f (missing)"
        fi
    done
done
