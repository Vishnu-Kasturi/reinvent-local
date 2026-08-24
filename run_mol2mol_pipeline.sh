#!/usr/bin/env bash
# ==============================================================================
# run_mol2mol_pipeline.sh
# ==============================================================================
# A unified bash pipeline script that:
# 1. Activates the reinvent-qsar conda environment.
# 2. Runs the REINVENT4 mol2mol staged learning or sampling run.
# 3. Parses the output path dynamically from the configuration TOML file.
# 4. Executes the python analysis script to generate RDKit PNGs, KDE distributions,
#    Tanimoto similarity graphs, shift dumbbells, and matching CSV numbers.
# 5. Automatically copies all artifacts to the Gemini App Data artifacts directory.
#
# Usage:
#   ./run_mol2mol_pipeline.sh --target pd1_pdl1
#   ./run_mol2mol_pipeline.sh --target pd1_pdl1 --skip_run (skip REINVENT, just plot/analyze)
# ==============================================================================

set -e

# Target workspace directory
export REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$REPO_ROOT/REINVENT4:${PYTHONPATH:-}"

# Defaults
TARGET="pd1_pdl1"
CONFIG=""
LEADS_CSV=""
OUTPUT_DIR=""
RUN_NAME=""
SKIP_RUN=false
PIC50_COL=""
SOL_COL=""
SA_COL=""
TARGET_PIC50=""
TARGET_SOL=""

# Auto-detect local Gemini artifacts directory if present
ARTIFACTS_DIR=""
GEMINI_DIR="/Users/vishnukasturi/.gemini/antigravity-ide/brain/ca6de21d-68e4-4095-a68c-c9c39230f853/artifacts"
if [[ -d "$GEMINI_DIR" ]]; then
    ARTIFACTS_DIR="$GEMINI_DIR"
fi

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --target) TARGET="$2"; shift ;;
        --config) CONFIG="$2"; shift ;;
        --leads_csv) LEADS_CSV="$2"; shift ;;
        --output_dir) OUTPUT_DIR="$2"; shift ;;
        --run_name) RUN_NAME="$2"; shift ;;
        --skip_run) SKIP_RUN=true ;;
        --pic50_col) PIC50_COL="$2"; shift ;;
        --sol_col) SOL_COL="$2"; shift ;;
        --sa_col) SA_COL="$2"; shift ;;
        --target_pic50) TARGET_PIC50="$2"; shift ;;
        --target_sol) TARGET_SOL="$2"; shift ;;
        --artifacts_dir) ARTIFACTS_DIR="$2"; shift ;;
        *) echo "Unknown parameter: $1"; exit 1 ;;
    esac
    shift
done

# Ensure valid target
if [[ "$TARGET" != "pd1_pdl1" && "$TARGET" != "jak2" ]]; then
    echo "[!] Error: --target must be 'jak2' or 'pd1_pdl1'"
    exit 1
fi

# Set target-specific defaults if not provided
if [[ -z "$CONFIG" ]]; then
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        CONFIG="pd1_pdl1_mol2mol_sol_opt.toml"
    else
        CONFIG="jak2_mol2mol.toml"
    fi
fi

if [[ -z "$LEADS_CSV" ]]; then
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        LEADS_CSV="Navneet_Git/MolID_Epoch/top15_balanced_with_TL.csv"
    else
        # JAK2 default leads
        LEADS_CSV="results/jak2_mol2mol_candidates.csv" # fallback/placeholder
    fi
fi

if [[ -z "$RUN_NAME" ]]; then
    RUN_NAME="${TARGET}_mol2mol"
fi

if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="results/${RUN_NAME}_analysis"
fi

echo "========================================================================"
# Print target header
echo " RUNNING MOL2MOL PIPELINE FOR TARGET: $(echo "$TARGET" | tr '[:lower:]' '[:upper:]')"
echo "========================================================================"
echo "  Config File : REINVENT4/configs/${CONFIG}"
echo "  Leads CSV   : ${LEADS_CSV}"
echo "  Run Name    : ${RUN_NAME}"
echo "  Output Dir  : ${OUTPUT_DIR}"
echo "  Skip Run    : ${SKIP_RUN}"
echo "========================================================================"

# Activate conda environment
echo "[*] Activating conda environment 'reinvent-qsar'..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate reinvent-qsar

# 1. Execute REINVENT4 mol2mol run (unless skipped)
if [ "$SKIP_RUN" = false ]; then
    echo ""
    echo "[*] Step 1: Running REINVENT4 Mol2Mol..."
    echo "[*] Executing: reinvent -l \"../logs/${RUN_NAME}.log\" \"configs/${CONFIG}\" inside REINVENT4/"
    
    (
        cd "$REPO_ROOT/REINVENT4"
        reinvent -l "../logs/${RUN_NAME}.log" "configs/${CONFIG}"
    )
    echo "[+] REINVENT4 run completed successfully."
else
    echo ""
    echo "[*] Step 1: Skipped REINVENT4 run (using existing results)."
fi

# 2. Dynamically extract output CSV file path from TOML configuration
echo ""
echo "[*] Step 2: Locating results CSV path..."
TOML_PATH="REINVENT4/configs/${CONFIG}"
RESULTS_CSV=""

if grep -q "summary_csv_prefix" "$TOML_PATH"; then
    PREFIX=$(grep "summary_csv_prefix" "$TOML_PATH" | cut -d'"' -f2 | cut -d"'" -f2)
    # Staged learning prefixes have "_1.csv" added for the first stage
    RESULTS_CSV="${PREFIX}_1.csv"
elif grep -q "output_file" "$TOML_PATH"; then
    RESULTS_CSV=$(grep "output_file" "$TOML_PATH" | cut -d'"' -f2 | cut -d"'" -f2)
fi

# Strip the leading ../ from path if present (since we will run from REPO_ROOT)
if [[ "$RESULTS_CSV" == ../* ]]; then
    RESULTS_CSV="${RESULTS_CSV#../}"
fi

# Fallback if extraction failed
if [[ -z "$RESULTS_CSV" || ! -f "$RESULTS_CSV" ]]; then
    echo "[!] Warning: Could not locate output file via TOML config or file doesn't exist."
    # Try guessing based on target
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        RESULTS_CSV="results/pd1_pdl1_mol2mol_sol_opt_1.csv"
    else
        RESULTS_CSV="results/jak2_mol2mol_candidates.csv"
    fi
    echo "[*] Using fallback results path: ${RESULTS_CSV}"
fi

if [[ ! -f "$RESULTS_CSV" ]]; then
    echo "[ERROR] Results CSV file does not exist: ${RESULTS_CSV}"
    exit 1
fi
echo "[+] Found results CSV at: ${RESULTS_CSV}"

# 3. Execute python analysis script
echo ""
echo "[*] Step 3: Running Post-Run Python Analysis..."
echo "[*] Executing: python Preprocess/scripts/run_mol2mol_analysis.py ..."

# Determine target-specific column and threshold defaults if not overridden
if [[ -z "$PIC50_COL" ]]; then
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        PIC50_COL="PD1PDL1pIC50 (raw)"
    else
        PIC50_COL="JAK2_pIC50 (raw)"
    fi
fi

if [[ -z "$SOL_COL" ]]; then
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        SOL_COL="PD1PDL1Sol (raw)"
    else
        SOL_COL="Solubility (raw)"
    fi
fi

if [[ -z "$SA_COL" ]]; then
    SA_COL="SAScore (raw)"
fi

if [[ -z "$TARGET_PIC50" ]]; then
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        TARGET_PIC50=8.5
    else
        TARGET_PIC50=7.0
    fi
fi

if [[ -z "$TARGET_SOL" ]]; then
    if [[ "$TARGET" == "pd1_pdl1" ]]; then
        TARGET_SOL=-3.0
    else
        TARGET_SOL=-4.0
    fi
fi

# Construct Python arguments dynamically
PYTHON_ARGS=(
    --results_csv "$RESULTS_CSV"
    --leads_csv "$LEADS_CSV"
    --output_dir "$OUTPUT_DIR"
    --run_name "$RUN_NAME"
    --pic50_col "$PIC50_COL"
    --sol_col "$SOL_COL"
    --sa_col "$SA_COL"
    --target_pic50 "$TARGET_PIC50"
    --target_sol "$TARGET_SOL"
)

if [[ -n "$ARTIFACTS_DIR" ]]; then
    PYTHON_ARGS+=(--artifacts_dir "$ARTIFACTS_DIR")
fi

python Preprocess/scripts/run_mol2mol_analysis.py "${PYTHON_ARGS[@]}"

echo ""
echo "[*] Step 4: Measuring Tanimoto vs JAK2 reference (SMILES-only)..."
MEASURE_ARGS=(
    --input_csv "$RESULTS_CSV"
    --output_csv "results/${RUN_NAME}_tanimoto.csv"
)
if [[ "$TARGET" == "jak2" ]]; then
    MEASURE_ARGS+=(--reference "Preprocess/Data_jak2/data_csvs/jak2_preprocess_all.csv")
fi
python Preprocess/scripts/measure_tanimoto.py "${MEASURE_ARGS[@]}" || echo "[!] Tanimoto measurement skipped (missing deps or input)"

echo ""
echo "========================================================================"
echo "[+] Mol2Mol Pipeline finished successfully!"
echo "========================================================================"
