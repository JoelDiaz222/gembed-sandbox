#!/bin/bash

# Benchmark Orchestrator Script
# Runs each benchmark multiple times and concatenates results

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/benchmark_config.yaml"
PYTHON_CMD="${SCRIPT_DIR}/venv/bin/python"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1" >&2
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1" >&2
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1" >&2
}

# Check if yq is installed (for YAML parsing)
if ! command -v yq &> /dev/null; then
    log_error "yq is not installed. Please install it: brew install yq"
    exit 1
fi

# Function to parse YAML config
get_benchmark_names() {
    yq eval '.benchmarks | keys | .[]' "$CONFIG_FILE"
}

is_benchmark_enabled() {
    local name=$1
    yq eval ".benchmarks.\"${name}\".enabled" "$CONFIG_FILE"
}

get_benchmark_script() {
    local name=$1
    yq eval ".benchmarks.\"${name}\".script" "$CONFIG_FILE"
}

get_benchmark_runs() {
    local name=$1
    yq eval ".benchmarks.\"${name}\".runs" "$CONFIG_FILE"
}

get_benchmark_sizes() {
    local name=$1
    yq eval ".benchmarks.\"${name}\".sizes | join(\" \")" "$CONFIG_FILE"
}

get_benchmark_db_size() {
    local name=$1
    local db_size=$(yq eval ".benchmarks.\"${name}\".db_size" "$CONFIG_FILE")
    if [ "$db_size" = "null" ]; then
        echo ""
    else
        echo "$db_size"
    fi
}

# Function to run a single benchmark iteration
run_benchmark_iteration() {
    local script_path=$1
    local sizes=$2
    local db_size=$3
    local run_num=$4
    
    local cmd="ulimit -n 8192 && PYTHONPATH=.:proto ${PYTHON_CMD} ${script_path} --sizes ${sizes}"
    
    if [ -n "$db_size" ]; then
        cmd="${cmd} --db-size ${db_size}"
    fi
    
    log_info "  Run ${run_num}: Executing..."
    
    # Run directly and capture exit code
    if bash -c "${cmd}"; then
        log_success "  Run ${run_num} completed"
        return 0
    else
        local exit_code=$?
        log_error "  Run ${run_num} failed with exit code ${exit_code}"
        return 1
    fi
}

# Function to concatenate CSV files
concatenate_results() {
    local output_dir=$1
    local timestamp=$2
    
    log_info "Concatenating results in ${output_dir}"
    
    # Find all CSV files with the timestamp
    local csv_files=("${output_dir}"/benchmark_*_run*.csv)
    
    if [ ${#csv_files[@]} -eq 0 ]; then
        log_warning "No CSV files found to concatenate"
        return 1
    fi
    
    local final_csv="${output_dir}/benchmark_${timestamp}.csv"
    
    # Concatenate CSV files (header from first file, then data from all)
    local first=true
    for csv_file in "${csv_files[@]}"; do
        if [ -f "$csv_file" ]; then
            if [ "$first" = true ]; then
                cat "$csv_file" > "$final_csv"
                first=false
            else
                tail -n +2 "$csv_file" >> "$final_csv"
            fi
        fi
    done
    
    log_success "Created concatenated CSV: ${final_csv}"
    
    # Clean up individual run CSV files
    for csv_file in "${csv_files[@]}"; do
        if [ -f "$csv_file" ]; then
            rm "$csv_file"
        fi
    done
    
    echo "$final_csv"
}

# Function to generate plots
generate_plots() {
    local csv_file=$1
    local output_dir=$2
    local timestamp=$3
    
    log_info "Generating plots from ${csv_file}"
    
    # Call Python script to generate plots
    ${PYTHON_CMD} -c "
import sys
sys.path.insert(0, '${SCRIPT_DIR}')
from plot_utils import generate_plots_from_csv
generate_plots_from_csv('${csv_file}', '${output_dir}', '${timestamp}')
" || {
        log_error "Failed to generate plots"
        return 1
    }
    
    log_success "Plots generated in ${output_dir}"
}

# Function to run a complete benchmark suite
run_benchmark_suite() {
    local name=$1
    
    echo ""
    echo "========================================================================"
    log_info "Running benchmark: ${name}"
    echo "========================================================================"
    
    local enabled=$(is_benchmark_enabled "$name")
    if [ "$enabled" != "true" ]; then
        log_warning "Benchmark ${name} is disabled, skipping"
        return 0
    fi
    
    local script=$(get_benchmark_script "$name")
    local runs=$(get_benchmark_runs "$name")
    local sizes=$(get_benchmark_sizes "$name")
    local db_size=$(get_benchmark_db_size "$name")
    
    local script_path="${SCRIPT_DIR}/${script}"
    
    if [ ! -f "$script_path" ]; then
        log_error "Script not found: ${script_path}"
        return 1
    fi
    
    log_info "Script: ${script}"
    log_info "Runs: ${runs}"
    log_info "Sizes: ${sizes}"
    if [ -n "$db_size" ]; then
        log_info "DB Size: ${db_size}"
    fi
    
    # Generate timestamp for this benchmark suite
    local timestamp=$(date +%Y%m%d_%H%M%S)
    
    # Determine output directory
    local output_dir="${SCRIPT_DIR}/$(dirname ${script})/output"
    
    # Handle subdirectories for multi-phase benchmarks
    for part in scenario1_{ingestion,serving} scenario1 scenario2_{ingestion,serving} scenario2 ingestion serving; do
        if [[ "$name" == *"$part"* ]]; then
            output_dir="${output_dir}/$part"
            break
        fi
    done
    
    mkdir -p "$output_dir"
    
    # Run benchmark multiple times
    local successful_runs=0
    for ((i=1; i<=runs; i++)); do
        if run_benchmark_iteration "$script_path" "$sizes" "$db_size" "$i"; then
            ((successful_runs++))
        else
            log_warning "Run ${i} failed, continuing with remaining runs"
        fi
        
        # Small delay between runs
        if [ $i -lt $runs ]; then
            sleep 2
        fi
    done
    
    if [ $successful_runs -eq 0 ]; then
        log_error "All runs failed for ${name}"
        return 1
    fi
    
    log_info "${successful_runs}/${runs} runs completed successfully"
    
    # Concatenate results
    local final_csv=$(concatenate_results "$output_dir" "$timestamp")
    
    if [ -z "$final_csv" ]; then
        log_error "Failed to concatenate results"
        return 1
    fi
    
    # Generate plots
    generate_plots "$final_csv" "$output_dir" "$timestamp"
    
    log_success "Benchmark ${name} completed!"
    echo ""
}

# Main execution
main() {
    echo "########################################################################"
    log_info "Benchmark Orchestrator"
    echo "########################################################################"
    log_info "Configuration: ${CONFIG_FILE}"
    log_info "Working Directory: ${SCRIPT_DIR}"
    echo ""
    
    # Get list of benchmarks to run
    local benchmarks
    if [ $# -gt 0 ]; then
        # Run only specified benchmarks
        benchmarks=("$@")
        log_info "Running only specified benchmarks: ${benchmarks[*]}"
    else
        # Run all benchmarks
        benchmarks=($(get_benchmark_names))
        log_info "Running all enabled benchmarks"
    fi
    
    # Run each benchmark
    local failed_benchmarks=()
    for name in "${benchmarks[@]}"; do
        if ! run_benchmark_suite "$name"; then
            failed_benchmarks+=("$name")
        fi
    done
    
    # Summary
    echo ""
    echo "########################################################################"
    if [ ${#failed_benchmarks[@]} -eq 0 ]; then
        log_success "All benchmarks completed successfully!"
    else
        log_warning "Some benchmarks failed:"
        for name in "${failed_benchmarks[@]}"; do
            log_error "  - ${name}"
        done
    fi
    echo "########################################################################"
}

# Run main function with all arguments
main "$@"
