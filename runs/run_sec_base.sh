#!/bin/bash

# SEC Evaluation Script for Multiple Models
# This script runs SEC evaluation on multiple models for both base and untrain evaluation types

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
CUDA_DEVICE=7
PYTHON_EXEC="python"

# Define models and their configurations
declare -A MODELS=(
    ["qwen2.5-3b"]="qwen2.5-3b"
    ["qwen2.5-7b"]="qwen2.5-7b"
    ["deepseek-6.7b"]="deepseek-6.7b"
    ["deepseek-1.3b"]="deepseek-1.3b"
    ["seedcoder-8b"]="seedcoder-8b"
)

# Evaluation types
EVAL_TYPES=("base" "untrain")

# Function to run evaluation
run_evaluation() {
    local model_name=$1
    local model_dir=$2
    local eval_type=$3
    local output_name="sec-eval-${model_name}-lm"
    
    echo "----------------------------------------"
    echo "Running SEC evaluation for ${model_name} (${eval_type})..."
    echo "----------------------------------------"
    
    # Run SEC evaluation
    CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} ${PYTHON_EXEC} sec_eval.py \
        --model_name "${model_name}" \
        --model_type lm \
        --model_dir "${model_dir}" \
        --output_name "${output_name}" \
        --eval_type "${eval_type}"
    
    # Check if evaluation succeeded
    if [ $? -eq 0 ]; then
        echo "Evaluation completed successfully for ${model_name} (${eval_type})"
        
        # Print results
        echo "Printing results for ${model_name} (${eval_type})..."
        ${PYTHON_EXEC} print_results.py \
            --eval_name "${output_name}" \
            --eval_type "${eval_type}"
    else
        echo "ERROR: Evaluation failed for ${model_name} (${eval_type})"
        exit 1
    fi
    
    echo ""
}

# Function to check if required files exist
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check if Python scripts exist
    if [ ! -f "sec_eval_sc.py" ]; then
        echo "ERROR: sec_eval_sc.py not found!"
        exit 1
    fi
    
    if [ ! -f "print_results.py" ]; then
        echo "ERROR: print_results.py not found!"
        exit 1
    fi
    
    # Check if CUDA device is available
    if ! nvidia-smi -L | grep -q "GPU ${CUDA_DEVICE}"; then
        echo "WARNING: CUDA device ${CUDA_DEVICE} may not be available"
    fi
    
    echo "Prerequisites check passed"
    echo ""
}

# Main execution
main() {
    echo "========================================="
    echo "SEC Evaluation Script"
    echo "========================================="
    echo "Start time: $(date)"
    echo ""
    
    # Check prerequisites
    check_prerequisites
    
    # Run evaluations for each evaluation type
    for eval_type in "${EVAL_TYPES[@]}"; do
        echo "========================================="
        echo "Starting ${eval_type} evaluations..."
        echo "========================================="
        echo ""
        
        # Run evaluation for each model
        for model_name in "${!MODELS[@]}"; do
            model_dir="${MODELS[$model_name]}"
            run_evaluation "${model_name}" "${model_dir}" "${eval_type}"
        done
    done
    
    echo "========================================="
    echo "All evaluations completed!"
    echo "End time: $(date)"
    echo "========================================="
}

# Run main function
main "$@"