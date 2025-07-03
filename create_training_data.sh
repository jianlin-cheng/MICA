#!/bin/bash

set -e  # Exit on any error

echo "*****************************************************************************"
echo "Creating Training Data..."
echo "This may take significantly long time..."
echo "*****************************************************************************"

# Function to run a step with error handling
run_step() {
    local step_name="$1"
    local script_path="$2"
    local step_num="$3"
    local total_steps="$4"
    
    echo ""
    echo "🚀 STARTING [$step_num/$total_steps]: $step_name"
    echo "Script: $script_path"
    echo "Started at: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "-----------------------------------------------------------------------------"
    
    # Record step start time
    step_start=$(date +%s)
    
    if python "$script_path"; then
        step_end=$(date +%s)
        step_duration=$((step_end - step_start))
        step_minutes=$((step_duration / 60))
        step_seconds=$((step_duration % 60))
        
        echo "-----------------------------------------------------------------------------"
        echo "✅ COMPLETED [$step_num/$total_steps]: $step_name"
        echo "Duration: ${step_minutes}m ${step_seconds}s"
        echo "Completed at: $(date '+%Y-%m-%d %H:%M:%S')"
    else
        echo "-----------------------------------------------------------------------------"
        echo "❌ FAILED [$step_num/$total_steps]: $step_name"
        echo "Script: $script_path"
        echo "Failed at: $(date '+%Y-%m-%d %H:%M:%S')"
        exit 1
    fi
    
    echo "-----------------------------------------------------------------------------"
}

# Define all steps
declare -a STEPS=(
    "Creating resampled and normalized maps from original cryo-EM density maps|scripts_for_training_data/create_normalized_map.py"
    "Creating backbone labels for training|scripts_for_training_data/create_backbone_mask.py"
    "Creating carbon alpha labels for training|scripts_for_training_data/create_carbon_alpha_mask.py"
    "Creating amino acid labels for training|scripts_for_training_data/create_amino_acid_mask.py"
    "Creating AlphaFold3 encodings for training|scripts_for_training_data/create_AF3_encodings.py"
    "Creating normalized map grids for training|scripts_for_training_data/create_grids_for_normalized_map.py"
    "Creating backbone grids for training|scripts_for_training_data/create_grids_for_BB_mask.py"
    "Creating carbon alpha grids for training|scripts_for_training_data/create_grids_for_CA_mask.py"
    "Creating amino acid grids for training|scripts_for_training_data/create_grids_for_AA_mask.py"
    "Creating AlphaFold3 encoding grids for training|scripts_for_training_data/create_grids_for_AF3_encodings.py"
)

TOTAL_STEPS=${#STEPS[@]}
CURRENT_STEP=1

# Record start time
START_TIME=$(date +%s)
echo "Started at: $(date)"

# Run all steps
for step in "${STEPS[@]}"; do
    IFS='|' read -r step_name script_path <<< "$step"
    run_step "$step_name" "$script_path" "$CURRENT_STEP" "$TOTAL_STEPS"
    ((CURRENT_STEP++))
done

# Calculate and display total time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "*****************************************************************************"
echo "✓ ALL STEPS COMPLETED SUCCESSFULLY!"
echo "Total time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Completed at: $(date)"
echo "*****************************************************************************"
echo "You can delete 'Training_Dataset/Raw_Data' and 'Training_Dataset/Processed_Data' to free some space if needed"
echo "If required: copy the following command and paste: rm -rf Training_Dataset/*Data"
echo "*****************************************************************************"