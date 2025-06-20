#!/bin/bash

# === Job: Brick Kiln Rotation Training ===
# This script launches training jobs based on the job_list defined in the main Python script
# using NOHUP and logs each output separately.

PYTHON_SCRIPT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/scripts/continent_embedding_crops.py"

# Extract job_list lines from Python file
JOB_LIST=$(awk '/job_list = \[/,/\]/ { 
    if ($0 ~ /^\s*#/) next
    if ($0 ~ /\(.*\)/) print $0 
}' "$PYTHON_SCRIPT")

# Loop through each job
while IFS= read -r line; do
    STATE=$(echo "$line" | awk -F'"' '{print $2}')
    GPU_ID=$(echo "$line" | awk -F',' '{gsub(/ /,"",$2); print $2}')
    MODEL_NAME=$(echo "$line" | awk -F',' '{gsub(/[" ]/,"",$3); print $3}')

    LOG_FILE="logs_crops/${STATE}_${MODEL_NAME}.log"
    echo "Launching: $STATE on GPU $GPU_ID with $MODEL_NAME"
    CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u "$PYTHON_SCRIPT" "$STATE" "$MODEL_NAME" > "$LOG_FILE" 2>&1 &

done <<< "$JOB_LIST"
