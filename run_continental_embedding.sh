#!/bin/bash

# === Job: Brick Kiln Rotation Training ===
# This script launches training jobs based on the job_list defined in main Python script
# using NOHUP and logs each output separately.

PYTHON_SCRIPT="/home/rishabh.mondal/Brick-Kilns-project/ijcai_2025_kilns/domain_experiment/scripts/continental_embeding.py"

# Extract job_list lines from Python file
JOB_LIST=$(awk '/job_list = \[/,/\]/ { if ($0 ~ /^\s*#/) next; if ($0 ~ /\(.*\)/) print $0 }' "$PYTHON_SCRIPT")

# Loop through each job
while read -r line; do
    STATE=$(echo "$line" | cut -d'"' -f2)
    GPU_ID=$(echo "$line" | cut -d',' -f2 | tr -d ' ')
    MODEL_NAME=$(echo "$line" | cut -d',' -f3 | tr -d ' "' )

    LOG_FILE="logs/${STATE}_${MODEL_NAME}.log"
    echo "Launching: $STATE on GPU $GPU_ID with $MODEL_NAME"
    nohup python -u "$PYTHON_SCRIPT" > "$LOG_FILE" 2>&1 &

done <<< "$JOB_LIST"
