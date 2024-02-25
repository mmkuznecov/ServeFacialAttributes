#!/bin/bash

# Define the base directory for models and where to store MAR files
MODEL_DIR="./models"
STORE_DIR="./deployment/model_store"

# Ensure the model store directory exists
mkdir -p $STORE_DIR

# Define the handler path (adjust this path as necessary)
HANDLER="basehandler.py"

# Loop through each model directory in the models directory
for MODEL_PATH in $MODEL_DIR/*; do
    if [ -d "$MODEL_PATH" ]; then
        # Extract model name from path
        MODEL_NAME=$(basename $MODEL_PATH)
        
        # Assume the weights file has a specific pattern; adjust as necessary
        WEIGHTS_FILE=$(find $MODEL_PATH/weights -type f -name "*.pth" | head -n 1)
        if [ -z "$WEIGHTS_FILE" ]; then
            echo "No weights file found for $MODEL_NAME, skipping..."
            continue
        fi
        
        # Use the first .json file found as the index_to_name file
        INDEX_TO_NAME_FILE=$(find $MODEL_PATH -type f -name "*.json" | head -n 1)
        if [ -z "$INDEX_TO_NAME_FILE" ]; then
            echo "No index_to_name.json file found for $MODEL_NAME, skipping..."
            continue
        fi
        
        # Define the output .mar file path
        MAR_FILE="$STORE_DIR/$MODEL_NAME.mar"
        
        # Command to generate the MAR file using torch-model-archiver
        torch-model-archiver --model-name $MODEL_NAME \
                             --version 1.0 \
                             --serialized-file $WEIGHTS_FILE \
                             --handler $HANDLER \
                             --extra-files $INDEX_TO_NAME_FILE,$HANDLER \
                             --export-path $STORE_DIR \
                             --force

        echo "Generated MAR file for $MODEL_NAME at $MAR_FILE"
    fi
done

echo "MAR file generation complete."
