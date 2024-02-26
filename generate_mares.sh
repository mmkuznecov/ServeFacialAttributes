#!/bin/bash

# Define the base directory for models and where to store MAR files
MODEL_DIR="./models"
STORE_DIR="./deployment/model_store"

# Ensure the model store directory exists
mkdir -p $STORE_DIR

# Define the handler and "model file" path (since we're using a dynamic handler approach)
HANDLER="basehandler.py"
MODEL_FILE="basehandler.py" # Dynamic handler acts as the model file in this context

# Loop through each model directory in the models directory
for MODEL_PATH in $MODEL_DIR/*; do
    if [ -d "$MODEL_PATH" ]; then
        # Extract model name from path
        MODEL_NAME=$(basename $MODEL_PATH)
        
        # Assume the weights file is located within a 'weights' subdirectory
        WEIGHTS_FILE=$(find $MODEL_PATH/weights -type f -name "*.pth" | head -n 1)
        if [ -z "$WEIGHTS_FILE" ]; then
            echo "No weights file found for $MODEL_NAME, skipping..."
            continue
        fi
        
        # Assume the index_to_name file is located directly within the model directory
        INDEX_TO_NAME_FILE="$MODEL_PATH/index_to_name.json"
        if [ ! -f "$INDEX_TO_NAME_FILE" ]; then
            echo "No index_to_name.json file found for $MODEL_NAME, skipping..."
            continue
        fi
        
        # Command to generate the MAR file using torch-model-archiver
        torch-model-archiver --model-name $MODEL_NAME \
                             --version 1.0 \
                             --model-file $MODEL_FILE \
                             --serialized-file $WEIGHTS_FILE \
                             --handler $HANDLER \
                             --extra-files $INDEX_TO_NAME_FILE,customresnetclassifier.py \
                             --export-path $STORE_DIR \
                             --force

        echo "Generated MAR file for $MODEL_NAME at $STORE_DIR/$MODEL_NAME.mar"
    fi
done

echo "MAR file generation complete."
