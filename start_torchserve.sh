#!/bin/bash

# Directly define the list of models to serve
MODELS="gender emotions beard"

# Path to the model store directory
MODEL_STORE="deployment/model_store"

# Split the MODELS string into an array
IFS=' ' read -r -a MODEL_ARRAY <<< "$MODELS"

echo "Loaded models: $MODELS"

# Initialize the models parameter for the TorchServe command
MODELS_PARAM=""

# Loop through each model name and append to the models parameter
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
    MODELS_PARAM+="$MODEL_NAME=$MODEL_NAME.mar,"
    echo "Processing model: $MODEL_NAME"
done

# Remove the trailing comma
MODELS_PARAM=${MODELS_PARAM%?}

# Command to start TorchServe with the specified models
TORCHSERVE_CMD="torchserve --start --model-store $MODEL_STORE --models $MODELS_PARAM"

echo "Starting TorchServe with command:"
echo $TORCHSERVE_CMD

# Execute the command
eval $TORCHSERVE_CMD
