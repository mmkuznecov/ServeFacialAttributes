#!/bin/bash

# Load configurations from .env file
if [ -f .env ]; then
    export $(cat .env | xargs)
else
    echo ".env file not found"
    exit 1
fi

# Path to the model store directory
MODEL_STORE="deployment/model_store"

# Split the MODELS string into an array
IFS=' ' read -r -a MODEL_ARRAY <<< "$MODELS"

# Initialize the models parameter for the TorchServe command
MODELS_PARAM=""

# Loop through each model name and append to the models parameter
for MODEL_NAME in "${MODEL_ARRAY[@]}"; do
    MODELS_PARAM+="$MODEL_NAME=$MODEL_NAME.mar,"
done

# Remove the trailing comma
MODELS_PARAM=${MODELS_PARAM%?}

# Command to start TorchServe with the specified models
TORCHSERVE_CMD="torchserve --start --model-store $MODEL_STORE --models $MODELS_PARAM"

echo "Starting TorchServe with command:"
echo $TORCHSERVE_CMD

# Execute the command
eval $TORCHSERVE_CMD
