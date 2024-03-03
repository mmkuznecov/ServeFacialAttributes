# #!/bin/bash

# # Define the base directory for models and where to store MAR files
# MODEL_DIR="./models"
# STORE_DIR="./deployment/model_store"

# # Ensure the model store directory exists
# mkdir -p $STORE_DIR

# ################################### CLASSIFIER MODELS ####################################

# # Define the handler and "model file" path (since we're using a dynamic handler approach)
# HANDLER="handlers/classifiers_handler/classifier_handler.py"
# MODEL_FILE="handlers/classifiers_handler/classifier_handler.py" # Dynamic handler acts as the model file in this context

# # Loop through each model directory in the models directory
# for MODEL_PATH in $MODEL_DIR/*; do
#     if [ -d "$MODEL_PATH" ]; then
#         # Extract model name from path
#         MODEL_NAME=$(basename $MODEL_PATH)
        
#         # Assume the weights file is located within a 'weights' subdirectory
#         WEIGHTS_FILE=$(find $MODEL_PATH/weights -type f -name "*.pth" | head -n 1)
#         if [ -z "$WEIGHTS_FILE" ]; then
#             echo "No weights file found for $MODEL_NAME, skipping..."
#             continue
#         fi
        
#         # Assume the index_to_name file is located directly within the model directory
#         INDEX_TO_NAME_FILE="$MODEL_PATH/index_to_name.json"
#         if [ ! -f "$INDEX_TO_NAME_FILE" ]; then
#             echo "No index_to_name.json file found for $MODEL_NAME, skipping..."
#             continue
#         fi
        
#         # Command to generate the MAR file using torch-model-archiver
#         torch-model-archiver --model-name $MODEL_NAME \
#                              --version 1.0 \
#                              --model-file $MODEL_FILE \
#                              --serialized-file $WEIGHTS_FILE \
#                              --handler $HANDLER \
#                              --extra-files $INDEX_TO_NAME_FILE,handlers/classifiers_handler/customresnetclassifier.py \
#                              --export-path $STORE_DIR \
#                              --force

#         echo "Generated MAR file for $MODEL_NAME at $STORE_DIR/$MODEL_NAME.mar"
#     fi
# done


# ################################### HEADPOSE MODEL ####################################

# # Set model specific variables
# HEADPOSE_MODEL_NAME="headpose"
# HEADPOSE_WEIGHTS_FILE="$MODEL_DIR/headpose/weights/headpose_weights.pth"
# HEADPOSE_EXTRA_FILES="$MODEL_DIR/headpose/sixdrepnet360.py,$MODEL_DIR/headpose/sixderpnet360_utils.py"

# # Check if the weights file exists
# if [ ! -f "$HEADPOSE_WEIGHTS_FILE" ]; then
#     echo "No weights file found for $HEADPOSE_MODEL_NAME, skipping..."
# else
#     # Command to generate the MAR file for headpose model
#     torch-model-archiver --model-name $HEADPOSE_MODEL_NAME \
#                          --version 1.0 \
#                          --model-file $HEADPOSE_MODEL_FILE \
#                          --serialized-file $HEADPOSE_WEIGHTS_FILE \
#                          --handler $HEADPOSE_HANDLER \
#                          --extra-files $HEADPOSE_EXTRA_FILES \
#                          --export-path $STORE_DIR \
#                          --force

#     echo "Generated MAR file for $HEADPOSE_MODEL_NAME at $STORE_DIR/$HEADPOSE_MODEL_NAME.mar"
# fi


# echo "MAR file generation complete."


#!/bin/bash

# Define the base directory for models and where to store MAR files
MODEL_DIR="./models"
STORE_DIR="./deployment/model_store"

# Ensure the model store directory exists
mkdir -p $STORE_DIR

################################### CLASSIFIER MODELS ####################################

# Define an array of model names that require index_to_name.json
declare -a CLASSIFIER_MODELS=("baldness" "beard" "emotions" "gender" "glasses" "happiness" "race")

# Define the handler and "model file" path (since we're using a dynamic handler approach)
CLASSIFIER_HANDLER="handlers/classifiers_handler"
HANDLER="$CLASSIFIER_HANDLER/classifier_handler.py"
MODEL_FILE="$CLASSIFIER_HANDLER/classifier_handler.py" # Dynamic handler acts as the model file in this context

# Loop through the classifier models
for MODEL_NAME in "${CLASSIFIER_MODELS[@]}"; do
    MODEL_PATH="$MODEL_DIR/$MODEL_NAME"
    WEIGHTS_FILE=$(find $MODEL_PATH/weights -type f -name "*.pth" | head -n 1)
    INDEX_TO_NAME_FILE="$MODEL_PATH/index_to_name.json"

    if [ -f "$WEIGHTS_FILE" ] && [ -f "$INDEX_TO_NAME_FILE" ]; then
        # Command to generate the MAR file using torch-model-archiver
        torch-model-archiver --model-name $MODEL_NAME \
                             --version 1.0 \
                             --model-file $MODEL_FILE \
                             --serialized-file $WEIGHTS_FILE \
                             --handler $HANDLER \
                             --extra-files $INDEX_TO_NAME_FILE,$CLASSIFIER_HANDLER/customresnetclassifier.py \
                             --export-path $STORE_DIR \
                             --force

        echo "Generated MAR file for $MODEL_NAME at $STORE_DIR/$MODEL_NAME.mar"
    else
        echo "Skipping $MODEL_NAME due to missing files..."
    fi
done

################################### HEADPOSE MODEL ####################################

# Variables specific to the headpose model

HEADPOSE_MODEL_NAME="headpose"
HEADPOSE_HANDLER="handlers/headpose_handler"
HEADPOSE_WEIGHTS_FILE="$MODEL_DIR/headpose/weights/headpose_weights.pth"
HEADPOSE_EXTRA_FILES="$HEADPOSE_HANDLER/sixdrepnet360.py,$HEADPOSE_HANDLER/sixderpnet360_utils.py"

if [ -f "$HEADPOSE_WEIGHTS_FILE" ]; then
    # Command to generate the MAR file for headpose model
    torch-model-archiver --model-name $HEADPOSE_MODEL_NAME \
                         --version 1.0 \
                         --model-file $HEADPOSE_HANDLER/headposehandler.py \
                         --serialized-file $HEADPOSE_WEIGHTS_FILE \
                         --handler $HEADPOSE_HANDLER/headposehandler.py \
                         --extra-files $HEADPOSE_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $HEADPOSE_MODEL_NAME at $STORE_DIR/$HEADPOSE_MODEL_NAME.mar"
else
    echo "No weights file found for $HEADPOSE_MODEL_NAME, skipping..."
fi

echo "MAR file generation complete."