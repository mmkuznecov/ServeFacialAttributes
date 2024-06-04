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
CLASSIFIER_HANDLER="src/handlers/classifiers_handler"
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
                             --extra-files $INDEX_TO_NAME_FILE,$INIT_FILE,$CLASSIFIER_HANDLER/customresnetclassifier.py \
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
HEADPOSE_HANDLER_DIR="src/handlers/headpose_handler"
HEADPOSE_HANDLER="$HEADPOSE_HANDLER_DIR/headposehandler.py"
HEADPOSE_WEIGHTS_FILE="$MODEL_DIR/headpose/weights/headpose_weights.pth"
HEADPOSE_EXTRA_FILES="$HEADPOSE_HANDLER_DIR/sixdrepnet360.py,$HEADPOSE_HANDLER_DIR/sixderpnet360_utils.py"

if [ -f "$HEADPOSE_WEIGHTS_FILE" ]; then
    # Command to generate the MAR file for headpose model
    torch-model-archiver --model-name $HEADPOSE_MODEL_NAME \
                         --version 1.0 \
                         --model-file $HEADPOSE_HANDLER \
                         --serialized-file $HEADPOSE_WEIGHTS_FILE \
                         --handler $HEADPOSE_HANDLER \
                         --extra-files $HEADPOSE_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $HEADPOSE_MODEL_NAME at $STORE_DIR/$HEADPOSE_MODEL_NAME.mar"
else
    echo "No weights file found for $HEADPOSE_MODEL_NAME, skipping..."
fi

################################### FACE DETECTION MODEL ####################################

FACE_DETECTION_MODEL_NAME="face_detection"
FACE_DETECTION_HANDLER_DIR="src/handlers/face_detection_handler"
FACE_DETECTION_HANDLER="$FACE_DETECTION_HANDLER_DIR/face_detection_handler.py"
FACE_DETECTION_WEIGHTS_FILE="$MODEL_DIR/face_detection/weights/yolov8n-face.pt"

if [ -f "$FACE_DETECTION_WEIGHTS_FILE" ]; then
    torch-model-archiver --model-name $FACE_DETECTION_MODEL_NAME \
                         --version 1.0 \
                         --model-file $FACE_DETECTION_HANDLER \
                         --serialized-file $FACE_DETECTION_WEIGHTS_FILE \
                         --handler $FACE_DETECTION_HANDLER \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $FACE_DETECTION_MODEL_NAME at $STORE_DIR/$FACE_DETECTION_MODEL_NAME.mar"
else
    echo "No weights file found for $FACE_DETECTION_MODEL_NAME, skipping..."
fi


################################### ITA CALCULATOR ####################################

ITA_CALCULATION_MODEL_NAME="ita"
ITA_CALCULATION_HANDLER_DIR="src/handlers/ita_handler"
ITA_CALCULATION_HANDLER="$ITA_CALCULATION_HANDLER_DIR/ita_handler.py"
LANDMARKS_MODEL="$MODEL_DIR/dlib/weights/shape_predictor_81_face_landmarks.dat" # reused in dlib segmentator
ITA_EXTRA_FILES="$ITA_CALCULATION_HANDLER_DIR/ita_calculator.py"

if [ -f "$LANDMARKS_MODEL" ]; then
    torch-model-archiver --model-name $ITA_CALCULATION_MODEL_NAME \
                         --version 1.0 \
                         --model-file $ITA_CALCULATION_HANDLER \
                         --serialized-file $LANDMARKS_MODEL \
                         --handler $ITA_CALCULATION_HANDLER \
                         --extra-files  $ITA_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $ITA_CALCULATION_MODEL_NAME at $STORE_DIR/$ITA_CALCULATION_MODEL_NAME.mar"
else
    echo "No weights file found for $ITA_CALCULATION_MODEL_NAME, skipping..."
fi

################################### MiVOLO AGE MODEL ####################################

AGE_MODEL_NAME="age"
AGE_HANDLER_DIR="src/handlers/mivolo_handler"
AGE_HANDLER="$AGE_HANDLER_DIR/mivolo_handler.py"
AGE_WEIGHTS="$MODEL_DIR/age/weights/model_age_utk_4.23.pth.tar"
AGE_SPECIFIC_REQUIREMENTS="age_requirements.txt"
AGE_REQUIREMENTS_FILE="$MODEL_DIR/age/$AGE_SPECIFIC_REQUIREMENTS"
AGE_EXTRA_FILES="$AGE_HANDLER_DIR/create_timm_model.py,$AGE_HANDLER_DIR/cross_bottleneck_attn.py,$AGE_HANDLER_DIR/mi_volo.py,$AGE_HANDLER_DIR/mivolo_model.py,$AGE_REQUIREMENTS_FILE"

if [ -f "$AGE_WEIGHTS" ]; then
    torch-model-archiver --model-name $AGE_MODEL_NAME \
                         --version 1.0 \
                         --model-file $AGE_HANDLER \
                         --serialized-file $AGE_WEIGHTS \
                         --handler $AGE_HANDLER \
                         --extra-files  $AGE_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --requirements-file $AGE_REQUIREMENTS_FILE \
                         --force

    echo "Generated MAR file for $AGE_MODEL_NAME at $STORE_DIR/$AGE_MODEL_NAME.mar"
else
    echo "No weights file found for $AGE_MODEL_NAME, skipping..."
fi

################################### DLIB SEGMENTATION MODEL ####################################

DLIB_SEGMENTATION_MODEL_NAME="dlib_face_segmentation"
DLIB_SEGMENTATION_DIR="src/handlers/dlib_segmentator_handler"
DLIB_SEGMENTATION_HANDLER="$DLIB_SEGMENTATION_DIR/dlib_segmentator_handler.py"
DLIB_SEGMENTATION_EXTRA_FILES="$DLIB_SEGMENTATION_DIR/dlib_segmentator.py"


if [ -f "$LANDMARKS_MODEL" ]; then
    torch-model-archiver --model-name $DLIB_SEGMENTATION_MODEL_NAME \
                         --version 1.0 \
                         --model-file $DLIB_SEGMENTATION_HANDLER \
                         --serialized-file $LANDMARKS_MODEL \
                         --handler $DLIB_SEGMENTATION_HANDLER \
                         --extra-files  $DLIB_SEGMENTATION_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $DLIB_SEGMENTATION_MODEL_NAME at $STORE_DIR/$DLIB_SEGMENTATION_MODEL_NAME.mar"
else
    echo "No weights file found for $DLIB_SEGMENTATION_MODEL_NAME, skipping..."
fi

################################### DEEPLABV3PLUS SEGMENTATION MODEL ####################################

DEEPLAB_SEGMENTATION_MODEL_NAME="deeplab_face_segmentation"
DEEPLAB_SEGMENTATION_DIR="src/handlers/deeplab_segmentator_handler"
DEEPLAB_SEGMENTATION_HANDLER="$DEEPLAB_SEGMENTATION_DIR/deeplab_segmentator_handler.py"
DEEPLAB_WEIGHTS="$MODEL_DIR/deeplabv3_face/weights/deeplabv3plus_celebamask.pth"
DEEPLAB_SEGMENTATION_EXTRA_FILES="$DEEPLAB_SEGMENTATION_DIR/deeplab_segmentator.py"

if [ -f "$DEEPLAB_WEIGHTS" ]; then
    torch-model-archiver --model-name $DEEPLAB_SEGMENTATION_MODEL_NAME \
                         --version 1.0 \
                         --model-file $DEEPLAB_SEGMENTATION_HANDLER \
                         --serialized-file $DEEPLAB_WEIGHTS \
                         --handler $DEEPLAB_SEGMENTATION_HANDLER \
                         --extra-files  $DEEPLAB_SEGMENTATION_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $DEEPLAB_SEGMENTATION_MODEL_NAME at $STORE_DIR/$DEEPLAB_SEGMENTATION_MODEL_NAME.mar"
else
    echo "No weights file found for $DEEPLAB_SEGMENTATION_MODEL_NAME, skipping..."
fi

################################### SKINCOLOR CALCULATOR ####################################

SKINCOLOR_CALCULATOR_NAME="apparent_skincolor"
SKINCOLOR_DIR="src/handlers/skincolor_handler"
SKINCOLOR_HANDLER="$SKINCOLOR_DIR/skincolor_handler.py"
SKINCOLOR_EXTRA_FILES="$SKINCOLOR_DIR/skincolor_calculator.py"

torch-model-archiver --model-name $SKINCOLOR_CALCULATOR_NAME \
                         --version 1.0 \
                         --model-file $SKINCOLOR_HANDLER \
                         --handler $SKINCOLOR_HANDLER \
                         --extra-files  $SKINCOLOR_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

echo "Generated MAR file for $SKINCOLOR_CALCULATOR_NAME at $STORE_DIR/$SKINCOLOR_CALCULATOR_NAME.mar"


################################### ARCFACE MODEL ####################################

ARCFACE_MODEL_NAME="arcface"
ARCFACE_DIR="src/handlers/arcface_handler"
ARCFACE_HANDLER="$ARCFACE_DIR/arcface_handler.py"
ARCFACE_WEIGHTS="$MODEL_DIR/arcface/weights/resnet18_110_arcface.pth"
ARCFACE_EXTRA_FILES="$ARCFACE_DIR/arcface_model.py,$ARCFACE_DIR/arcface_resnet.py"

if [ -f "$DEEPLAB_WEIGHTS" ]; then
    torch-model-archiver --model-name $ARCFACE_MODEL_NAME \
                         --version 1.0 \
                         --model-file $ARCFACE_HANDLER \
                         --serialized-file $ARCFACE_WEIGHTS \
                         --handler $ARCFACE_HANDLER \
                         --extra-files  $ARCFACE_EXTRA_FILES \
                         --export-path $STORE_DIR \
                         --force

    echo "Generated MAR file for $ARCFACE_MODEL_NAME at $STORE_DIR/$ARCFACE_MODEL_NAME.mar"
else
    echo "No weights file found for $ARCFACE_MODEL_NAME, skipping..."
fi

echo "MAR file generation complete."