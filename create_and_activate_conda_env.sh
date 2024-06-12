#!/bin/bash

# Get the base path to the Miniconda or Anaconda installation
CONDA_BASE_PATH=$(conda info --base)

# Check if the conda path was found
if [ -z "$CONDA_BASE_PATH" ]; then
  echo "Conda not found. Please ensure Conda is installed and available in the PATH."
  exit 1
fi

# Get the name of the source conda environment
SOURCE_CONDA_ENV=$1

# Check if the source environment name was provided
if [ -z "$SOURCE_CONDA_ENV" ]; then
  echo "Usage: source create_and_activate_conda_env.sh <SOURCE_CONDA_ENV> <POSTFIX>"
  exit 2
fi

# Get the postfix for the new environment
POSTFIX=$2

# Check if the postfix was provided
if [ -z "$POSTFIX" ]; then
  echo "Usage: source create_and_activate_conda_env.sh <SOURCE_CONDA_ENV> <POSTFIX>"
  exit 3
fi

# Check if the source environment exists
if [ ! -d "$CONDA_BASE_PATH/envs/$SOURCE_CONDA_ENV" ]; then
  echo "Conda environment '$SOURCE_CONDA_ENV' not found in $CONDA_BASE_PATH/envs"
  echo "Please ensure the correct source environment is specified"
  exit 4
fi

# Create a new environment name
NEW_CONDA_ENV="${SOURCE_CONDA_ENV}_${POSTFIX}"

# Check if the new environment already exists
if [ -d "$CONDA_BASE_PATH/envs/$NEW_CONDA_ENV" ]; then
  echo "Conda environment '$NEW_CONDA_ENV' already exists"
  echo "Choose a different postfix or delete the existing environment"
  exit 5
fi

# Clone the source environment into the new environment
conda create --name "$NEW_CONDA_ENV" --clone "$SOURCE_CONDA_ENV"
if [ $? -ne 0 ]; then
  echo "Error cloning environment '$SOURCE_CONDA_ENV' into '$NEW_CONDA_ENV'"
  exit 6
fi

# Initialize conda for the current terminal session
eval "$($CONDA_BASE_PATH/bin/conda shell.bash hook)"

# Activate the new environment
conda activate "$NEW_CONDA_ENV"

# Install dependencies for the new environment
pip install pytest
pip install -r models/age/age_requirements.txt

# Output information about the active environment
echo "Activated environment: $(conda info --envs | grep '*' | awk '{print $1}')"

# Run the tests
pytest -vv tests/handler_tests/mivolo_handler_test.py
