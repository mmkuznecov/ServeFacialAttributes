#!/bin/bash

# Download model weights
echo "Downloading model weights..."
python utils/download.py

# Run tests (excluding mivolo_handler_test.py)
echo "Running tests..."
pytest -vv tests/handler_tests/ --ignore tests/handler_tests/mivolo_handler_test.py

# Check the exit status of the pytest command
if [ $? -eq 0 ]; then
    echo "All tests passed successfully!"
    
    # Generate .mar files
    echo "Generating .mar files..."
    bash generate_mares.sh
    
    # Start TorchServe deployment
    echo "Starting TorchServe deployment..."
    bash start_torchserve.sh
else
    echo "Some tests failed. Please check the test output for more details."
    exit 1
fi
