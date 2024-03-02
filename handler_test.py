import pytest
from unittest.mock import Mock
from basehandler import DynamicModelHandler
import os


# Parametrized Mock Context fixture
@pytest.fixture
def mock_context(request):
    model_dir = request.param['model_dir']
    pth_name = request.param['pth_name']
    mock_ctx = Mock()
    mock_ctx.system_properties = {
        "model_dir": model_dir,
        "gpu_id": "0"  # Assuming a GPU is available; use "cpu" otherwise
    }
    mock_ctx.manifest = {"model": {"serializedFile": f"{pth_name}"}}
    return mock_ctx


@pytest.mark.parametrize('mock_context', [
    {'model_dir': 'models/emotions', 'pth_name': 'weights/emotion_model.pth'},
    {'model_dir': 'models/gender', 'pth_name': 'weights/gender_model.pth'},
], indirect=["mock_context"])
def test_handler_with_mock(mock_context):
    # Define the path to your real test image
    image_path = os.path.join('test_images', 'hairboy.jpg')

    # Function to load image as request input
    def load_image_as_request_input(image_path):
        with open(image_path, 'rb') as img_file:
            image_bytes = img_file.read()
        return [{"body": image_bytes}]

    # Load the image as a request input
    req_input = load_image_as_request_input(image_path)

    # Initialize and configure the handler instance
    handler_instance = DynamicModelHandler()
    handler_instance.initialize(mock_context)

    # Process the image through the handler workflow
    preprocessed_image = handler_instance.preprocess(req_input)
    result = handler_instance.inference(preprocessed_image)
    final_result = handler_instance.postprocess(result)

    # Example assertion (customize as needed based on expected output)
    # This is a placeholder assertion. Replace it with actual test conditions.
    assert isinstance(final_result, list), "Final result should be a list"

    print("Final Result for model:", mock_context.system_properties['model_dir'], final_result)