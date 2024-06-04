import pytest
import os
from ..test_utils import load_image_as_request_input, mock_context
from src.handlers.arcface_handler.arcface_handler import ResnetArcfaceHandler

@pytest.mark.parametrize(
    "mock_context",
    [
        {
            "model_dir": "models/arcface",
            "serialized_file": "weights/resnet18_110_arcface.pth",
        }
    ],
    indirect=True,
)
def test_resnet_arcface_handler_with_real_weights(mock_context):
    # Set up the image input
    image_path = os.path.join("tests/test_images", "not_bald.jpg")  # Adjust the image name/path as needed
    req_input = load_image_as_request_input(image_path)
    
    # Initialize the handler
    handler_instance = ResnetArcfaceHandler()
    handler_instance.initialize(mock_context)
    
    # Process the image through the model workflow
    preprocessed_image = handler_instance.preprocess(req_input)
    embeddings = handler_instance.inference(preprocessed_image)
    final_result = handler_instance.postprocess(embeddings)
    
    # Assertions to validate outputs; adjust according to what your model outputs
    assert len(final_result[0]) == 512
    
    # Optionally, you can print the result to help in debugging
    print("Final Result for model:", mock_context.system_properties["model_dir"], final_result)