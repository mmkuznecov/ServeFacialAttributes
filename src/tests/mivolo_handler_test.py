import pytest
import torch
from src.handlers.mivolo_handler.mivolo_handler import MiVOLOHandler
from src.utils.test_utils import load_image_as_request_input, mock_context


@pytest.mark.parametrize(
    "mock_context",
    [
        {
            "model_dir": "models/age",
            "serialized_file": "weights/model_age_utk_4.23.pth.tar",
            "gpu_id": 0 if torch.cuda.is_available() else None,
        }
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "image_path", ["test_images/bald.jpg", "test_images/not_bald.jpg"]
)
def test_mivolo_handler(mock_context, image_path):
    request_input = load_image_as_request_input(image_path)

    # Initialize handler instance
    handler = MiVOLOHandler()
    handler.initialize(mock_context)
    preprocessed_input = handler.preprocess(request_input)
    inference_output = handler.inference(preprocessed_input)
    postprocessed_output = handler.postprocess(inference_output)

    # Assertions to validate the output
    assert isinstance(postprocessed_output, list), "Output should be a list"
    assert len(postprocessed_output) == 1, "Output list should contain one item"
    output = postprocessed_output[0]
    assert "age" in output, "Output should contain 'age'"
    assert isinstance(output["age"], float), "'age' should be a float"
    assert 0 <= output["age"] <= 100, "'age' should be between 0 and 100"
    print(f"Test passed for image: {image_path} with predicted age: {output['age']}")
