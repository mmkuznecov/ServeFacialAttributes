import pytest
from src.handlers.headpose_handler.headposehandler import SixDRepNet360Handler
from ..test_utils import load_image_as_request_input, mock_context
import os


@pytest.mark.parametrize(
    "mock_context",
    [
        {
            "model_dir": "models/headpose",
            "serialized_file": "weights/headpose_weights.pth",
        }
    ],
    indirect=True,
)
def test_headpose_handler_with_real_weights(mock_context):
    image_path = os.path.join("tests/test_images", "not_bald.jpg")
    req_input = load_image_as_request_input(image_path)
    handler_instance = SixDRepNet360Handler()
    handler_instance.initialize(mock_context)
    preprocessed_image = handler_instance.preprocess(req_input)
    result = handler_instance.inference(preprocessed_image)
    final_result = handler_instance.postprocess(result)
    # Example validation (adjust according to expected model output)
    assert isinstance(final_result, list), "Final result should be a list"
    for output in final_result:
        assert all(
            key in output for key in ["yaw", "pitch", "roll"]
        ), "Output missing expected keys"
        assert all(
            isinstance(output[key], float) for key in ["yaw", "pitch", "roll"]
        ), "Values should be float"
    print(
        "Final Result for model:",
        mock_context.system_properties["model_dir"],
        final_result,
    )
