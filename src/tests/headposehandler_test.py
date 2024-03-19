import pytest
from unittest.mock import Mock
from src.handlers.headpose_handler.headposehandler import SixDRepNet360Handler
import os


@pytest.fixture
def mock_context(request):
    model_dir = request.param["model_dir"]
    pth_name = request.param["pth_name"]
    mock_ctx = Mock()
    mock_ctx.system_properties = {
        "model_dir": model_dir,
        "gpu_id": "0",  # Assuming GPU is available; otherwise, use "cpu"
    }
    mock_ctx.manifest = {"model": {"serializedFile": pth_name}}
    return mock_ctx


@pytest.mark.parametrize(
    "mock_context",
    [
        {"model_dir": "models/headpose", "pth_name": "weights/headpose_weights.pth"},
    ],
    indirect=["mock_context"],
)
def test_headpose_handler_with_real_weights(mock_context):
    image_path = os.path.join("test_images", "not_bald.jpg")

    def load_image_as_request_input(image_path):
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        return [{"body": image_bytes}]

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
