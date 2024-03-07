import pytest
from unittest.mock import Mock
from classifier_handler import DynamicModelHandler
import os


@pytest.fixture
def mock_context(request):
    model_dir = request.param["model_dir"]
    pth_name = request.param["pth_name"]
    mock_ctx = Mock()
    mock_ctx.system_properties = {"model_dir": model_dir, "gpu_id": "0"}
    mock_ctx.manifest = {"model": {"serializedFile": f"{pth_name}"}}
    return mock_ctx


@pytest.mark.parametrize(
    "mock_context",
    [
        {"model_dir": "models/baldness", "pth_name": "weights/bald_weights.pth"},
        {"model_dir": "models/beard", "pth_name": "weights/beard_weights.pth"},
        {"model_dir": "models/emotions", "pth_name": "weights/emotion_model.pth"},
        {"model_dir": "models/gender", "pth_name": "weights/gender_model.pth"},
        {"model_dir": "models/glasses", "pth_name": "weights/glasses_weights.pth"},
        {"model_dir": "models/happiness", "pth_name": "weights/happy_model.pth"},
        {"model_dir": "models/race", "pth_name": "weights/race_model.pth"},
    ],
    indirect=["mock_context"],
)
def test_handler_with_mock(mock_context):
    image_path = os.path.join("test_images", "hairboy.jpg")

    def load_image_as_request_input(image_path):
        with open(image_path, "rb") as img_file:
            image_bytes = img_file.read()
        return [{"body": image_bytes}]

    req_input = load_image_as_request_input(image_path)
    handler_instance = DynamicModelHandler()
    handler_instance.initialize(mock_context)

    preprocessed_image = handler_instance.preprocess(req_input)
    result = handler_instance.inference(preprocessed_image)
    final_result = handler_instance.postprocess(result)

    assert isinstance(final_result, list), "Final result should be a list"
    print(
        "Final Result for model:",
        mock_context.system_properties["model_dir"],
        final_result,
    )
