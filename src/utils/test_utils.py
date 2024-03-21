import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_context(request):
    model_dir = request.param["model_dir"]
    serialized_file = request.param["serialized_file"]
    context = Mock()
    context.system_properties = {
        "model_dir": model_dir,
        "gpu_id": "0",  # Assuming GPU is available; otherwise, use "cpu"
    }
    context.manifest = {"model": {"serializedFile": serialized_file}}

    return context


def load_image_as_request_input(image_path):
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    return [{"body": image_bytes}]
