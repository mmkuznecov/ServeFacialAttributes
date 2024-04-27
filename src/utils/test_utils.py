import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_context(request):
    context = Mock()
    context.system_properties = {
        "gpu_id": "0",  # Assuming GPU is available; otherwise, use "cpu"
    }
    context.manifest = {}

    if hasattr(request, "param"):
        if "model_dir" in request.param:
            context.system_properties["model_dir"] = request.param["model_dir"]
        if "serialized_file" in request.param:
            context.manifest["model"] = {"serializedFile": request.param["serialized_file"]}

    return context


def load_image_as_request_input(image_path):
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    return [{"body": image_bytes}]
