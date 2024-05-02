import pytest
from unittest.mock import Mock
import base64


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
            context.manifest["model"] = {
                "serializedFile": request.param["serialized_file"]
            }

    return context


def load_image_as_request_input(image_path, encode_base64=False):
    with open(image_path, "rb") as img_file:
        data = img_file.read()

    if encode_base64:
        data = base64.b64encode(data).decode("utf-8")

    return [{"body": data}]
