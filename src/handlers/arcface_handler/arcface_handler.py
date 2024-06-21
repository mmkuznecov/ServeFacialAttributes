from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import numpy as np
from typing import List, Dict, Any
import os
import base64

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))
if TS_IS_RUNNING:
    from arcface_model import ArcfaceModel
else:
    from .arcface_model import ArcfaceModel


class ResnetArcfaceHandler(BaseHandler):
    def initialize(self, context: Any) -> None:
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        weights_path = os.path.join(model_dir, serialized_file)
        self.model = ArcfaceModel(weights_path)

    def preprocess(self, data: List[Dict[str, Any]]) -> List[Image.Image]:
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                image = Image.open(io.BytesIO(base64.b64decode(image)))
            else:
                image = Image.open(io.BytesIO(image))
            images.append(image)
        return images

    def inference(self, data: List[Image.Image]) -> np.ndarray:
        pil_images = data
        outputs = self.model.predict(pil_images)
        return outputs

    def postprocess(self, data: np.ndarray) -> List[List[float]]:
        result = []
        for vector in data:
            result.append(vector.tolist())
        return result
