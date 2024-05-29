from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
from typing import List
import os
import base64


TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))
if TS_IS_RUNNING:
    from arcface_model import ArcfaceModel
else:
    from .arcface_model import ArcfaceModel


class ResnetArcfaceHandler(BaseHandler):
    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        weights_path = os.path.join(model_dir, serialized_file)
        self.model = ArcfaceModel(weights_path)

    def preprocess(self, data):
        images = []
        for row in data:
            image = row.get("data") or row.get("body")
            if isinstance(image, str):
                # if the image is a string of bytesarray.
                image = Image.open(io.BytesIO(base64.b64decode(image)))
            else:
                # if the image is a list
                image = Image.open(io.BytesIO(image))
            images.append(image)
        return images

    def inference(self, data):
        pil_images = data
        outputs = self.model.predict(pil_images)
        return outputs

    def postprocess(self, data):
        return data
