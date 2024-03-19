from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import numpy as np
import cv2
import os

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from ita_calculator import ITACalculator
else:
    from .ita_calculator import ITACalculator


class ITAHandler(BaseHandler):
    """
    Handler for calculating Individual Typology Angle (ITA).
    """

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = f"{model_dir}/{serialized_file}"

        # Initialize the ITACalculator with the model path
        self.ita_calculator = ITACalculator(model_path)

    def preprocess(self, data):
        images = []
        for request_data in data:
            image_data = request_data.get("data") or request_data.get("body")
            image = Image.open(io.BytesIO(image_data))
            image = np.array(image)
            # Ensure image is in RGB format
            if image.shape[2] == 4:  # RGBA
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) < 3 or image.shape[2] == 1:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            images.append(image)
        return images

    def inference(self, imgs, *args, **kwargs):
        ita_values = []
        for img in imgs:
            ita = self.ita_calculator.calculate_ita(img)
            ita_values.append(ita)
        return ita_values

    def postprocess(self, inference_output):
        results = [{"ita_value": ita} for ita in inference_output]
        return results
