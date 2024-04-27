from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import io
import numpy as np
import cv2
import os

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from skincolor_calculator import SkinColorPredictor
else:
    from .skincolor_calculator import SkinColorPredictor


class SkinColorHandler(BaseHandler):
    """Handler for predicting skin color values."""

    def initialize(self, context):
        self.manifest = context.manifest
        properties = context.system_properties
        self.skin_color_predictor = SkinColorPredictor()

    def preprocess(self, data):
        input_data = []
        for request_data in data:
            # Extract image and mask data from the request
            image_data = request_data.get("image")
            mask_data = request_data.get("mask")

            if image_data is None or mask_data is None:
                raise ValueError(
                    "Both 'image' and 'mask' must be provided in the request."
                )

            # Convert image and mask data to PIL Image
            image = Image.open(io.BytesIO(image_data))
            mask = Image.open(io.BytesIO(mask_data))

            # Convert PIL Image to numpy array
            image = np.array(image)
            mask = np.array(mask)

            # Ensure image is in RGB format
            if image.shape[2] == 4:
                # RGBA image
                image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            elif len(image.shape) < 3 or image.shape[2] == 1:
                # Grayscale image
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Ensure mask is binary [0, 1]
            mask = (mask == 1).astype(int)

            input_data.append({"image": image, "mask": mask})

        return input_data

    def inference(self, input_data):
        results = []
        for data in input_data:
            image = data["image"]
            mask = data["mask"]
            result = self.skin_color_predictor.predict(image, mask)
            results.append(result)
        return results

    def postprocess(self, inference_output):
        processed_output = [
            {
                "lum": result["lum"],
                "hue": result["hue"],
                "lum_std": result["lum_std"],
                "hue_std": result["hue_std"],
                "a_values": result["a_values"].tolist(),
                "b_values": result["b_values"].tolist(),
            }
            for result in inference_output
        ]
        return processed_output
