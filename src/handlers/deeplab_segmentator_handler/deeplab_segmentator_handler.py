from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import numpy as np
import os
import cv2
import io
import base64
import torch

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from deeplab_segmentator import FaceSegmentationPredictor
else:
    from .deeplab_segmentator import FaceSegmentationPredictor


class FaceSegmentationHandler(BaseHandler):
    def initialize(self, context):
        """Initialize method loads the model."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        model_path = f"{model_dir}/{serialized_file}"
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FaceSegmentationPredictor(model_path, device)

    def preprocess(self, data):
        images = []
        for request_data in data:
            image_data = request_data.get("data") or request_data.get("body")
            image = Image.open(io.BytesIO(image_data)).convert("RGB")
            images.append(image)
        return images

    def inference(self, images):
        segmentation_masks = []
        for image in images:
            mask = self.model.predict(image)
            segmentation_masks.append(mask)
        return segmentation_masks

    def postprocess(self, segmentation_masks):
        # Convert segmentation masks to base64-encoded PNG format
        base64_masks = []
        for mask in segmentation_masks:
            # Convert the mask to uint8
            mask = np.array(mask).astype(np.uint8)
            # Encode the mask as PNG
            _, png_data = cv2.imencode(".png", mask)
            # Convert PNG data to base64
            base64_data = base64.b64encode(png_data).decode("utf-8")
            base64_masks.append(base64_data)

        # Prepare the response
        results = []
        for base64_mask in base64_masks:
            results.append({"segmentation_mask": base64_mask})
        return results
