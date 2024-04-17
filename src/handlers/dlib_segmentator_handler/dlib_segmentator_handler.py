from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import numpy as np
import os
import cv2
import io
import base64


TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from dlib_segmentator import DlibSegmentator
else:
    from .dlib_segmentator import DlibSegmentator


class DlibSegmentatorHandler(BaseHandler):

    def initialize(self, context):
        """Initialize method loads the model."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest["model"]["serializedFile"]
        landmarks_model = f"{model_dir}/{serialized_file}"
        self.model = DlibSegmentator(landmarks_model)

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

    def inference(self, imgs):
        segmentation_masks = []
        for img in imgs:
            mask = self.model.segment_face(img)
            segmentation_masks.append(mask)
        return segmentation_masks

    def postprocess(self, segmentation_masks):
        # Convert segmentation masks to base64-encoded PNG format
        base64_masks = []
        for mask in segmentation_masks:
            # Convert the mask to uint8
            mask = mask.astype(np.uint8)
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
