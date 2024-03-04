from ts.torch_handler.base_handler import BaseHandler
from ultralytics import YOLO
from PIL import Image
import io


class YOLOv8FaceDetectionHandler(BaseHandler):
    """
    TorchServe handler for YOLOv8 face detection.
    """
    def initialize(self, context):
        """Initialize method loads the model."""
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        serialized_file = self.manifest['model']['serializedFile']
        model_pt_path = f"{model_dir}/{serialized_file}"
        self.model = YOLO(model_pt_path)

    def preprocess(self, data):
        """Preprocess the input data."""
        images = []

        for row in data:
            image = row.get("data") or row.get("body")
            image = Image.open(io.BytesIO(image))
            images.append(image)

        return images

    def inference(self, imgs):
        """Run inference on the preprocessed data."""
        results = []
        for img in imgs:
            result = self.model(img, verbose=False)
            results.append(result)
        return results

    def postprocess(self, inference_output):
        """Postprocess the inference output."""
        results = []
        for result in inference_output:
            # Convert tensor to a list of xywh format bounding boxes
            boxes = result[0].boxes.xywh.cpu().tolist()
            results.append({"boxes": boxes})
        return results
