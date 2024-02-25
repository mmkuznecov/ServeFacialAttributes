from torchvision import transforms, models
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
from typing import List, Union

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class CustomResnetClassifier:
    
    def __init__(self, weights: str, num_classes=1):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(weights, map_location=self.device))
        model.to(self.device)
        self.model = model
        self.model.eval()  # Set the model to evaluation mode

    def process_image(self, image: Union[str, np.ndarray, Image.Image]) -> torch.Tensor:
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        # If the image is already a PIL Image, no conversion is needed
        image_tensor = transform(image)
        return image_tensor

    def predict(self, image: Union[str, np.ndarray, Image.Image, List]) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        batch = torch.stack([self.process_image(img) for img in image]).to(self.device)
        with torch.no_grad():
            result = self.model(batch)
        return result
    
    def predict_label(self, image: Union[str, np.ndarray, Image.Image, List], mapping) -> List[dict]:
        result = self.predict(image)
        predictions = []
        if self.num_classes > 1: 
            # Multi-class classification
            probabilities = F.softmax(result, dim=1).cpu().numpy()
            for prob in probabilities:
                class_predictions = {mapping[i]: float(p) for i, p in enumerate(prob)}
                predictions.append(class_predictions)
        else:
            # Binary classification
            probabilities = torch.sigmoid(result).cpu().numpy()
            for prob in probabilities:
                # Assuming mapping contains exactly 2 classes for binary classification
                class_predictions = {mapping[0]: float(1 - prob), mapping[1]: float(prob)}
                predictions.append(class_predictions)
        return predictions