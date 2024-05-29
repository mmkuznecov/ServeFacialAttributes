from ts.torch_handler.base_handler import BaseHandler
from PIL import Image
import torchvision.transforms as transforms
import torch
from typing import List
import os
import numpy as np

TS_IS_RUNNING = bool(os.environ.get("TS_IS_RUNNING"))

if TS_IS_RUNNING:
    from arcface_resnet import resnet_face18
else:
    from .arcface_resnet import resnet_face18


def load_model(filepath):
    # Create the model instance
    model = resnet_face18(False)

    # Load the state dict with DataParallel prefixes
    state_dict = torch.load(filepath)

    # Remove 'module.' prefix if exists
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # Load the state dict into the model
    model.load_state_dict(new_state_dict)

    return model


class ArcfaceModel:
    def __init__(self, weights_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.resnet = load_model(weights_path)
        self.resnet.to(self.device)
        self.resnet.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((128, 128)),  # Resize the image to 128x128
                transforms.Grayscale(
                    num_output_channels=1
                ),  # Convert image to grayscale
                transforms.ToTensor(),  # Convert the PIL Image to a tensor
                transforms.Normalize(
                    mean=[0.485], std=[0.229]
                ),  # Normalize grayscale images
            ]
        )

    def predict(self, images: List[Image.Image]) -> np.ndarray:
        # Convert PIL images to tensors
        image_tensors = [self.transform(image) for image in images]

        # Stack the image tensors into a single batch
        batch_tensor = torch.stack(image_tensors).to(self.device)

        # Make predictions
        with torch.no_grad():
            output = self.resnet(batch_tensor)

        # Convert the output tensor to numpy array
        output = output.cpu().numpy()

        return output
