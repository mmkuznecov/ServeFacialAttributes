import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
import segmentation_models_pytorch as smp
from typing import Tuple, Dict

CLASS_COLOR_MAP: Dict[int, Tuple[int, int, int]] = {
    0: (0, 0, 0),  # Background
    1: (255, 0, 0),  # Skin
    2: (0, 255, 0),  # Left eyebrow
    3: (0, 0, 255),  # Right eyebrow
    4: (255, 255, 0),  # Left eye
    5: (255, 0, 255),  # Right eye
    6: (0, 255, 255),  # Eyeglasses
    7: (128, 0, 0),  # Left ear
    8: (0, 128, 0),  # Right ear
    9: (0, 0, 128),  # Earrings
    10: (128, 128, 0),  # Nose
    11: (128, 0, 128),  # Mouth
    12: (0, 128, 128),  # Upper lip
    13: (128, 128, 128),  # Lower lip
    14: (64, 0, 0),  # Neck
    15: (192, 0, 0),  # Necklace
    16: (64, 128, 0),  # Cloth
    17: (192, 128, 0),  # Hair
    18: (64, 0, 128),  # Hat
}


class FaceSegmentationPredictor:
    def __init__(self, model_path: str, device: torch.device) -> None:
        self.model = smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights=None, in_channels=3, classes=19
        )
        self.model.load_state_dict(torch.load(model_path, map_location=device))
        self.model.to(device)
        self.model.eval()

        self.device = device
        self.transform = Compose(
            [
                ToTensor(),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.CLASS_COLOR_MAP = CLASS_COLOR_MAP

    def predict(self, image: Image.Image) -> Image.Image:
        original_size = image.size

        # Pad the image to make its dimensions divisible by 16
        width, height = image.size
        new_width = (width + 15) // 16 * 16
        new_height = (height + 15) // 16 * 16
        padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
        padded_image.paste(
            image, ((new_width - width) // 2, (new_height - height) // 2)
        )

        # Preprocess the padded image
        input_tensor = self.transform(padded_image).unsqueeze(0).to(self.device)

        # Make the prediction
        with torch.no_grad():
            output = self.model(input_tensor)
            pred_mask = output.argmax(dim=1).squeeze().cpu().numpy()

        # Create a RGB segmentation map
        rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
        for class_index, color in CLASS_COLOR_MAP.items():
            rgb_mask[pred_mask == class_index] = color

        # Crop the RGB mask to the original image size
        rgb_mask = Image.fromarray(rgb_mask)
        rgb_mask = rgb_mask.crop(
            (
                (new_width - width) // 2,
                (new_height - height) // 2,
                (new_width + width) // 2,
                (new_height + height) // 2,
            )
        )

        # Resize the RGB mask to the original image size without interpolation
        rgb_mask = rgb_mask.resize(original_size, resample=Image.NEAREST)

        return rgb_mask
