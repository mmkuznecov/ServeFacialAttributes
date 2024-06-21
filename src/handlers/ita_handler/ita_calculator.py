import numpy as np
import cv2
import dlib
from typing import Optional


def extract_non_black(img: np.ndarray) -> np.ndarray:
    img_total = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
    non_black = np.where(img_total > 0)
    extract_img = []
    for i in range(len(non_black[0])):
        extract_img.append(img[non_black[0][i], non_black[1][i], :])
    extract_img = np.array(extract_img).reshape(1, -1, 3)
    return extract_img


def calc_ita(masked_img: np.ndarray) -> float:
    lab_img = cv2.cvtColor(masked_img, cv2.COLOR_RGB2Lab)
    l_img, a_img, b_img = cv2.split(lab_img)
    l_img = l_img * (100 / 255)
    b_img = b_img - 128
    ITA_img = (np.arctan((l_img - 50) / b_img) * 180) / np.pi
    a_hist, a_bins = np.histogram(ITA_img, bins=1000)
    ITA = a_bins[a_hist.argmax()]
    return round(ITA, 2)


class ITACalculator:
    def __init__(self, predictor_path: str):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def calculate_ita(self, image: np.ndarray) -> Optional[float]:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = self.detector(image_rgb)

        # Define the landmark indexes for the nose region
        nose_indexes = [31, 32, 33, 34, 35, 27]

        # Iterate over the detected faces
        for face in faces:
            # Get the facial landmarks
            landmarks = self.predictor(image_rgb, face)

            # Extract the forehead landmark coordinates
            forehead_points = []
            for index in nose_indexes:
                x = landmarks.part(index).x
                y = landmarks.part(index).y
                forehead_points.append((x, y))

            # Convert the forehead points to a NumPy array
            forehead_points = np.array(forehead_points, dtype=np.int32)

            # Create a mask for the forehead region
            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [forehead_points], 255)

            # Apply the mask to the original image
            forehead_image = cv2.bitwise_and(image, image, mask=mask)

            # Extract non-black pixels from the forehead image
            forehead_non_black = extract_non_black(forehead_image)

            # Calculate ITA for the forehead region
            ita = calc_ita(forehead_non_black)

            return ita

        return None
