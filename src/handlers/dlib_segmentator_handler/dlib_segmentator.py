import dlib
import numpy as np
import cv2

CHIN = [6, 7, 8, 9, 10]
RIGHT_CHEEK = [1, 2, 3, 4, 31]
LEFT_CHEEK = [15, 14, 13, 12, 35]
NOSE = [31, 32, 33, 34, 35, 27]
LEFT_EYE = [42, 43, 44, 45, 46, 47]
RIGHT_EYE = [36, 37, 38, 39, 40, 41]
FOREHEAD = [
    0, 77, 75, 76, 68, 69, 70, 71, 80, 72, 73,
    79, 74, 78, 16, 26, 25, 24, 23, 22, 27,
    21, 20, 19, 18, 17]

SEGMENTATION_COLOR_MAPPING = {
    tuple(NOSE): 60,
    tuple(LEFT_EYE): 70,
    tuple(RIGHT_EYE): 80,
    tuple(FOREHEAD): 50,
    tuple(LEFT_CHEEK): 40,
    tuple(RIGHT_CHEEK): 30,
    tuple(CHIN): 20,
}


class DlibSegmentator:
    def __init__(self, predictor_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)

    def segment_face(self, image):
        # Convert the image to RGB color space
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect faces in the image
        faces = self.detector(image_rgb)

        # Create a blank mask with the same dimensions as the input image
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Iterate over the detected faces
        for face in faces:
            # Get the facial landmarks
            landmarks = self.predictor(image_rgb, face)

            # Convert the landmarks to a NumPy array
            landmarks_points = np.array(
                [
                    (landmarks.part(i).x, landmarks.part(i).y)
                    for i in range(landmarks.num_parts)
                ],
                dtype=np.int32,
            )

            # Iterate over the segmentation color mapping
            for region_indices, color_value in SEGMENTATION_COLOR_MAPPING.items():
                # Extract the landmark points for the current region
                region_points = landmarks_points[list(region_indices)]

                # Create a mask for the current region
                region_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                cv2.fillPoly(region_mask, [region_points], color_value)

                # Combine the region mask with the overall mask based on priority
                mask = np.where(region_mask != 0, region_mask, mask)

        return mask
