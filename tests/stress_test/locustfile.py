from locust import HttpUser, task, between
import os
import random

MODELS_LIST = [
    "beard",
    "baldness",
    "gender",
    "face_detection",
    "glasses",
    "happiness",
    "ita",
    "headpose",
    "race",
    "emotions",
]


class FacialAttributesUser(HttpUser):
    wait_time = between(
        1, 5
    )  # Simulate users waiting between 1 and 5 seconds between tasks

    def on_start(self):
        """On start, list all available models and images"""
        self.models = MODELS_LIST  # Assuming this retrieves model names correctly
        self.images = [
            os.path.join("tests/test_images", img_path)
            for img_path in os.listdir("tests/test_images")
            if img_path.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ]

    @task
    def predict_image_on_all_models(self):
        """Send the same random image to all models for prediction"""
        if not self.images:
            return

        image_path = random.choice(self.images)
        for model_name in self.models:
            self.send_prediction(model_name, image_path)

    def send_prediction(self, model_name, image_path):
        """Helper function to send prediction request"""
        url = f"/predictions/{model_name}"
        with open(image_path, "rb") as img_file:
            self.client.post(
                url,
                data=img_file.read(),
                headers={"Content-Type": "application/octet-stream"},
                name=f"POST /predictions/{model_name}",  # Name parameter groups requests in the UI
            )
