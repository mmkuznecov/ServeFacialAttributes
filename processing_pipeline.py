import aiohttp
import asyncio
import json
from io import BytesIO
from PIL import Image
import os


def crop_image(image, bbox):
    x, y, w, h = bbox
    left = x
    top = y
    right = x + w
    bottom = y + h
    cropped_image = image.crop((left, top, right, bottom))
    return cropped_image


async def predict_image_async(session, image, model_url):
    img_buffer = BytesIO()
    image.save(img_buffer, format="PNG")
    img_binary = img_buffer.getvalue()
    async with session.post(
        model_url, data=img_binary, headers={"Content-Type": "application/octet-stream"}
    ) as response:
        if response.status == 200:
            response_text = await response.text()
            try:
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                return f"Error decoding JSON response: {e}"
        else:
            return f"Error: Received status code {response.status}"


async def predict_images_async(image, service_url, models):
    async with aiohttp.ClientSession() as session:
        tasks = [
            predict_image_async(session, image, f"{service_url}{model}")
            for model in models
        ]
        predictions = await asyncio.gather(*tasks)
        results = dict(zip(models, predictions))
    return results


def predict_image(image, service_url, models):
    return asyncio.run(predict_images_async(image, service_url, models))


if __name__ == "__main__":
    service_url = "http://localhost:8080/predictions/"
    images_dir = "test_images"
    images = [os.path.join(images_dir, img_path) for img_path in os.listdir(images_dir)]
    preprocessing_models = ["face_detection"]
    models_no_crop = [
        "beard",
        "baldness",
        "gender",
        "glasses",
        "happiness",
        "ita",
        "headpose",
        "race",
        "emotions",
    ]
    models_crop = ["age"]

    for i, image_path in enumerate(images):
        image = Image.open(image_path)
        face_detection_predictions = predict_image(
            image, service_url, preprocessing_models
        )
        bboxes = face_detection_predictions["face_detection"]["boxes"]

        if len(bboxes):

            no_crop_predictions = predict_image(image, service_url, models_no_crop)

            crop_predictions = {}
            if len(models_crop):
                bbox = bboxes[0]
                cropped_image = crop_image(image, bbox)
                crop_predictions = predict_image(
                    cropped_image, service_url, models_crop
                )

            # Concatenate all predictions into a single dictionary
            all_predictions = {
                **face_detection_predictions,
                **no_crop_predictions,
                **crop_predictions,
            }

            print(f"Results for image {i+1}")
            for model, result in all_predictions.items():
                print(f"Model: {model}, Prediction: {result}")

        else:

            print(f"Faces not found on image: {image_path}")
