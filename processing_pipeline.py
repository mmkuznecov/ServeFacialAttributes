import aiohttp
import asyncio
import json
from io import BytesIO
from PIL import Image
import os
import numpy as np
import cv2
import base64

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

def save_colored_mask(mask_data, output_path, use_color_map=True):
    # Decode the base64-encoded segmentation mask
    decoded_mask = base64.b64decode(mask_data)
    mask_array = np.frombuffer(decoded_mask, dtype=np.uint8)
    mask = cv2.imdecode(mask_array, cv2.IMREAD_UNCHANGED)

    if use_color_map:
        # Create a color map for visualization
        color_map = {
            0: (0, 0, 0),  # Black
            20: (255, 0, 0),  # Blue
            30: (0, 255, 0),  # Green
            40: (0, 0, 255),  # Red
            50: (255, 255, 0),  # Cyan
            60: (255, 0, 255),  # Magenta
            70: (0, 255, 255),  # Yellow
            80: (128, 128, 128)  # Gray
        }
        # Create a color mask for visualization
        color_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        for pixel_value, color in color_map.items():
            color_mask[mask == pixel_value] = color
        # Save the color mask
        cv2.imwrite(output_path, color_mask)
    else:
        # Save the mask directly
        cv2.imwrite(output_path, mask)

if __name__ == "__main__":
    service_url = "http://localhost:8080/predictions/"
    images_dir = "test_images"
    output_dir = "segmentation_results"
    os.makedirs(output_dir, exist_ok=True)
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
    segmentation_models = ["dlib_face_segmentation", "deeplab_face_segmentation"]
    save_output = True

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
                crop_predictions = predict_image(cropped_image, service_url, models_crop)

            segmentation_predictions = predict_image(
                image, service_url, segmentation_models
            )

            # Concatenate all predictions into a single dictionary
            all_predictions = {
                **face_detection_predictions,
                **no_crop_predictions,
                **crop_predictions,
                **segmentation_predictions,
            }

            print(f"Results for image {i+1}")
            for model, result in all_predictions.items():
                print(f"Model: {model}, Prediction: {result}")

            if save_output:
                for model in segmentation_models:
                    if model in segmentation_predictions:
                        segmentation_mask = segmentation_predictions[model]["segmentation_mask"]
                        output_path = os.path.join(output_dir, f"{model}_mask_{i+1}.png")
                        use_color_map = model == "dlib_face_segmentation"
                        save_colored_mask(segmentation_mask, output_path, use_color_map)
                        print(f"Segmentation mask saved for {model} on image {i+1}")
        else:
            print(f"Faces not found on image: {image_path}")