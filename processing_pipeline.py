import aiohttp
import asyncio
import json
import os


async def predict_image_async(session, image_path, model_url):
    with open(image_path, "rb") as img_file:
        img_binary = img_file.read()

    async with session.post(
        model_url, data=img_binary, headers={"Content-Type": "application/octet-stream"}
    ) as response:
        if response.status == 200:
            response_text = await response.text()
            try:
                # Manually decode the JSON content
                return json.loads(response_text)
            except json.JSONDecodeError as e:
                return f"Error decoding JSON response: {e}"
        else:
            return f"Error: Received status code {response.status}"


async def predict_images_async(image_path, service_url, models):
    results = {}
    async with aiohttp.ClientSession() as session:
        tasks = [
            predict_image_async(session, image_path, f"{service_url}{model}")
            for model in models
        ]
        predictions = await asyncio.gather(*tasks)
        results = dict(zip(models, predictions))
    return results


def predict_image(image_path, service_url, models):
    return asyncio.run(predict_images_async(image_path, service_url, models))


if __name__ == "__main__":

    service_url = "http://localhost:8080/predictions/"
    images_dir = "test_images"
    images = [os.path.join(images_dir, img_path) for img_path in os.listdir(images_dir)]

    models = [
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
        "age"
    ]

    for image_path in images:
        print(f"Results for image {image_path}")
        predictions = predict_image(image_path, service_url, models)
        for model, result in predictions.items():
            print(f"Model: {model}, Prediction: {result}")
