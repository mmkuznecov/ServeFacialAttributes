import requests


def predict_image(image_path, service_url, models):
    results = {}

    with open(image_path, 'rb') as img_file:
        img_binary = img_file.read()

        for model in models:
            model_url = f"{service_url}{model}"
            # Send the image as raw binary data
            response = requests.post(model_url, data=img_binary, headers={'Content-Type': 'application/octet-stream'})

            if response.status_code == 200:
                results[model] = response.json()
            else:
                results[model] = f"Error: Received status code {response.status_code}"

    return results


if __name__ == "__main__":
    # Example usage
    image_path = 'test_images/baldboi.jpg'
    service_url = 'http://localhost:8080/predictions/'
    models = ['emotions', 'gender', 'headpose', 'face_detection', 'ita']  # Add more model names as needed

    # Get predictions
    predictions = predict_image(image_path, service_url, models)

    # Print the predictions
    for model, result in predictions.items():
        print(f"Model: {model}, Prediction: {result}")
