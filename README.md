# Facial Attributes Service

This project offers a suite of services similar to Microsoft Cognitive Services, but open source and built on top of TorchServe. It provides various facial attribute predictions, such as emotions, gender, head pose, face detection, and individual typology angle (ITA).

## Getting Started

### Prerequisites

Ensure you have Python 3.8+ and TorchServe installed on your machine. If not, you can install TorchServe by following the instructions [here](https://github.com/pytorch/serve)

### Setting Up

**Downloading Model Weights**

To download the necessary model weights, run the following command:

```bash
python utils/download.py
```

**Generating Model Archives**

Generate .mar archives for deployment by executing:

```bash
bash generate_mares.sh
```

**Starting the Service**

Launch the TorchServe service with:

```bash
bash start_torchserve.sh
```

**Verifying the Service**

To verify that the service is running correctly, execute:

```bash
python processing_pipeline.py
```

**Running the Stress Test**

To run the stress test of service:

```bash
locust -f locustfile.py --host=http://localhost:8080 --web-port=[your_port]
```

After running the command, open a web browser and navigate to http://localhost:your_port to access Locust's web interface. From here, you can start the test, specify the number of users to simulate, the spawn rate, and monitor the performance metrics in real-time.


### Testing Custom Handlers

Run the following command to test custom handler for classification with context mocking:

```bash
pytest -v handlers/classifiers_handler/classifier_handler_test.py
```

## Running with Docker

### Building the Docker Image

To build the Docker image for the Facial Attributes Service, navigate to the root directory of the project and run the following command:

```bash
docker build . -t serve-facial-attributes -f deployment/Dockerfile
```

This command builds a Docker image named `serve-facial-attributes` using the Dockerfile located in the deployment directory.

### Running the Service in a Docker Container

After building the image, you can start the service in a Docker container using the following command:

```bash
docker run --gpus all -p 8080:8080 -p 8081:8081 -p 8082:8082 serve-facial-attributes
```

This command runs the Docker container with GPU support enabled (make sure your Docker setup supports GPUs), mapping ports 8080, 8081 and 8082 from the container to the host.

## API Reference

### Health Check

Check if the service is running:

```bash
curl http://localhost:8080/ping
```

### Making Predictions

To get predictions from a specific model:

```bash
curl -X POST http://localhost:8080/predictions/{model_name} -T {path_to_image}
```

## Model List

The table below summarizes the currently implemented models in the service:

| Model Name      | Model Type      | Output Description                                                                                     |
|-----------------|-----------------|--------------------------------------------------------------------------------------------------------|
| beard           | Classification  | Probabilities for beard presence: no_beard, beard                                                      |
| baldness        | Classification  | Probabilities for baldness: not_bald, bald                                                             |
| gender          | Classification  | Probabilities for gender: Female, Male                                                                 |
| glasses         | Classification  | Probabilities for glasses presence: no_glasses, glasses                                                |
| happiness       | Classification  | Probabilities for happiness: not_happy, happy                                                          |
| emotions        | Classification  | Probabilities for emotions: angry, disgust, fear, happy, neutral, sad, surprise                        |
| face_detection  | Detection       | Bounding boxes for detected faces                                                                      |
| headpose        | Regression      | Yaw, pitch, and roll angles of the head                                                                |
| ita             | Regression      | Individual Typology Angle (ITA) value                                                                  |
| race            | Classification  | Probabilities for race: Black, East Asian, Indian, Latino_Hispanic, Middle Eastern, Southeast Asian, White |
