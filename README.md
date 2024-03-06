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

### Testing Custom Handlers

Run the following command to test custom handler for classification with context mocking:

```bash
pytest -v handlers/classifiers_handler/classifier_handler_test.py
```

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
