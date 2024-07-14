# Facial Attributes Service

This project offers a suite of services similar to Microsoft Cognitive Services, but open source and built on top of TorchServe. It provides various facial attribute predictions, such as emotions, gender, head pose, face detection, and individual typology angle (ITA).

## Updates

- **02.04.2024** - Age estimation model is added. Model is built upon [MiVOLO](https://github.com/WildChlamydia/MiVOLO) project which is currently a [SOTA](https://paperswithcode.com/task/age-estimation) in age estimation.
- **26.04.2024** - Face segmentation services are added. Two alternatives are available:
    1. The "classical" approach, built upon [dlib facial landmarks predictor with 81 points](https://github.com/codeniko/shape_predictor_81_face_landmarks), which uses cropping for segmentation.
    2. The advanced approach, utilizing the DeepLabV3Plus architecture, trained on the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset for enhanced performance.
- **02.05.2024** - Face skin color evaluator build using method provided in [Beyond Skin Tone: A Multidimensional Measure of Apparent Skin Color](https://arxiv.org/abs/2309.05148) paper.
- **31.05.2024** - ArcFace model added. Built upon [Pytorch model](https://github.com/ronghuaiyang/arcface-pytorch).

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

<!-- Run the following command to test custom handler for classification with context mocking:

```bash
pytest -vv tests/handler_tests
```

If you want to check tests code coverage, you can also run:

```bash
pytest --cov=src --cov-report=term-missing -vv tests/handler_tests
```

If you want to get an interactive `.html` coverage report run:

```bash
pytest --cov=src --cov-report=term-missing --cov-report=html -vv tests/handler_tests/
```

Also you can additionally use [nox](https://nox.thea.codes/en/stable/) for test automation. This method  -->

## Testing

This project uses [Nox](https://nox.thea.codes/en/stable/) for test automation and environment management. Nox allows you to run tests in isolated environments with specific dependencies, ensuring consistent and reliable test results.

### Using Nox (Recommended)

1. Install Nox if you haven't already:

   ```bash
   pip install nox
   ```

2. Run all tests (including MiVOLO):

   ```bash
   nox
   ```

3. Run specific test suites:

   - All tests except MiVOLO:
     ```bash
     nox -s tests
     ```

   - Only MiVOLO tests:
     ```bash
     nox -s mivolo_tests
     ```

Nox automatically manages the test environments and dependencies, ensuring that each test suite runs in its appropriate context.

### Customizing Test Environments

Nox allows you to customize test environments for different parts of your project. You can modify the `noxfile.py` to add or change test environments as needed. For example, the MiVOLO tests use a separate environment with specific dependencies:

```python
@nox.session(name="mivolo_tests", venv_backend="venv")
def mivolo_tests(session):
    # Install pytest
    session.install("pytest")

    # Install the current project and its dependencies
    session.install("-r", "requirements.txt")

    # Install specific requirements for MiVOLO
    session.install("-r", "models/age/age_requirements.txt")

    # Run MiVOLO tests
    session.run("pytest", "-vv", "tests/handler_tests/mivolo_handler_test.py")
```

**Note:** Assembly of environments for specific handlers can require a notable amount of time.

For more information on how to customize test environments and sessions, refer to the [Nox documentation](https://nox.thea.codes/en/stable/).

### Using pytest directly (Alternative)

While Nox is the recommended approach, you can still run tests directly using pytest if needed:

1. Run all tests:

   ```bash
   pytest -vv tests/handler_tests
   ```

2. Check test coverage:

   ```bash
   pytest --cov=src --cov-report=term-missing -vv tests/handler_tests
   ```

3. Generate an interactive HTML coverage report:

   ```bash
   pytest --cov=src --cov-report=term-missing --cov-report=html -vv tests/handler_tests/
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

| Model Name | Model Type | Output Description | Additional Info |
|-----------------|-----------------|--------------------------------------------------------------------------------------------------------|-----------------|
| `beard` | Classification | Probabilities for beard presence: `no_beard`, `beard` | Binary classifier trained on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) to detect facial hair presence. |
| `baldness` | Classification | Probabilities for baldness: `not_bald`, `bald` | Binary classifier trained on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) for baldness detection. |
| `gender` | Classification | Probabilities for gender: `Female`, `Male` | Binary classifier trained on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [FairFace](https://github.com/joojs/fairface) for gender classification. |
| `glasses` | Classification | Probabilities for glasses presence: `no_glasses`, `glasses` | Binary classifier trained on [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [Glasses or No Glasses](https://www.kaggle.com/datasets/jeffheaton/glasses-or-no-glasses) for glasses detection. |
| `happiness` | Classification | Probabilities for happiness: `not_happy`, `happy` | Binary classifier trained on [FER2013](https://paperswithcode.com/dataset/fer2013) and [AffectNet](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) for happiness detection. |
| `emotions` | Classification | Probabilities for emotions: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise` | Multiclass classifier trained on [FER2013](https://paperswithcode.com/dataset/fer2013) and [AffectNet](https://www.kaggle.com/datasets/noamsegal/affectnet-training-data) for emotion classification. |
| `face_detection` | Detection | Bounding boxes for detected faces | YOLOv8-nano model for face detection. Model available [here](https://drive.usercontent.google.com/download?id=1qcr9DbgsX3ryrz2uU8w4Xm3cOrRywXqb&export=download&authuser=0). |
| `headpose` | Regression | Yaw, pitch, and roll angles of the head | Regression model [6DRepNet360](https://github.com/thohemp/6DRepNet360) for head pose estimation. |
| `ita` | Calculation | Individual Typology Angle (ITA) value | Calculator for [Individual Typology Angle](https://ieeexplore.ieee.org/document/1038016). |
| `race` | Classification | Probabilities for race: `Black`, `East Asian`, `Indian`, `Latino_Hispanic`, `Middle Eastern`, `Southeast Asian`, `White` | Multiclass classifier trained on [FairFace](https://github.com/joojs/fairface) for race classification. |
| `age` | Regression | Estimated age in years | [SOTA](https://paperswithcode.com/task/age-estimation) regression model built upon [MiVOLO](https://github.com/WildChlamydia/MiVOLO) project for age estimation. |
| `dlib_face_segmentation` | Segmentation | Base64-encoded segmentation mask | The "classical" approach, built upon [dlib facial landmarks predictor with 81 points](https://github.com/codeniko/shape_predictor_81_face_landmarks), which uses cropping for segmentation. |
| `deeplab_face_segmentation` | Segmentation | Base64-encoded segmentation mask | The advanced approach, utilizing the DeepLabV3Plus architecture, trained on the [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset for enhanced performance. |
| `apparent_skincolor`  | Calculation    | Luminance (lum), hue, luminance standard deviation (lum_std), hue standard deviation (hue_std), a* values, b* values | Face skin color evaluator built using the method provided in [Beyond Skin Tone: A Multidimensional Measure of Apparent Skin Color](https://arxiv.org/abs/2309.05148) paper. | 
| `arcface` | Face Recognition | 512-dimensional face embedding vector | [ArcFace](https://arxiv.org/abs/1801.07698) model for face recognition, built upon the PyTorch implementation. Can be used for face verification and identification tasks. |