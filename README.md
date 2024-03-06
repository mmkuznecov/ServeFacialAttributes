## Service running

In order to download weights of models, please, run:

```bash
python utils/download.py
```

In order to generate `.mar` archives for your models, just run

```bash
bash generate_mares.sh
```

To run torchserve service, run:

```bash
bash start_torchserve.sh
```

To check if service works, run:

```bash
python processing_pipeline.py
```

Example of testing custom handler with context mocking:

```bash
pytest -v handlers/classifiers_handler/classifier_handler_test.py
```

## API Specification

In order to check if service as running, please, use:

```bash
curl http://localhost:8080/ping
```

To get prediction from the specific model:

```bash
curl -X POST http://localhost:8080/predictions/{model_name} -T {path_to_image}
```