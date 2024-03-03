## API Specification

```bash
curl http://localhost:8080/ping
```

to check if instance is okay

## Service running

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