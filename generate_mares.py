import os
import subprocess
from typing import List, Dict


def run_command(command: List[str]) -> None:
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")


def generate_mar_file(config: Dict[str, str]) -> None:
    command = [
        "torch-model-archiver",
        "--model-name",
        config["model_name"],
        "--version",
        "1.0",
        "--model-file",
        config["model_file"],
        "--handler",
        config["handler"],
        "--export-path",
        STORE_DIR,
        "--force",
    ]

    if "serialized_file" in config:
        command.extend(["--serialized-file", config["serialized_file"]])

    if "extra_files" in config:
        command.extend(["--extra-files", config["extra_files"]])

    if "requirements_file" in config:
        command.extend(["--requirements-file", config["requirements_file"]])

    run_command(command)
    print(
        f"Generated MAR file for {config['model_name']} at {STORE_DIR}/{config['model_name']}.mar"
    )


# Define base directories
MODEL_DIR = "./models"
STORE_DIR = "./deployment/model_store"

# Ensure the model store directory exists
os.makedirs(STORE_DIR, exist_ok=True)

# Classifier models configuration
CLASSIFIER_MODELS = [
    "baldness",
    "beard",
    "emotions",
    "gender",
    "glasses",
    "happiness",
    "race",
]
CLASSIFIER_HANDLER = "src/handlers/classifiers_handler"
HANDLER = f"{CLASSIFIER_HANDLER}/classifier_handler.py"
MODEL_FILE = f"{CLASSIFIER_HANDLER}/classifier_handler.py"

# Generate MAR files for classifier models
for model_name in CLASSIFIER_MODELS:
    model_path = os.path.join(MODEL_DIR, model_name)
    weights_file = next(
        (
            os.path.join(model_path, "weights", f)
            for f in os.listdir(os.path.join(model_path, "weights"))
            if f.endswith(".pth")
        ),
        None,
    )
    index_file = os.path.join(model_path, "index_to_name.json")

    if weights_file and os.path.exists(index_file):
        config = {
            "model_name": model_name,
            "model_file": MODEL_FILE,
            "serialized_file": weights_file,
            "handler": HANDLER,
            "extra_files": f"{index_file},{CLASSIFIER_HANDLER}/customresnetclassifier.py",
        }
        generate_mar_file(config)
    else:
        print(f"Skipping {model_name} due to missing files...")

# Configuration for other models
OTHER_MODELS = [
    {
        "model_name": "headpose",
        "model_file": "src/handlers/headpose_handler/headposehandler.py",
        "handler": "src/handlers/headpose_handler/headposehandler.py",
        "weights_file": f"{MODEL_DIR}/headpose/weights/headpose_weights.pth",
        "extra_files": "src/handlers/headpose_handler/sixdrepnet360.py,src/handlers/headpose_handler/sixderpnet360_utils.py",
    },
    {
        "model_name": "face_detection",
        "model_file": "src/handlers/face_detection_handler/face_detection_handler.py",
        "handler": "src/handlers/face_detection_handler/face_detection_handler.py",
        "weights_file": f"{MODEL_DIR}/face_detection/weights/yolov8n-face.pt",
    },
    {
        "model_name": "ita",
        "model_file": "src/handlers/ita_handler/ita_handler.py",
        "handler": "src/handlers/ita_handler/ita_handler.py",
        "weights_file": f"{MODEL_DIR}/dlib/weights/shape_predictor_81_face_landmarks.dat",
        "extra_files": "src/handlers/ita_handler/ita_calculator.py",
    },
    {
        "model_name": "age",
        "model_file": "src/handlers/mivolo_handler/mivolo_handler.py",
        "handler": "src/handlers/mivolo_handler/mivolo_handler.py",
        "weights_file": f"{MODEL_DIR}/age/weights/model_age_utk_4.23.pth.tar",
        "extra_files": "src/handlers/mivolo_handler/create_timm_model.py,src/handlers/mivolo_handler/cross_bottleneck_attn.py,src/handlers/mivolo_handler/mi_volo.py,src/handlers/mivolo_handler/mivolo_model.py",
        "requirements_file": f"{MODEL_DIR}/age/age_requirements.txt",
    },
    {
        "model_name": "dlib_face_segmentation",
        "model_file": "src/handlers/dlib_segmentator_handler/dlib_segmentator_handler.py",
        "handler": "src/handlers/dlib_segmentator_handler/dlib_segmentator_handler.py",
        "weights_file": f"{MODEL_DIR}/dlib/weights/shape_predictor_81_face_landmarks.dat",
        "extra_files": "src/handlers/dlib_segmentator_handler/dlib_segmentator.py",
    },
    {
        "model_name": "deeplab_face_segmentation",
        "model_file": "src/handlers/deeplab_segmentator_handler/deeplab_segmentator_handler.py",
        "handler": "src/handlers/deeplab_segmentator_handler/deeplab_segmentator_handler.py",
        "weights_file": f"{MODEL_DIR}/deeplabv3_face/weights/deeplabv3plus_celebamask.pth",
        "extra_files": "src/handlers/deeplab_segmentator_handler/deeplab_segmentator.py",
    },
    {
        "model_name": "apparent_skincolor",
        "model_file": "src/handlers/skincolor_handler/skincolor_handler.py",
        "handler": "src/handlers/skincolor_handler/skincolor_handler.py",
        "extra_files": "src/handlers/skincolor_handler/skincolor_calculator.py",
    },
    {
        "model_name": "arcface",
        "model_file": "src/handlers/arcface_handler/arcface_handler.py",
        "handler": "src/handlers/arcface_handler/arcface_handler.py",
        "weights_file": f"{MODEL_DIR}/arcface/weights/resnet18_110_arcface.pth",
        "extra_files": "src/handlers/arcface_handler/arcface_model.py,src/handlers/arcface_handler/arcface_resnet.py",
    },
]

# Generate MAR files for other models
for model in OTHER_MODELS:
    config = {
        "model_name": model["model_name"],
        "model_file": model["model_file"],
        "handler": model["handler"],
    }

    if "weights_file" in model:
        if os.path.exists(model["weights_file"]):
            config["serialized_file"] = model["weights_file"]
        else:
            print(f"No weights file found for {model['model_name']}, skipping...")
            continue

    if "extra_files" in model:
        config["extra_files"] = model["extra_files"]

    if "requirements_file" in model:
        config["requirements_file"] = model["requirements_file"]

    generate_mar_file(config)

print("MAR file generation complete.")
