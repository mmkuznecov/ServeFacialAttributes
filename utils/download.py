import os
import yaml
import subprocess


def download_weights(model_name, weights_link, weights_dir):
    os.makedirs(weights_dir, exist_ok=True)
    command = f"wget --content-disposition -P {weights_dir} {weights_link}"
    subprocess.run(command, shell=True, check=True)
    print(f"Downloaded weights for {model_name}")


models_dir = "models"
for model_name in os.listdir(models_dir):
    model_dir = os.path.join(models_dir, model_name)
    if os.path.isdir(model_dir):
        settings_file = os.path.join(model_dir, f"{model_name}_settings.yaml")
        if os.path.exists(settings_file):
            with open(settings_file, "r") as f:
                settings = yaml.safe_load(f)
            weights_link = settings.get("weights_link")
            if weights_link:
                weights_dir = os.path.join(model_dir, "weights")
                download_weights(model_name, weights_link, weights_dir)
