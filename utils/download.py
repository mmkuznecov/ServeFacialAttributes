import os
import subprocess

model_urls = {
    "baldness": "https://sc.link/nkRS1",
    "beard": "https://sc.link/rcSlI",
    "emotions": "https://sc.link/OFYFc",
    "face_detection": "https://sc.link/Z9ZyJ",
    "gender": "https://sc.link/DoP3G",
    "glasses": "https://sc.link/x1dH4",
    "happiness": "https://sc.link/XtepI",
    "headpose": "https://sc.link/3jwoQ",
    "ita": "https://sc.link/5EJ6X",
    "race": "https://sc.link/2y1Zf",
}

models_dir = "models"


def download_model_weights(model_name, url):
    target_dir = os.path.join(models_dir, model_name, "weights")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Downloading weights for {model_name}...")
    subprocess.run(["wget", "--content-disposition", url, "-P", target_dir],
                   check=True)
    print(f"Download completed for {model_name}")


if __name__ == "__main__":
    for model_name, url in model_urls.items():
        download_model_weights(model_name, url)
