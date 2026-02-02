"""
download.py

Script to download CARACOR-GNN datasets and pretrained models from Zenodo.

- Datasets are saved to: datasets/
- Models are saved to: models/
"""

import os
import requests


DATASETS_URL = "https://zenodo.org/record/XXXXXX/files/dataset.zip?download=1"
MODELS_URL = "https://zenodo.org/record/XXXXXX/files/models.zip?download=1"


def download_file(url, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    print(f"Downloading from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Saved to {output_path}\n")


def main():
    download_file(
        url=DATASETS_URL,
        output_dir="datasets",
        filename="datasets.zip",
    )

    download_file(
        url=MODELS_URL,
        output_dir="models",
        filename="models.zip",
    )


if __name__ == "__main__":
    main()
