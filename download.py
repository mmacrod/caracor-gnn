"""
download.py

Script to download CARACOR-GNN datasets and pretrained models from Zenodo.

- Datasets are saved to: datasets/
- Models are saved to: models/
"""

import os
import requests
import zipfile

DATASETS_URL = "https://zenodo.org/record/18465971/files/dataset.zip?download=1"
MODELS_URL = "https://zenodo.org/record/18465971/files/models.zip?download=1"


def download_and_extract(url, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    print(f"Downloading from {url} ...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)


    # ---- Extract ----
    with zipfile.ZipFile(output_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    # Delete compressed file
    os.remove(output_path)

    print(f"Saved to {output_dir}")


def main():
    download_and_extract(
        url=DATASETS_URL,
        output_dir="datasets/processed",
        filename="dataset.zip",
    )

    download_and_extract(
        url=MODELS_URL,
        output_dir="models",
        filename="models.zip",
    )

    print()
    print("Done!")


if __name__ == "__main__":
    main()
