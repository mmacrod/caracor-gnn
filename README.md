<div align="center">

# CARACOR-GNN: ChARcterization of Atrial CardiOmyopathy from body surface potential maps using Graph Neural Networks

</div>

<div style="text-align: center;">
    <img src="logo.png" alt="Network Diagram">
</div>

## Overview
The CARACOR-GNN model uses Graph Neural Networks to classify the density and location of atrial cardiomyopathy, facilitating advanced cardiac analysis. This document guides you through the setup and use of the CARACOR-GNN for different classification tasks.
The datasets provided for this project, available via Zenodo, correspond to body surface potential maps (BSPMs) acquired with 128 electrodes, with no missing electrodes and noise-free signals (infinite signal-to-noise ratio, SNR). These datasets are intended to serve as reference data for reproducible and controlled experimental evaluation. The datasets and pretrained models are available at Zenodo (10.5281/zenodo.18465971).

## Project Structure
The main branch of the project contains three files and three directories:

### Files
- **requirements.txt**: Lists packages required for the project to function.
- **download.py**: Script to download CARACOR-GNN dataset and pretrained models from Zenodo.
- **test.py**: Script for testing models saved in the `models/` directory.

### Directories
- **datasets/**: Contains `dataset_reader.py` for reading the graph dataset and a `processed/` subdirectory for storing processed datasets.
- **models/**: Contains `.pth` and `.yaml` files with the saved trained models. Includes pretrained models `model_den` for density classification and `model_loc` for location classification.
- **scripts/**: Contains `args.py` with the arguments to configure the classification test, `gnn_arc.py` with the GNN network architecture, and `utils.py` with functions needed for `test.py`.

## Getting Started

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://github.com/mmacrod/caracor-gnn.git
   cd caracor-gnn
   ```
2. **Install required packages:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

### Download the Dataset
Before you begin testing, download the necessary dataset and models:
```bash
python3 download.py
```

## Usage

### Testing Models
- **Density Classification Test:**
  ```bash
  python3 test.py --model_name=den --testset_name=testset
  ```

- **Location Classification Test:**
  ```bash
  python3 test.py --model_name=loc --testset_name=testset
  ```

## Support
For any additional questions or support, please open an issue in the repository, and we will assist you as soon as possible.

## Citation

If you use this code in your research, please cite:

### APA
> Macarulla-Rodríguez, M., Sánchez, J., Barrios Espinosa, C., Loewe, A., Zacur, E., Climent, A. M., & Guillem, M. S. (2026). *CARACOR-GNN: ChARcterization of Atrial CardiOmyopathy from body surface potential maps using Graph Neural Networks*. Under review.

### BibTeX
```bibtex
@article{caracor2026,
  title   = {CARACOR-GNN: ChARcterization of Atrial CardiOmyopathy from body surface potential maps using Graph Neural Networks},
  author  = {Macarulla-Rodr\'iguez, Maria and S\'anchez, Jorge and Barrios Espinosa, Cristian and Loewe, Axel and Zacur, Ernesto and M. Climent, Andreu and Guillem, Mar\'ia S.},
  journal = {Under review},
  year    = {2026}
}





