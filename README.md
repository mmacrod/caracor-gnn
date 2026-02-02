<div align="center">

# CARACOR-GNN: ChARcterization of Atrial CardiOmyopathy from body surface potential maps using Graph Neural Networks

</div>

<div style="text-align: center;">
    <img src="logo.png" alt="Network Diagram">
</div>

## Overview
The CARACOR-GNN model uses Graph Neural Networks to classify the density and location of atrial cardiomyopathy, facilitating advanced cardiac analysis. This document guides you through the setup and use of the CARACOR-GNN for different classification tasks.

## Project Structure
The main branch of the project contains five files and two directories:

### Files
- **args.py**: Contains arguments to configure the network architecture such as dropout, number of layers, units per layer, number of epochs, learning rate, etc.
- **utils.py**: Utility script containing functions needed for `train.py` and `test.py`.
- **train.py**: Script for training the model.
- **test.py**: Script for testing models saved in the `models/` directory.
- **requirements.txt**: Lists packages required for the project to function.

### Directories
- **datasets/**: Contains `dataset_reader.py` for reading the graph dataset and a `processed/` subdirectory for storing processed datasets.
- **model/**: Contains `gnn_arc.py` with the GNN network architecture and `.pth` and `.yaml` files with the saved trained models. Includes pretrained models `model_den` for density classification and `model_loc` for location classification.

## Getting Started

### Installation
1. **Clone the repository:**
   ```bash
   git clone https://your-repository-url
   cd caracor-gnn
   ```
2. **Install required packages:**
   ```bash
   python3 -m pip install -r requirements.txt
   ```

### Download the Dataset
Before you begin training or testing, download the necessary dataset:
```bash
python3 datasets/download_dataset.py
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

### Training Models
- **Density Classification Training:**
  ```bash
  python3 train.py --trainset_name=trainset --valset_name=valset --testset_name=testset --batch_size=50 --gnn_units=88 --gnn_layers=4 --gnn_heads=7 --dropout=0.2 --num_epochs=300 --num_classes=3
  ```

- **Location Classification Training:**
  ```bash
  python3 train.py --trainset_name=trainset --valset_name=valset --testset_name=testset --batch_size=50 --gnn_units=110 --gnn_layers=4 --gnn_heads=6 --dropout=0.2 --num_epochs=300 --num_classes=6
  ```

## Support
For any additional questions or support, please open an issue in the repository, and we will assist you as soon as possible.

## Citation

If you use this code in your research, please cite:

### APA
> Macarulla-Rodríguez, M. et al. (2026). *CARACOR-GNN: ChARcterization of Atrial CardiOmyopathy from body surface potential maps using Graph Neural Networks*. Under review.

### BibTeX
```bibtex
@article{caracor2026,
  title   = {CARACOR-GNN: ChARcterization of Atrial CardiOmyopathy from body surface potential maps using Graph Neural Networks},
  author  = {Macarulla-Rodr\'iguez, Maria and S\'anchez, Jorge and Barrios Espinosa, Cristian and Loewe, Axel and Zacur, Ernesto and M. Climent, Andreu and Guillem, Mar\'ia S.},
  journal = {Under review},
  year    = {2026}
}





