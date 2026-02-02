"""
utils.py

Utility functions for CARACOR-GNN evaluation.

Includes helpers for:
- Loading LMDB datasets into PyTorch Geometric DataLoaders
- Reading YAML configuration files
- Running evaluation on a test split and printing classification metrics
"""

from datasets.dataset_reader import BSPMDataset
from torch_geometric.loader import DataLoader

import torch
import yaml

  
def load_dataset(root,test_filename,batch_size=50):
    """ Load BSPM LMDB dataset and return PyG DataLoader. """

    test_dataset = []
    test_loader = []

    test_dataset = BSPMDataset(root=root,name=test_filename)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True)

    print(f"Size dataset: {len(test_dataset)} BSPMs")

    return test_loader

def read_configuration_file(filename):
    """ Read YAML file with the parameters. """

    with open(f"{filename}.yaml", 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data

def test(model, test_loader,device='cpu',print_flag=False):
    """ Test the model on the test dataset. """

    model.eval()
    correct = 0
    num_classes = model.gnn_layers[-1].out_channels
    confusion_matrix = torch.zeros(num_classes, num_classes)
    confused_cases = {}

    target_attr = "y_den" if num_classes == 3 else "y_loc" # Choose label: density or location classification

    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            batch_y = getattr(data, target_attr)
            
            output = model(data)
            _, predicted = torch.max(output, 1)

            correct += (predicted == batch_y).sum().item()

            for i in range(len(predicted)):
                confusion_matrix[batch_y[i]][predicted[i]] += 1
                if batch_y[i] != predicted[i]:
                    # Dictionary: (real, estimated)
                    key = (batch_y[i].item(), predicted[i].item())
                    if key not in confused_cases:
                        confused_cases[key] = []
                    confused_cases[key].append(data.name[i])

          
    accuracy = 100 * correct / len(test_loader.dataset)
    if print_flag:
        print()
        print('Confusion Matrix:')
        print(confusion_matrix)
        print()

        print()
        print("Classification metrics:")
        print(f"Test Accuracy: {round(accuracy)}%")
        print()
        for i in range (0,num_classes):
            print(f"Metrics for class {i}:")
            if sum(confusion_matrix[:,i]) == 0:
                spe = 0
            else:
                spe = (100*confusion_matrix[i,i] / sum(confusion_matrix[:,i])).item()
            if sum(confusion_matrix[i,:]) == 0:
                sen = 0
            else:
                sen = (100*confusion_matrix[i,i] / sum(confusion_matrix[i,:])).item()
            if spe + sen == 0:
                f1 = 0
            else:
                f1 = 2*(spe * sen)/(spe + sen)

            print(f" - Specificity({i}): {round(spe)}") # TP / (TP + FP) 
            print(f" - Sensitivity({i}): {round(sen)}") # TP / (TP + FN) 
            print(f" - F1-Score({i}):    {round(f1)}") # 2*(spe * sen)/(spe + sen)
            print()

        print()

    return accuracy

