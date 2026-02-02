"""
test.py

Testing entrypoint for CARACOR-GNN pretrained models.

This script loads a dataset test split, restores a pretrained checkpoint from `models/`,
and reports classification metrics (confusion matrix, accuracy, and per-class scores).
"""

import os
import sys
import torch

from scripts.gnn_arc import GNNmodel
from scripts.utils import load_dataset, read_configuration_file, test
from scripts.args import get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):

    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    
    ##############################
    # Load dataset
    print()
    print('Loading dataset...')

    test_loader = load_dataset(root=f"datasets",
                                test_filename=args.testset_name,
                                batch_size=args.batch_size)

    ##############################
    # Load model
    print()
    print(f"Loading pretrained model {args.model_name}.pth...")

    config_data = read_configuration_file(f"models/model_{args.model_name}")
    args.gnn_units = config_data["gnn_units"]
    args.gnn_layers = config_data["gnn_layers"]
    args.gnn_heads = config_data["gnn_heads"]
    args.num_classes = config_data["num_classes"]

    model = GNNmodel(args).to(device)
    model_state = torch.load(f"models/model_{args.model_name}.pth")
    model.load_state_dict(model_state)

    ##############################
    # Test
    print()
    print("Test results:")
    print()
    test(model, test_loader, device = device, print_flag = True)


if __name__ == '__main__':
    main(get_args())
