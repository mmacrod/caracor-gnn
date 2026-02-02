import argparse


def get_args():
    parser = argparse.ArgumentParser('Arguments for location and density ACM classifier using GNN and BSPM')

    # Dataset and model names
    parser.add_argument('--testset_name',
                        type=str)
    parser.add_argument('--model_name',
                        type=str)
    
    # Architecture args 
    parser.add_argument('--batch_size',
                        type=int,
                        default=50)  
    parser.add_argument('--dropout',
                        type=float,
                        default=0.2)
    
    args = parser.parse_args()
    return args
