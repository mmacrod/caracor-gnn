"""
dataset_reader.py

PyTorch Geometric dataset wrapper for CARACOR-GNN.

This module defines `BSPMDataset`, which loads graph samples stored in LMDB files
(pickled PyG `Data` objects) from the `datasets/processed/` directory.
"""

from torch_geometric.data import Dataset
import lmdb
import pickle


class BSPMDataset(Dataset):
    def __init__(self, root,name="data.lmdb"):
        self.name = name if name.endswith(".lmdb") else f"{name}.lmdb"
        super(BSPMDataset, self).__init__(root)

    @property
    def processed_file_names(self):
        return [self.name]
    
    def len(self): # datasets are small
        env = lmdb.open(self.processed_paths[0], map_size=int(1e11))
        length = env.stat()['entries']
        env.close()
        return length

    def get(self, idx): # datasets are small
        env = lmdb.open(self.processed_paths[0], map_size=int(1e11))
        with env.begin() as txn:
            data_bytes = txn.get(str(idx).encode())
        env.close()
        return pickle.loads(data_bytes)

if __name__ == '__main__':
    dataset = BSPMDataset(root=f"datasets", name = "testset")

    print() 
    print(f'Dataset: {dataset}:')
    print('====================')
    print(f'Number of graphs: {len(dataset)}')
    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')

    data = dataset[0]  # Get the first graph object.

    print()
    print(data)
