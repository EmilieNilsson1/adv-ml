
import torch
from torch.utils.data import random_split
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

def get_data(splits = (100/188, 44/188, 44/188), batch_sizes = (100, 44, 44), split_rng = 420, device = 'cpu'):
    
    dataset = TUDataset(root='./data/', name='MUTAG').to(device)

    train_size = int(len(dataset) * splits[0])
    val_size = int(len(dataset) * splits[1])
    test_size = len(dataset) - train_size - val_size
    
    # Split into training and validation
    rng = torch.Generator().manual_seed(split_rng)
    train_dataset, validation_dataset, test_dataset = random_split(dataset, (train_size, val_size, test_size), generator=rng)

    max_num_nodes = max([d.num_nodes for d in dataset])
    
    data_info = {
        'num_node_features': dataset.num_node_features,
        'num_classes': dataset.num_classes,
        'max_num_nodes': max_num_nodes
    }

    # Create dataloader for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_sizes[0])
    validation_loader = DataLoader(validation_dataset, batch_size=batch_sizes[1])
    test_loader = DataLoader(test_dataset, batch_size=batch_sizes[2])
    
    print("Successfully loaded the datasets of length:")
    print(f">> Train: {len(train_dataset)}, Validation: {len(validation_dataset)}, Test: {len(test_dataset)}")
    
    return train_loader, validation_loader, test_loader, data_info