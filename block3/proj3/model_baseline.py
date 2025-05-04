
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_data

from utils import reorder_adj_matrix_by_degree, data_to_A_matrix

def generate_baseline_datapoints(ref_dataloader, max_num_nodes, n_samples = 1000):
    
    
    def generate_graph(num_nodes, edge_density):
        """Generate a random graph with the given number of nodes and edge density"""
        # Generate a random adjacency matrix
        
        A = np.random.rand(num_nodes, num_nodes) < edge_density
        
        # Make sure the matrix is symmetric
        A = np.triu(A, 1) + np.triu(A, 1).T
        A = reorder_adj_matrix_by_degree(A)
        
        return A
    
    num_nodes_dist = np.array([d.num_nodes for d in ref_dataloader.dataset])
    
    # Get all graphs with the same number of nodes
    same_num_nodes_graphs = {
        num_nodes : [data_to_A_matrix(d, max_num_nodes) for d in ref_dataloader.dataset
                     if d.num_nodes == num_nodes
                    ]
        for num_nodes in np.unique(num_nodes_dist)
    }
    average_num_edges = {
        num_nodes : np.mean([np.sum(np.triu(d)) for d in same_num_nodes_graphs[num_nodes]])
        for num_nodes in same_num_nodes_graphs.keys()
    }
    edge_densities = {
        num_nodes : average_num_edges[num_nodes] / (num_nodes * (num_nodes - 1) / 2)
        for num_nodes in same_num_nodes_graphs.keys()
    }
    
    all_num_nodes = np.random.choice(num_nodes_dist, n_samples, replace = True)
    As = np.zeros((n_samples, max_num_nodes, max_num_nodes), dtype=int)
    for i, num_nodes in tqdm.tqdm(enumerate(all_num_nodes)):
        # Get the edge density for the current node count
        edge_density = edge_densities[num_nodes]
        
        # Generate a random graph with the given number of nodes and edge density
        As[i, :num_nodes, :num_nodes] = generate_graph(num_nodes, edge_density)

    return As, all_num_nodes
        

if __name__ == "__main__":
    
    from plotting import draw_graphs
    
    # %% Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, validation_loader, test_loader, data_info = get_data(device=device)
    
    max_num_nodes = data_info["max_num_nodes"]
    
    As_erdos, node_counts = generate_baseline_datapoints(train_loader, max_num_nodes, n_samples = 25)
    
    draw_graphs(As_erdos, node_counts = node_counts)    
    plt.tight_layout()
    plt.savefig(f"graph_erdos.png")
    plt.close()
    
    train_As = np.array([data_to_A_matrix(d, max_num_nodes) for d in train_loader.dataset])[:25]
    node_counts = np.array([d.num_nodes for d in train_loader.dataset])[:25]
    
    draw_graphs(train_As, node_counts = node_counts)
    plt.tight_layout()
    plt.savefig(f"graph_train.png")
    plt.close()
    
    
    