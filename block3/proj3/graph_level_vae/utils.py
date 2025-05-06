import numpy as np
import torch

import networkx as nx

def torch_batch_data_to_A_matrix(
        batch_data,
        max_num_nodes,
    ) -> torch.Tensor:
    
    
    num_batches = batch_data.batch.max() + 1
    As = torch.zeros((num_batches, max_num_nodes, max_num_nodes), dtype=torch.float32)
    
    graph_num_nodes = batch_data.batch.bincount()
    graph_num_nodes_cumsum = graph_num_nodes.cumsum(0)
    
    for edge_index in batch_data.edge_index.T:
        src, dst = edge_index
        A_index = batch_data.batch[src]
        
        batch_shift = graph_num_nodes_cumsum[A_index] - graph_num_nodes[0]
        src = src - batch_shift
        dst = dst - batch_shift
        
        As[A_index, src, dst] = 1
        As[A_index, dst, src] = 1
        
    for A_index in range(num_batches):
        As[A_index] = reorder_adj_matrix_by_degree(As[A_index], is_torch=True)
        
    return As
        

def data_to_A_matrix(data, max_num_nodes) -> np.ndarray:
    """Convert a PyG data object to an adjacency matrix"""
    # Get the edge indices and edge attributes
    edge_index = data.edge_index
    
    # Create an empty adjacency matrix
    A = np.zeros((max_num_nodes, max_num_nodes), dtype=np.float32)
    
    # Fill the adjacency matrix with edge attributes
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[:, i]
        if(src == dst):
            print(f"Self-loop detected at index {i}.")
        A[src.item(), dst.item()] = 1
        A[dst.item(), src.item()] = 1
        
    A = reorder_adj_matrix_by_degree(A)  
    
    return A


def reorder_adj_matrix_by_degree(adj_matrix : np.ndarray|torch.Tensor, is_torch : bool = False) -> np.ndarray|torch.Tensor:
    
    if is_torch:
        degrees = torch.sum(adj_matrix, dim=1)
        permutation_indices = torch.argsort(-degrees, stable=True)
        permuted_tensor = adj_matrix[permutation_indices]
        # Then, reorder the columns of the row-permuted tensor
        permuted_mat = permuted_tensor[:, permutation_indices]
        
    
    else:
        degrees = np.sum(adj_matrix, axis=1) # Sum each row
        permutation_indices = np.argsort(-degrees, kind='stable') # Use stable sort for ties
        permuted_mat = adj_matrix[np.ix_(permutation_indices, permutation_indices)]


    return permuted_mat


def remove_isoltated_nodes(As : np.ndarray|torch.Tensor, is_torch : bool = False) -> list[np.ndarray]|list[torch.Tensor]:
    """Remove isolated nodes from the adjacency matrix"""
    
    cleaned_As = []
    for i in range(As.shape[0]):
        A = As[i]
        if is_torch:
            # Both row and column since symmetric is not guaranteed
            degrees_row = torch.sum(A, dim=1)
            degrees_col = torch.sum(A, dim=0)
            degrees = degrees_row + degrees_col
            non_isolated_nodes = torch.where(degrees > 0)[0]
        else:
            degrees_row = np.sum(A, axis=1)
            degrees_col = np.sum(A, axis=0)
            degrees = degrees_row + degrees_col
            non_isolated_nodes = np.where(degrees > 0)[0]
        A_reduced = A[non_isolated_nodes][:, non_isolated_nodes]
        cleaned_As.append(A_reduced)
        
    return cleaned_As