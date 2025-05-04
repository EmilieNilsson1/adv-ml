import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as patches

def draw_graphs(As_ordered : np.ndarray, node_counts : np.ndarray) -> None:
    """
    Draw a graph using NetworkX and Matplotlib
    
    Args:
        As_ordered (np.ndarray): Adjacency matricies of graphs. Each adjacency matrix is assumed to be
        ordered by the node degree.
        node_counts (np.ndarray): Number of nodes in each graph
    """
    
    
    w = np.ceil(np.sqrt(len(As_ordered))).astype(int)  # Number of rows
    h = np.ceil(len(As_ordered) / w).astype(int)  # Number of columns
    plt.figure(figsize=(w * 5, h * 5))  # Set the figure size
    for i, A in enumerate(As_ordered):
        
        num_nodes = node_counts[i]
        ax = plt.subplot(h, w, i + 1)
        
        A = A[:num_nodes,:num_nodes].astype(int)  # Ensure the adjacency matrix is binary
        G = nx.from_numpy_array(A)  # Create a graph from the adjacency matrix
    
        nx.draw(G, ax=ax, node_size=int(200/w))
        
        rect = patches.Rectangle((0, 0), 1, 1,             # Coordinates and size
                                 transform=ax.transAxes,   # Use Axes coordinates
                                 linewidth=1,
                                 edgecolor='black',
                                 facecolor='none',         # No fill
                                 clip_on=False,            # Draw lines even if slightly outside bounds
                                 zorder=10)                # Draw on top

        # Add the rectangle to the Axes
        ax.add_patch(rect)
         