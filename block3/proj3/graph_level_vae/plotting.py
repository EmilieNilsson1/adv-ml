import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import matplotlib.patches as patches

def draw_graphs(As : np.ndarray) -> None:
    """
    Draw a graph using NetworkX and Matplotlib
    
    Args:
        As (list[np.ndarray]): List of adjacency matrices of graphs. Each adjacency matrix is assumed to be
        ordered by the node degree.
        node_counts (np.ndarray): Number of nodes in each graph
    """

    w = np.ceil(np.sqrt(len(As))).astype(int)  # Number of rows
    h = np.ceil(len(As) / w).astype(int)  # Number of columns
    plt.figure(figsize=(w * 5, h * 5))  # Set the figure size
    for i, A in enumerate(As):
        
        ax = plt.subplot(h, w, i + 1)
        
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
         
def plot_adjacency_matricies(As_ordered : np.ndarray) -> None:
    """
    Plot the adjacency matrices of the graphs.
    
    Args:
        As_ordered (list[np.ndarray]): List of adjacency matrices of graphs. Each adjacency matrix is assumed to be
    """
    
    w = np.ceil(np.sqrt(len(As_ordered))).astype(int)  # Number of rows
    h = np.ceil(len(As_ordered) / w).astype(int)  # Number of columns
    plt.figure(figsize=(w * 5, h * 5))  # Set the figure size
    for i, A in enumerate(As_ordered):
        ax = plt.subplot(h, w, i + 1)
        ax.matshow(A, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])