
import tqdm
import torch
from utils import torch_batch_data_to_A_matrix

def train(model, optimizer, data_loader, epochs, max_num_nodes, device):
    """
    Train a VAE model.

    Parameters:
    model: [VAE]
       The VAE model to train.
    optimizer: [torch.optim.Optimizer]
         The optimizer to use for training.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for training.
    epochs: [int]
        Number of epochs to train for.
    max_num_nodes: [int]
        The maximum number of nodes in the graphs.
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm.trange(total_steps, desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for batch_data in data_iter:
            # x = x[0].to(device)
            optimizer.zero_grad()
            As = torch_batch_data_to_A_matrix(batch_data, max_num_nodes).to(device)
            loss = model(batch_data.x, batch_data.edge_index, batch=batch_data.batch, A = As)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()

def evaluate(model, data_loader, device):
    """
    Evaluate the ELBO of a VAE model on a given data loader.

    Parameters:
    model: [VAE]
       The VAE model to evaluate.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for evaluation.
    device: [torch.device]
        The device to use for evaluation.
    """
    model.eval()
    elbo = 0
    with torch.no_grad():
        for x in data_loader:
            x = x[0].to(device)
            elbo += model.elbo(x) * len(x)
            
    return elbo/len(data_loader.dataset)

# def sample_posterior(model, data_loader, M, name ,device):
#     """
#     Sample from the posterior of a VAE model on a given data loader.

#     Parameters:
#     model: [VAE]
#         The VAE model to sample from.
#     data_loader: [torch.utils.data.DataLoader]
#             The data loader to use for sampling.
#     device: [torch.device]
#         The device to use for sampling.
#     """
#     model.eval()
#     encodings = torch.zeros((len(data_loader.dataset), M))
#     labels = torch.zeros(len(data_loader.dataset))
#     idx = int(0)
#     with torch.no_grad():
#         for i, x in enumerate(data_loader):
#             # Save labels
#             labels[int(idx):int(idx+len(x[0]))] = x[1]

#             x = x[0].to(device)
#             encodings[int(idx):int(idx+len(x))] = model.encoder(x).rsample()
#             idx += len(x)

#     # Perform PCA on the encodings
#     pca = PCA(n_components=2)
#     pca.fit(encodings)
#     encoding_pca = pca.transform(encodings)

#     # Plot the encodings
#     plt.figure()
#     plt.scatter(encoding_pca[:, 0], encoding_pca[:, 1], c=labels)
#     plt.colorbar()
#     plt.savefig(f'pca_{name}.png')


#     unique_labels = torch.unique(labels)
#     colors = plt.cm.jet(torch.linspace(0, 1, len(unique_labels)))  # Generate distinct colors

#     # Create the scatter plot
#     plt.figure(figsize=(8, 6))
#     for label, color in zip(unique_labels, colors):
#         mask = labels == label  # Boolean mask for the current label
#         plt.scatter(encoding_pca[mask, 0], encoding_pca[mask, 1], color=color, label=f'{int(label)}', alpha=0.6, edgecolors='k')
#     plt.legend()
#     plt.savefig(f'pca_{name}2.png')

if __name__ == "__main__":
    
    from dataset import get_data
    from model import GraphVAE, SimpleGNN, GaussianPrior, get_decoder_net, GaussianGraphEncoder, BernoulliImageDecoder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, validation_loader, test_loader, data_info = get_data()#splits=(0.9, 0.05, 0.05), device=device)
    
    
    latent_dim = 4
    state_dim = 32 # was 16
    num_message_passing_rounds = 5 # was 4
    
    encoder_net = SimpleGNN(
        data_info["num_node_features"],
        state_dim,
        num_message_passing_rounds,
        output_dim=latent_dim*2,
    ).to(device)
    
    decoder_net = get_decoder_net(
        latent_dim,
        max_num_nodes=data_info["max_num_nodes"]
    )
    
    decoder = BernoulliImageDecoder(
        decoder_net
    )
    
    encoder = GaussianGraphEncoder(
        encoder_net
    )
    
    prior = GaussianPrior(
        M = latent_dim,
    )
    
    vae = GraphVAE(
        prior=prior,
        decoder=decoder,
        encoder=encoder,
        max_num_nodes=data_info["max_num_nodes"],
    ).to(device)
    
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    
    train(
        vae, 
        optimizer, 
        train_loader, 
        epochs=500, 
        max_num_nodes=data_info["max_num_nodes"],
        device=device
    )
    
    vae.eval()
    with torch.no_grad():
        samples = (vae.sample(64)).cpu() 
        
        import matplotlib.pyplot as plt
        from plotting import draw_graphs
        draw_graphs(samples.detach().cpu().numpy(), [data_info["max_num_nodes"]]*len(samples))
        plt.savefig("generated_graphs.png")