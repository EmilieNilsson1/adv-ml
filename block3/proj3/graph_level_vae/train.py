
import tqdm
import torch
from block3.proj3.graph_level_vae.utils import torch_batch_data_to_A_matrix, remove_isoltated_nodes
import matplotlib.pyplot as plt

from block3.proj3.graph_level_vae.plotting import draw_graphs, plot_adjacency_matricies

def train(
        model,
        optimizer,
        data_loader,
        epochs,
        max_num_nodes,
        sort_by = "degree",
        device = 'cpu',
    ):
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
    sort_by: [str]
        Sorting method for the adjacency matrix. Can be "degree" or "cluster".
    device: [torch.device]
        The device to use for training.
    """
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm.trange(total_steps, desc="Training")

    losses = []
    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for batch_data in data_iter:
            # x = x[0].to(device)
            optimizer.zero_grad()
            As = torch_batch_data_to_A_matrix(batch_data, max_num_nodes, sort_by=sort_by).to(device)
            loss = model(batch_data.x, batch_data.edge_index, batch=batch_data.batch, A = As)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()
            
        if epoch % 10 == 0:
            As_fake = model.sample(25).cpu()
            plot_As = remove_isoltated_nodes(As_fake.detach().cpu().numpy())
            draw_graphs(plot_As)
            plt.savefig(f"graphs_epoch.png")
            plt.close()
            plot_adjacency_matricies(plot_As)
            plt.savefig(f"adjacency_epoch.png")
            plt.close()
            plot_As_real = remove_isoltated_nodes(As[:25].detach().cpu().numpy())
            draw_graphs(plot_As_real)
            plt.savefig(f"graphs_epoch_real.png")
            plt.close()
            plot_adjacency_matricies(plot_As_real)
            plt.savefig(f"adjacency_epoch_real.png")
            plt.close()
            
        plt.plot(losses)
        plt.grid()
        plt.savefig("loss.png")
        plt.close()

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


if __name__ == "__main__":
    
    from block3.proj3.graph_level_vae.dataset import get_data
    from block3.proj3.graph_level_vae.vae import GraphVAE, SimpleGNN, GaussianPrior, get_decoder_net, GaussianGraphEncoder, BernoulliImageDecoder
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    train_loader, validation_loader, test_loader, data_info = get_data(device=device)
    
    
    latent_dim = 4
    state_dim = 64 # was 16
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
        sort_by="degree",
        device=device
    )
    
    vae.eval()
    with torch.no_grad():
        samples = (vae.sample(64)).cpu() 
        
        draw_graphs(samples.detach().cpu().numpy(), [data_info["max_num_nodes"]]*len(samples))
        plt.savefig("generated_graphs.png")