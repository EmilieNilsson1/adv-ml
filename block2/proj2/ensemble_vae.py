# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from tqdm import tqdm
from copy import deepcopy
import os
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

class GaussianPrior(nn.Module):
    def __init__(self, M):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int]
           Dimension of the latent space.
        """
        super(GaussianPrior, self).__init__()
        self.M = M
        self.mean = nn.Parameter(torch.zeros(self.M), requires_grad=False)
        self.std = nn.Parameter(torch.ones(self.M), requires_grad=False)

    def forward(self):
        """
        Return the prior distribution.

        Returns:
        prior: [torch.distributions.Distribution]
        """
        return td.Independent(td.Normal(loc=self.mean, scale=self.std), 1)


class GaussianEncoder(nn.Module):
    def __init__(self, encoder_net):
        """
        Define a Gaussian encoder distribution based on a given encoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The encoder network that takes as a tensor of dim `(batch_size,
           feature_dim1, feature_dim2)` and output a tensor of dimension
           `(batch_size, 2M)`, where M is the dimension of the latent space.
        """
        super(GaussianEncoder, self).__init__()
        self.encoder_net = encoder_net

    def forward(self, x):
        """
        Given a batch of data, return a Gaussian distribution over the latent space.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        mean, std = torch.chunk(self.encoder_net(x), 2, dim=-1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(std)), 1)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_nets : list[nn.Module]):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_nets = decoder_nets
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z, decoder_idx : int):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_nets[decoder_idx](z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

class FlattenOutDecoder(nn.Module): 
    def __init__(self, decoder, output_scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder 
        self.output_scale = output_scale
        
    def forward(self, z, decoder_idx : int = 0): 
        decoding_dist = self.decoder(z, decoder_idx)
        mean = torch.flatten(decoding_dist.base_dist.loc, start_dim=1, end_dim=-1)
        std = torch.flatten(decoding_dist.base_dist.scale, start_dim=1, end_dim=-1)

        if self.output_scale: return mean, std 
        else: return mean 
    

class VAE(nn.Module):
    """
    Define a Variational Autoencoder (VAE) model.
    """

    def __init__(self, prior, decoder, encoder):
        """
        Parameters:
        prior: [torch.nn.Module]
           The prior distribution over the latent space.
        decoder: [torch.nn.Module]
              The decoder distribution over the data space.
        encoder: [torch.nn.Module]
                The encoder distribution over the latent space.
        """

        super(VAE, self).__init__()
        self.prior = prior
        self.decoder = decoder
        self.encoder = encoder

    def elbo(self, x, decoder_idx : int):
        """
        Compute the ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2, ...)`
           n_samples: [int]
           Number of samples to use for the Monte Carlo estimate of the ELBO.
        """
        q = self.encoder(x)
        z = q.rsample()

        elbo = torch.mean(
            self.decoder(z, decoder_idx).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
        return elbo

    def sample(self, n_samples=1, decoder_idx : int = 0):
        """
        Sample from the model.

        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z, decoder_idx).sample()

    def forward(self, x, decoder_idx : int):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor]
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x, decoder_idx)


def train(model, optimizer, data_loader, epochs, device):
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
    device: [torch.device]
        The device to use for training.
    """

    num_steps = len(data_loader) * epochs
    epoch = 0

    def noise(x, std=0.05):
        eps = std * torch.randn_like(x)
        return torch.clamp(x + eps, min=0.0, max=1.0)

    with tqdm(range(num_steps)) as pbar:
        for step in pbar:
            try:
                x = next(iter(data_loader))[0]
                x = noise(x.to(device))
                model = model
                optimizer.zero_grad()
                # from IPython import embed; embed()
                decoder_idx = np.random.randint(0, len(model.decoder.decoder_nets))
                loss = model(x, decoder_idx)
                loss.backward()
                optimizer.step()

                # Report
                if step % 5 == 0:
                    loss = loss.detach().cpu()
                    pbar.set_description(
                        f"total epochs ={epoch}, step={step}, loss={loss:.1f}"
                    )

                if (step + 1) % len(data_loader) == 0:
                    epoch += 1
            except KeyboardInterrupt:
                print(
                    f"Stopping training at total epoch {epoch} and current loss: {loss:.1f}"
                )
                break

def get_geo_latent(start_p, end_p, decoder, device, N=10, scale=False):
    decoder = FlattenOutDecoder(decoder, output_scale=scale) 
    
    def metric(z): 
        jac = torch.autograd.functional.jacobian(decoder, z).squeeze() 
        if not scale : 
            riemannian = jac.T @ jac
        else : 
            raise NotImplementedError 
        return riemannian
    
    # start_p.requires_grad = True 
    # end_p.requires_grad = True 
    
    return geo_min_energy_latent(start_p=start_p, end_p=end_p, metric=metric, N=N, device=device)
    
def geo_min_energy_latent(start_p, end_p, metric, N, device, max_epochs=100):
    """
    Calculate the geodesic by minimizing the energy using the Jacobian of the decoder.
    """
    
    d = start_p.shape[1]
    points = torch.randn((N, d), requires_grad=True, device=device)
    optimizer = torch.optim.LBFGS([points], lr=1e-2)

    def closure():
        optimizer.zero_grad()
        trajectory = torch.concat((start_p, points, end_p))
        
        loss = 0
        for i in range(len(trajectory)- 1):
            dif = (trajectory[i+1:i+2] - trajectory[i:i+1])
            G = metric(trajectory[i:i+1]) 
            loss += dif @ G @ dif.T
        
        loss.backward()
        return loss  # LBFGS needs the loss value to be returned
    
    loss = torch.inf 
    for i in range(max_epochs): 
        loss_new = optimizer.step(closure)  # Pass the closure to LBFGS
        # print(loss_new)
        if loss - loss_new < 0.1: break 
        loss = loss_new 
        

    return torch.concat((start_p, points, end_p)), loss

def geo_min_energy_manifold(start_p, end_p, decoder, N, device, max_epochs=100, num_mc_samples=10, data_dim=28*28):
    """
    Calculate the geodesic by minimizing the energy using the Jacobian of the decoder.
    """
    d = start_p.shape[1]
    
    # Initialize the N points of dimension d as a linear combination of start_p and end_p
    points = torch.zeros((N, d), device=device)
    for i in range(N):
        points[i] = start_p + (end_p - start_p) * (i / (N - 1))
    points.requires_grad = True
    
    optimizer = torch.optim.LBFGS([points], lr=1e-1)

    flattenDecoder = FlattenOutDecoder(decoder, output_scale=False)

    def closure():
        optimizer.zero_grad()
        trajectory = torch.concat((start_p, points, end_p))
        
        if len(flattenDecoder.decoder.decoder_nets) > 1:
            loss = 0
            for _ in range(num_mc_samples):
                decoded_trajectory = torch.zeros((trajectory.shape[0], data_dim), device=device)
                for i, point in enumerate(trajectory):
                    decoder_idx = np.random.randint(0, len(flattenDecoder.decoder.decoder_nets))
                    decoded_trajectory[i] = flattenDecoder(point[None,...], decoder_idx)
                    
                diff = torch.diff(decoded_trajectory, dim=0)
                loss += torch.sum(torch.norm(diff, p=2, dim=1)**2)
            loss /= num_mc_samples
        else:
            decoded_trajectory = flattenDecoder(trajectory)
            diff = torch.diff(decoded_trajectory, dim=0)
        
            loss = torch.sum(torch.norm(diff, p=2, dim=1)**2)
        
        # Calcualte: loss = ||diff||^2
        # for i in range(len(diff)- 1):
        #     loss += torch.norm(diff[i:i+1], p=2)**2
        # loss /= len(diff)
        
        loss.backward()
        return loss  # LBFGS needs the loss value to be returned
    
    loss = torch.inf 
    for i in range(max_epochs): 
        loss_new = optimizer.step(closure)  # Pass the closure to LBFGS
        if i % 10 == 0:
            print(f"Epoch {i}, Loss: {loss_new.item()}")
        if loss - loss_new < 0.01: break 
        loss = loss_new 
        
    length = 0
    for i in range(len(points)- 1):
        dif = (points[i+1:i+2] - points[i:i+1])
        G = torch.eye(d, device=device)
        length += torch.sqrt(dif @ G @ dif.T)

    return torch.concat((start_p, points, end_p)), length

def get_encoder_nn():
    encoder_net = nn.Sequential(
        nn.Conv2d(1, 16, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.Conv2d(16, 32, 3, stride=2, padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.Conv2d(32, 32, 3, stride=2, padding=1),
        nn.Flatten(),
        nn.Linear(512, 2 * M),
    )
    return encoder_net

def get_decoder_nn():
    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.Unflatten(-1, (32, 4, 4)),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
        nn.Softmax(),
        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
        nn.Softmax(),
        nn.BatchNorm2d(16),
        nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
    )
    return decoder_net

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image

    # Parse arguments
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="geodesics",
        choices=["train", "sample", "eval", "geodesics"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiment",
        help="folder to save and load experiment results in (default: %(default)s)",
    )
    parser.add_argument(
        "--samples",
        type=str,
        default="samples.png",
        help="file to save samples in (default: %(default)s)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="torch device (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="batch size for training (default: %(default)s)",
    )
    parser.add_argument(
        "--epochs-per-decoder",
        type=int,
        default=100,
        metavar="N",
        help="number of training epochs per each decoder (default: %(default)s)",
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=2,
        metavar="N",
        help="dimension of latent variable (default: %(default)s)",
    )
    parser.add_argument(
        "--num-decoders",
        type=int,
        default=3,
        metavar="N",
        help="number of decoders in the ensemble (default: %(default)s)",
    )
    parser.add_argument(
        "--num-reruns",
        type=int,
        default=10,
        metavar="N",
        help="number of reruns (default: %(default)s)",
    )
    parser.add_argument(
        "--num-curves",
        type=int,
        default=10,
        metavar="N",
        help="number of geodesics to plot (default: %(default)s)",
    )
    parser.add_argument(
        "--num-t",  # number of points along the curve
        type=int,
        default=100,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
    )
    parser.add_argument(
        "--optimize_in",
        type=str,
        default="manifold",
        choices=["latent", "manifold"],
        help="optimize in latent space or manifold (default: %(default)s)",
    )
    parser.add_argument(
        "--saved_latent_points",
        type=str,
        default=None,#"experiment/results25latentspace.pth",
        help="path to saved latent points (default: %(default)s)",
    )
    parser.add_argument(
        "--inference_num_decoders",
        type=int,
        default=1,
        metavar="N",
        help="number of decoders to use for inference/evaluation/geodesics (default: %(default)s)",
    )
    parser.add_argument(
        "--geod_mc_samples",
        type=int,
        default=10,
        metavar="N",
        help="number of Monte Carlo samples for geodesics (default: %(default)s)",
    )

    args = parser.parse_args()
    print("# Options")
    for key, value in sorted(vars(args).items()):
        print(key, "=", value)

    device = args.device

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1).to(torch.float32) / 255
        new_targets = targets[idx][:num_data]

        return torch.utils.data.TensorDataset(new_data, new_targets)

    num_train_data = 2048
    num_classes = 3
    train_tensors = datasets.MNIST(
        "data/",
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    test_tensors = datasets.MNIST(
        "data/",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()]),
    )
    train_data = subsample(
        train_tensors.data, train_tensors.targets, num_train_data, num_classes
    )
    test_data = subsample(
        test_tensors.data, test_tensors.targets, num_train_data, num_classes
    )

    mnist_train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True
    )
    mnist_test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.batch_size, shuffle=False
    )

    # Define prior distribution
    M = args.latent_dim

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        
        for rerun in tqdm(range(args.num_reruns)):
            encoder = get_encoder_nn()
            for num_decoders in range(args.num_decoders):
                decoders = []
                for _ in range(num_decoders + 1):
                    decoders.append(get_decoder_nn())
                    
                model = VAE(
                    GaussianPrior(M),
                    GaussianDecoder(decoders),
                    GaussianEncoder(encoder),
                ).to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                train(
                    model,
                    optimizer,
                    mnist_train_loader,
                    args.epochs_per_decoder,
                    args.device,
                )
        
                torch.save(
                    model.state_dict(),
                    f"{experiments_folder}/rr{rerun}_ndecoder{num_decoders}_model.pt",
                )

    elif args.mode == "sample":
        
        encoder = get_encoder_nn()
        decoders = []
        for _ in range(args.num_decoders + 1):
            decoders.append(get_decoder_nn())
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(decoders),
            GaussianEncoder(encoder),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + f"/rr0_ndecoder{args.inference_num_decoders-1}_model.pt"))
        model.eval()

        with torch.no_grad():
            samples = (model.sample(64)).cpu()
            save_image(samples.view(64, 1, 28, 28), args.samples)

            data = next(iter(mnist_test_loader))[0].to(device)
            recon = model.decoder(model.encoder(data).mean).mean
            save_image(
                torch.cat([data.cpu(), recon.cpu()], dim=0), "reconstruction_means.png"
            )

    elif args.mode == "eval":
        # Load trained model
        encoder = get_encoder_nn()
        decoders = []
        for _ in range(args.num_decoders + 1):
            decoders.append(get_decoder_nn())
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(decoders),
            GaussianEncoder(encoder),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + f"/rr0_ndecoder{args.inference_num_decoders-1}_model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)

    elif args.mode == "geodesics":

        encoder = get_encoder_nn()
        decoders = []
        for _ in range(args.num_decoders + 1):
            decoders.append(get_decoder_nn())
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(decoders),
            GaussianEncoder(encoder),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + f"/rr1_ndecoder{args.inference_num_decoders-1}_model.pt"))
        model.eval()
        
        # Go through K pairs randomly sampled? 
        K = 10
        N = 50
        
        if args.saved_latent_points is not None:
            data = torch.load(args.saved_latent_points)
            latent_points = data['latent_points']
            paths_saved = data['paths']
            lengths = data['lengths']
            curr_pointer = 0
            labels = torch.zeros((len(mnist_train_loader.dataset)), device=device)
            for batch in mnist_train_loader:
                zs = model.encoder(batch[0].to(device)).base_dist.mean
                labels[curr_pointer:curr_pointer+len(zs)] = batch[1]
                curr_pointer += len(zs)
            
            start_p = paths_saved[:, 0, :]
            end_p = paths_saved[:, -1, :]
            
        else:
            curr_pointer = 0
            latent_points = torch.zeros((len(mnist_train_loader.dataset), 2), device=device)
            labels = torch.zeros((len(mnist_train_loader.dataset)), device=device)
            for batch in mnist_train_loader:
                imgs = batch[0].to(device)
                zs = model.encoder(imgs).base_dist.mean
                labels[curr_pointer:curr_pointer+len(zs)] = batch[1]
                latent_points[curr_pointer:curr_pointer+len(zs)] = zs.detach()
                curr_pointer += len(zs) 
                
            start_p = torch.randn((K, 2), device=device)
            end_p = torch.randn((K, 2), device=device)
            
            for i in range(K): 
                choices = torch.multinomial(torch.ones(len(latent_points))/len(latent_points), 2).to(device)
                endpoints = latent_points[choices]
                start_p[i] = endpoints[0][None, :]
                end_p[i] = endpoints[1][None, :]
            
        
        paths = torch.zeros((K, N+2, 2), device=device)
        lengths = torch.zeros((K), device=device)
        for i in tqdm(range(K)): 
            
            if args.optimize_in == "latent": 
                trajectory, length = geo_min_energy_latent(start_p[i][None], end_p[i][None], model.decoder, N=N, device=device)
            elif args.optimize_in == "manifold":
                trajectory, length = geo_min_energy_manifold(
                    start_p[i][None], 
                    end_p[i][None], 
                    model.decoder, 
                    N=N, 
                    device=device,
                    max_epochs=100,
                    num_mc_samples=args.geod_mc_samples,
                    data_dim=28*28
                )
            else:
                raise NotImplementedError("Unknown optimization space")
            
            paths[i] = trajectory.detach()
            lengths[i] = length.detach()
        
        latent_points = latent_points.to('cpu')
        paths = paths.to('cpu')
        lengths = lengths.to('cpu')
        
        cmap = plt.get_cmap("tab10")
        colors = [cmap(int(label)) for label in labels]
        unique_labels = torch.unique(labels)
        
        # Plot latent representation of all points 
        plt.scatter(latent_points[:, 0], latent_points[:,1], c=colors, s=1, marker='o')
        legend = [Line2D([0], [0], color=cmap(int(i)), marker='o', linestyle='None', markersize=10, label=f'{int(i)}') for i in unique_labels]
        plt.legend(handles=legend)
        # Plot trajectories 
        plt.plot(paths[:, :, 0].T, paths[:, :, 1].T, 'g-', alpha=0.5)
        plt.plot(paths[:, 0, 0], paths[:, 0, 1], 'g.')
        plt.plot(paths[:, -1, 0], paths[:, -1, 1], 'g.')
        # Show the distance for each line 
        for line in range(len(paths)): 
            pos = paths[line, N//2]
            plt.text(pos[0], pos[1], f'{lengths[line].item():.1f}', fontsize=8, ha='center', va='center', color='black')
        plt.savefig('25path_latent_space_new.png')
        #TODO grid? 
        plt.show()
