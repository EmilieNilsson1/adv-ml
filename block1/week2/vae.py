# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.2 (2024-02-06)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py

import torch
import torch.nn as nn
import torch.distributions as td
import torch.utils.data
from torch.nn import functional as F
from tqdm import tqdm



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


class BernoulliDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(BernoulliDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        # return td.Independent(td.Bernoulli(logits=logits), 1)
        return td.Independent(td.Bernoulli(logits=logits), 2)


class GaussianDecoder(nn.Module):
    def __init__(self, decoder_net):
        """
        Define a Gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        self.log_std = nn.Parameter(torch.full((28, 28), 0.0), requires_grad=True)
    
    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        mean = self.decoder_net(z)
        # return td.Independent(td.Normal(loc=mean, scale=torch.exp(self.log_std)), 1)
        return td.Independent(td.Normal(loc=mean, scale=torch.exp(self.log_std)), 2)



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

    def elbo(self, x):
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

        # NOTE: Calculating week 2, slide 37, equation line 2 directly
        elbo = torch.mean(self.decoder(z).log_prob(x) + self.prior.log_prob(z) - q.log_prob(z)) 
        return elbo

    def sample(self, n_samples=1, mean=False):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior.sample((n_samples,))
        
        if mean:
            return self.decoder(z).mean
        else:
            return self.decoder(z).sample()
    
    def forward(self, x):
        """
        Compute the negative ELBO for the given batch of data.

        Parameters:
        x: [torch.Tensor] 
           A tensor of dimension `(batch_size, feature_dim1, feature_dim2)`
        """
        return -self.elbo(x)


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
    model.train()

    total_steps = len(data_loader)*epochs
    progress_bar = tqdm(range(total_steps), desc="Training")

    for epoch in range(epochs):
        data_iter = iter(data_loader)
        for x in data_iter:
            x = x[0].to(device)
            optimizer.zero_grad()
            loss = model(x)
            loss.backward()
            optimizer.step()

            # Update progress bar
            progress_bar.set_postfix(loss=f"⠀{loss.item():12.4f}", epoch=f"{epoch+1}/{epochs}")
            progress_bar.update()


if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    from sklearn.decomposition import PCA
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    
    from flow_vae import Flow, GaussianBase, MaskedCouplingLayer
    from draw_contours import draw_contours_points

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model.pt', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=32, metavar='N', help='dimension of latent variable (default: %(default)s)')

    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    thresshold = 0.5
    mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True, persistent_workers=True, num_workers=4)
    mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                    batch_size=args.batch_size, shuffle=True)
    # mnist_train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         'data/',
    #         train=True,
    #         download=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(), 
    #             transforms.Lambda(lambda x: x.float().squeeze())
    #         ])
    #     ),
    #     batch_size=args.batch_size, 
    #     shuffle=True,
    #     persistent_workers=True,
    #     num_workers=4
    # )
    # mnist_test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST(
    #         'data/', 
    #         train=False, 
    #         download=True,
    #         transform=transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Lambda(lambda x: x.float().squeeze())
    #         ])
    #     ),
    #     batch_size=args.batch_size, 
    #     shuffle=True
    # )


    # Define prior distribution
    M = args.latent_dim
    # prior = GaussianPrior(M)
    # prior = MixtureOfGaussianPrior(M, 10)
    # Define prior distribution
    base = GaussianBase(M)

    # Define transformations
    transformations = []
    
    num_transformations = 8
    num_hidden = 256
    mask_type = "random"
    
    for i in range(num_transformations):
        if mask_type == "default":
            mask = torch.zeros((M,))
            mask[:M//2] = (i+1) % 2
            mask[M//2:] = i % 2
        elif mask_type == "random":
            mask = torch.randint(0, 2, (M,))
        elif mask_type == "checkerboard":
            mask = torch.tensor([1 if (i + j+k) % 2 == 0 else 0 for j in range(28) for k in range(28)])

        scale_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M), nn.Tanh())
        translation_net = nn.Sequential(nn.Linear(M, num_hidden), nn.ReLU(), nn.Linear(num_hidden, M))
        transformations.append(MaskedCouplingLayer(scale_net, translation_net, mask))

    prior = Flow(base, transformations).to(args.device)
    
    
    

    # Define encoder and decoder networks
    encoder_net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, M*2),
    )

    decoder_net = nn.Sequential(
        nn.Linear(M, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 784),
        nn.Unflatten(-1, (28, 28))
    )

    # Define VAE model
    decoder = BernoulliDecoder(decoder_net)
    # decoder = GaussianDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)
    model = VAE(prior, decoder, encoder).to(device)

    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), args.model)

    elif args.mode == 'sample':
        # model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # # Generate samples
        # model.eval()
        # with torch.no_grad():
        #     samples = (model.sample(64, mean=True)).cpu()
        #     save_image(samples.view(64, 1, 28, 28), args.samples)
            
        #     print(f"ELBO (test): {torch.mean(torch.tensor([model.elbo(x.to(device)).item() for x, _ in tqdm(mnist_test_loader, desc="Estimating ELBO")])).item()}")
            
        #     n_samples = 1
        #     z: list[np.ndarray] = []
        #     c: list[np.ndarray] = []
            
        #     for x, y in mnist_test_loader:
        #         x = x.to(device)

        #         z.append(model.encoder(x).sample((n_samples,)).cpu().numpy().reshape((n_samples*len(x), -1)))
        #         c.append(np.array([0] * len(x)*n_samples))
                
        #         z.append(model.prior.sample((len(x)*n_samples,)).cpu().numpy())
        #         c.append(np.array([1] * len(x)*n_samples))

        #     z = np.concatenate(z)
        #     zz = PCA(n_components=2).fit_transform(z)
        #     cc = np.concatenate(c)
            
        #     for c in np.unique(cc):
        #         plt.scatter(zz[cc == c, 0], zz[cc == c, 1], s=0.5, label=f"{c}")
        #     plt.legend()
        #     plt.show()


        model.load_state_dict(torch.load(args.model, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(4, mean=True)).cpu()
            save_image(samples.view(4, 1, 28, 28), args.samples, nrow=2)
            
            print(f"ELBO (test): {torch.mean(torch.tensor([model.elbo(x.to(device)).item() for x, _ in tqdm(mnist_test_loader, desc="Estimating ELBO")])).item()}")
            
            
            
            
            
            
            
            
            
            """
            Encoder returns a Gaussian with diagonal covariance
            The inverse flow to get to the variational posterior is 
                a bunch of affine transformations on each dimension independently
            The variational posterior is therefore a Gaussian with diagonal covariance
            The mean can be found by running the encoder mean through the inverse
            The variance is m*torch.ones_like(z) + (1-m)*(-s).exp() for an individual inverse transform
            
            
            TODO: Consolidate flow code
            """
            
            means = []
            std = []
            labels = []

            for batch in tqdm(mnist_train_loader):
                data = batch[0].to(device)
                dist = model.encoder(data)
                z, _, scale = model.prior.inverse(dist.mean, return_scale=True)
                
                
                means.append(z.cpu())
                std.append((dist.stddev * scale).cpu())
                labels.append(batch[1])



            means = torch.cat(means, dim=0)
            std = torch.cat(std, dim=0)
            labels = torch.cat(labels, dim=0)


            draw_contours_points(means, std, GaussianBase(model.prior.base.D)(), f"contours.png", title="VAE Flow Bernoulli", multivariate=False, labels=labels)
            
            
            # n_samples = 1
            # z: list[np.ndarray] = []
            # c: list[np.ndarray] = []
            
            # for x, y in mnist_test_loader:
            #     x = x.to(device)

            #     z.append(model.encoder(x).sample((n_samples,)).cpu().numpy().reshape((n_samples*len(x), -1)))
            #     c.append(np.array([0] * len(x)*n_samples))
                
            #     z.append(model.prior.sample((len(x)*n_samples,)).cpu().numpy())
            #     c.append(np.array([1] * len(x)*n_samples))

            # z = np.concatenate(z)
            # zz = PCA(n_components=2).fit_transform(z)
            # cc = np.concatenate(c)
            
            # for c in np.unique(cc):
            #     plt.scatter(zz[cc == c, 0], zz[cc == c, 1], s=0.5, label=f"{c}")
            # plt.legend()
            # plt.show()
            
