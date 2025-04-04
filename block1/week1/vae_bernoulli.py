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
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

import json

from draw_contours import draw_contours, draw_contours_points, draw_contours_points_2d


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
    
class VampPrior(nn.Module):
    def __init__(self, D, K, encoder):
        """
        Define a Gaussian prior distribution with zero mean and unit variance.

                Parameters:
        M: [int] 
           Dimension of the latent space.
        """
        super(VampPrior, self).__init__()
        self.D = D
        self.K = K
        self.x = nn.Parameter(torch.randn(self.K, self.D), requires_grad=True)
        self.encoder = encoder
    
    def forward(self):
        q = self.encoder(self.x)
        mix = td.Categorical(torch.ones(self.K) / self.K)
        return td.MixtureSameFamily(mix, q)


class MoGPrior(nn.Module):
    """ 
    Mixture of Gaussians prior
    """

    def __init__(self, M, k=10):
        super(MoGPrior, self).__init__()
        self.M = M
        self.k = k
        self.mean = nn.Parameter(torch.randn(self.k,self.M), requires_grad=True)
        self.std = nn.Parameter(torch.randn(self.k,self.M), requires_grad=True)
        self.weights = nn.Parameter(torch.ones(self.k,), requires_grad=True)
    
    def forward(self):
        mix = td.Categorical(F.softmax(self.weights))
        comp = td.Independent(td.Normal(self.mean, torch.sqrt(self.std**2)), 1)
        return td.MixtureSameFamily(mix, comp)
    
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
        self.std = nn.Parameter(torch.ones(28, 28)*0.5, requires_grad=True)

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        logits = self.decoder_net(z)
        return td.Independent(td.Normal(loc=logits, scale = torch.exp(self.std)), 2)

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
        # elbo = torch.mean(self.decoder(z).log_prob(x) - td.kl_divergence(q, self.prior()), dim=0) gp
        elbo = torch.mean(self.decoder(z).log_prob(x) + self.prior().log_prob(z) - q.log_prob(z)) # mog 
        #elbo = torch.mean(self.decoder(z).log_prob(x) + self.prior(z) - q.log_prob(z))
        return elbo

    def sample(self, n_samples=1):
        """
        Sample from the model.
        
        Parameters:
        n_samples: [int]
           Number of samples to generate.
        """
        z = self.prior().sample(torch.Size([n_samples]))
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

def sample_posterior(model, data_loader, M, name ,device):
    """
    Sample from the posterior of a VAE model on a given data loader.

    Parameters:
    model: [VAE]
        The VAE model to sample from.
    data_loader: [torch.utils.data.DataLoader]
            The data loader to use for sampling.
    device: [torch.device]
        The device to use for sampling.
    """
    model.eval()
    encodings = torch.zeros((len(data_loader.dataset), M))
    labels = torch.zeros(len(data_loader.dataset))
    idx = int(0)
    with torch.no_grad():
        for i, x in enumerate(data_loader):
            # Save labels
            labels[int(idx):int(idx+len(x[0]))] = x[1]

            x = x[0].to(device)
            encodings[int(idx):int(idx+len(x))] = model.encoder(x).rsample()
            idx += len(x)

    # Perform PCA on the encodings
    pca = PCA(n_components=2)
    pca.fit(encodings)
    encoding_pca = pca.transform(encodings)

    # Plot the encodings
    plt.figure()
    plt.scatter(encoding_pca[:, 0], encoding_pca[:, 1], c=labels)
    plt.colorbar()
    plt.savefig(f'pca_{name}.png')


    unique_labels = torch.unique(labels)
    colors = plt.cm.jet(torch.linspace(0, 1, len(unique_labels)))  # Generate distinct colors

    # Create the scatter plot
    plt.figure(figsize=(8, 6))
    for label, color in zip(unique_labels, colors):
        mask = labels == label  # Boolean mask for the current label
        plt.scatter(encoding_pca[mask, 0], encoding_pca[mask, 1], color=color, label=f'{int(label)}', alpha=0.6, edgecolors='k')
    plt.legend()
    plt.savefig(f'pca_{name}2.png')

if __name__ == "__main__":
    from torchvision import datasets, transforms
    from torchvision.utils import save_image, make_grid
    import glob

    # Parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'sample', 'eval', 'pca', 'train_multiple'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model', type=str, default='model', help='file to save model to or load model from (default: %(default)s)')
    parser.add_argument('--samples', type=str, default='samples.png', help='file to save samples in (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda', 'mps'], help='torch device (default: %(default)s)')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='batch size for training (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--latent-dim', type=int, default=2, metavar='N', help='dimension of latent variable (default: %(default)s)')
    parser.add_argument('--prior', type=str, default='mog', choices=['gaussian', 'mog', 'vamp'], help='prior distribution (default: %(default)s)')
    parser.add_argument('--decode', type=str, default='Bernoulli', choices=['Bernoulli', 'Gaussian'], help='decoder distribution (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=1, help='Seed to train on (default: %(default)s)')
    
    args = parser.parse_args()
    print('# Options')
    for key, value in sorted(vars(args).items()):
        print(key, '=', value)

    device = args.device

    # Load MNIST as binarized at 'thresshold' and create data loaders
    if args.decode == "Bernoulli":
        thresshold = 0.5
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
                                                        batch_size=args.batch_size, shuffle=True)
    else:
        mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
                                                                        transform=transforms.Compose([transforms.ToTensor(), transforms . Lambda ( lambda x : x. squeeze () )])),
                                                        batch_size=args.batch_size, shuffle=True)
        mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
                                                                    transform=transforms.Compose([transforms.ToTensor(), transforms . Lambda ( lambda x : x. squeeze () )])),
                                                        batch_size=args.batch_size, shuffle=True)

    M = args.latent_dim
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
    decoder = GaussianDecoder(decoder_net) if args.decode == "Gaussian" else BernoulliDecoder(decoder_net)
    encoder = GaussianEncoder(encoder_net)

    # Define prior distribution
    if args.prior == 'vamp':
        prior = VampPrior(784, 10, encoder)
    elif args.prior == 'gaussian':
        prior = GaussianPrior(M)
    else:
        prior = MoGPrior(M) if args.prior == 'mog' else GaussianPrior(M)
    # prior = MoGPrior(M) if args.prior == 'mog' else GaussianPrior(M)

    model = VAE(prior, decoder, encoder).to(device)
    model_name = args.model + args.prior + "ld"+str(args.latent_dim) + "e"+str(args.epochs) + ".pt"


    # Choose mode to run
    if args.mode == 'train':
        # Define optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Train model
        train(model, optimizer, mnist_train_loader, args.epochs, args.device)

        # Save model
        torch.save(model.state_dict(), model_name)

    elif args.mode == 'sample':
        model.load_state_dict(torch.load(model_name, map_location=torch.device(args.device)))

        # Generate samples
        model.eval()
        with torch.no_grad():
            samples = (model.sample(64)).cpu() 
            #samples = (samples - samples.amin((1,2), keepdim=True)) / (samples.amax((1,2),keepdim=True) - samples.amin((1,2),keepdim=True))
            samples = torch.clip(samples, 0, 1)
            save_image(samples.view(64, 1, 28, 28), f"samples_{args.prior}.png")

            means = []
            std = []
            labels = []
            for batch in mnist_train_loader:
                data = batch[0].to(device)
                dist = model.encoder(data)
                means.append(dist.mean)
                std.append(dist.stddev)
                labels.append(batch[1])
            means = torch.cat(means, dim=0)
            std = torch.cat(std, dim=0)
            labels = torch.cat(labels, dim=0)

        # draw_contours(means, std, model.prior(), f"contours_{args.prior}_{args.latent_dim}.png", multivariate="gaussian"!=args.prior)
        draw_contours_points(means, std, model.prior(), f"contoursp_{args.prior}_{args.latent_dim}.png", title=args.prior, multivariate="gaussian"!=args.prior, labels=labels)
        # draw_contours_points_2d(means, std, model.prior(), f"contoursp2_{args.prior}_{args.latent_dim}.png", multivariate="gaussian"!=args.prior)

    
    elif args.mode == 'eval':

        model.load_state_dict(torch.load(model_name, map_location=torch.device(args.device)))

        # Evaluate ELBO
        elbo_test = evaluate(model, mnist_test_loader, args.device)
        print(f"Test ELBO: {elbo_test:.4f}")

    elif args.mode == "pca":

        # Sample from posterior
        sample_posterior(model, mnist_test_loader, M, model_name,  args.device)


# if __name__ == "__main__":
#     from torchvision import datasets, transforms
#     from torchvision.utils import save_image, make_grid
#     import glob

#     priors = ['gaussian', 'mog', 'vamp']
#     seeds = [0, 1, 2]#, 3, 4]
#     epochs = 100
#     latent_dim = 2
#     batch_size = 64
#     M = latent_dim
#     # Define encoder and decoder networks


#     def main(prior_n, seed):
#         encoder_net = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(784, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, M*2),
#         )

#         decoder_net = nn.Sequential(
#             nn.Linear(M, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 784),
#             nn.Unflatten(-1, (28, 28))
#         )

#         # Define VAE model
#         decoder = BernoulliDecoder(decoder_net)
#         encoder = GaussianEncoder(encoder_net)

#         torch.manual_seed(seed)
#         device = 'cpu'

#             # Define prior distribution
#         if prior_n == 'vamp':
#             prior = VampPrior(784, 10, encoder)
#         elif prior_n == 'gaussian':
#             prior = GaussianPrior(M)
#         else:
#             prior = MoGPrior(M) 


#         thresshold = 0.5
#         mnist_train_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=True, download=True,
#                                                                     transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
#                                                     batch_size=batch_size, shuffle=True)
#         mnist_test_loader = torch.utils.data.DataLoader(datasets.MNIST('data/', train=False, download=True,
#                                                                 transform=transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: (thresshold < x).float().squeeze())])),
#                                                     batch_size=batch_size, shuffle=True)

#         model = VAE(prior, decoder, encoder).to(device)
#         # Define optimizer
#         optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

#         # Train model
#         train(model, optimizer, mnist_train_loader, epochs, device)
#         torch.save(model.state_dict(), f"models_{prior_n}/model_p-{prior_n}_d-{latent_dim}_s-{seed}_e-{epochs}.pt")

#         return evaluate(model, mnist_test_loader, device)

#     elbos = {}
#     for prior in priors:
#         elbos[prior] = []
#         for seed in seeds:
#             elbos[prior].append(main(prior, seed))

#     # calculate mean and std of elbos
#     for prior in priors:
#         mean = torch.tensor(elbos[prior]).mean()
#         std = torch.tensor(elbos[prior]).std()
#         print(f"Prior: {prior}, Mean ELBO: {mean:.4f}, Std ELBO: {std:.4f}")
#     torch.save(elbos, "elbos.pth")