# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024







# TODO : We calculate the energy, but if we want the length we need to take the sqrt. 








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
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patheffects as pe

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
    def __init__(self, decoder_net):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(GaussianDecoder, self).__init__()
        self.decoder_net = decoder_net
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        means = self.decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

class EnsembleDecoder(nn.Module): 
    def __init__(self, decoder_nets):
        """
        Define a Bernoulli decoder distribution based on a given decoder network.

        Parameters:
        encoder_net: [torch.nn.Module]
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super(EnsembleDecoder, self).__init__()
        self.decoder_net = nn.ModuleList(decoder_nets)
        self.number_of_decoders = len(decoder_nets)
        # self.std = nn.Parameter(torch.ones(28, 28) * 0.5, requires_grad=True) # In case you want to learn the std of the gaussian.

    def forward(self, z, choice=None):
        """
        Given a batch of latent variables, return a Bernoulli distribution over the data space.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        """
        if choice is None : choice = torch.randint(0, self.number_of_decoders, size=(1,))[0]
        decoder_net = self.decoder_net[choice]
        means = decoder_net(z)
        return td.Independent(td.Normal(loc=means, scale=1e-1), 3)

class FlattenOutDecoder(nn.Module): 
    def __init__(self, decoder, output_scale=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = decoder 
        self.output_scale = output_scale
        
    def forward(self, z, **kwargs): 
        decoding_dist = self.decoder(z, **kwargs)
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

        elbo = torch.mean(
            self.decoder(z).log_prob(x) - q.log_prob(z) + self.prior().log_prob(z)
        )
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
                loss = model(x)
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

def get_geo(start_p, end_p, decoder, device, mc_samples = 1,  N=100, max_epochs=10):
    decoder = FlattenOutDecoder(decoder, output_scale=False) 
    d = start_p.shape[1]
    good_opt = False
    while not good_opt: 
        # points = torch.randn((N, d), requires_grad=True, device=device)
        points = torch.stack([torch.linspace(start_p[0,0], end_p[0,0], steps=N+2, device=device), torch.linspace(start_p[0, 1], end_p[0, 1], steps=N+2, device=device)], dim=1).detach()
        points = points[1:-1].requires_grad_()
        optimizer = torch.optim.LBFGS([points], lr=1e-1)

        def closure():
            optimizer.zero_grad()
            trajectory = torch.concat((start_p, points, end_p))
            
            curve_energy = 0
            for i in range(len(trajectory)- 1): 
                star_points = torch.cat([decoder(trajectory[i:i+1]) for _ in range(mc_samples)])
                end_points = torch.cat([decoder(trajectory[i+1:i+2]) for _ in range(mc_samples)])
                curve_energy += torch.nn.functional.mse_loss(star_points, end_points)        
            curve_energy.backward()
            return curve_energy  # LBFGS needs the curve_energy value to be returned
        
        curve_energy = torch.inf 
        for i in range(max_epochs):
            curve_energy_new = optimizer.step(closure)  # Pass the closure to LBFGS
            print(curve_energy)
            if i == 0: 
                init_curve_energy = curve_energy_new
            if abs(curve_energy - curve_energy_new) <= 0.00001: 
                good_opt = True
                break 
            curve_energy = curve_energy_new 
        
        if init_curve_energy > curve_energy: good_opt = True

    return torch.concat((start_p, points, end_p)), curve_energy

def curve_length(trajectory, decoder, mode="mean"):
    decoder = FlattenOutDecoder(decoder, output_scale=False)
    if mode=="mean":
        dist = 0 
        for i in range(decoder.decoder.number_of_decoders):    
            img_traj = decoder(trajectory, choice=i)
            dif = img_traj[1:] - img_traj[:-1]
            dist += torch.sum(torch.norm(dif, dim=1))
        dist = dist / decoder.decoder.number_of_decoders
    else: 
        img_traj = decoder(trajectory)
        dif = img_traj[1:] - img_traj[:-1]
        dist = torch.sum(torch.norm(dif, dim=1))
    return dist 
    
# def get_geo(start_p, end_p, decoder, device, N=10, scale=False):
#     decoder = FlattenOutDecoder(decoder, output_scale=scale) 
    
#     def metric(z): 
#         jac = torch.autograd.functional.jacobian(decoder, z).squeeze() 
#         if not scale : 
#             riemannian = jac.T @ jac
#         else : 
#             raise NotImplementedError 
#         return riemannian
    
#     # start_p.requires_grad = True 
#     # end_p.requires_grad = True 
    
#     return geo_min_energy(start_p=start_p, end_p=end_p, metric=metric, N=N, device=device)
    
# def geo_min_energy(start_p, end_p, metric, N, device, max_epochs=100):
#     d = start_p.shape[1]
#     points = torch.randn((N, d), requires_grad=True, device=device)
#     optimizer = torch.optim.LBFGS([points], lr=1e-1)

#     def closure():
#         optimizer.zero_grad()
#         trajectory = torch.concat((start_p, points, end_p))
        
#         loss = 0
#         for i in range(len(trajectory)- 1):
#             dif = (trajectory[i+1:i+2] - trajectory[i:i+1])
#             G = metric(trajectory[i:i+1]) 
#             loss += dif @ G @ dif.T
        
#         loss.backward()
#         return loss  # LBFGS needs the loss value to be returned
    
#     loss = torch.inf 
#     for i in range(max_epochs): 
#         loss_new = optimizer.step(closure)  # Pass the closure to LBFGS
#         print(loss_new)
#         if loss - loss_new < 0.1: break 
#         loss = loss_new 

#     return torch.concat((start_p, points, end_p)), loss

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
        choices=["train", "ntrain", "save_data", "sample", "eval", "geodesics", "evaluate_cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="bigchunker",
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
        default=50,
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
        default=10,
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
        default=20,
        metavar="N",
        help="number of points along the curve (default: %(default)s)",
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

    def new_encoder():
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

    def new_decoder():
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

    # Choose mode to run
    if args.mode == "train":

        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        
        if args.num_decoders == 1: 
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            epochs = args.epochs_per_decoder
        else: 
            model = VAE(
                GaussianPrior(M),
                EnsembleDecoder([new_decoder() for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)
            epochs = args.epochs_per_decoder * args.num_decoders
        
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        train(
            model,
            optimizer,
            mnist_train_loader,
            epochs,
            args.device,
        )
        os.makedirs(f"{experiments_folder}", exist_ok=True)

        torch.save(
            model.state_dict(),
            f"{experiments_folder}/model.pt",
        )
    
    elif args.mode == "ntrain":
        experiments_folder = args.experiment_folder
        os.makedirs(f"{experiments_folder}", exist_ok=True)
        
        for i in range(args.num_reruns):
            if args.num_decoders == 1: 
                model = VAE(
                    GaussianPrior(M),
                    GaussianDecoder(new_decoder()),
                    GaussianEncoder(new_encoder()),
                ).to(device)
                epochs = args.epochs_per_decoder
            else: 
                model = VAE(
                    GaussianPrior(M),
                    EnsembleDecoder([new_decoder() for _ in range(args.num_decoders)]),
                    GaussianEncoder(new_encoder()),
                ).to(device)
                epochs = args.epochs_per_decoder * args.num_decoders
            
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            train(
                model,
                optimizer,
                mnist_train_loader,
                epochs,
                args.device,
            )

            torch.save(
                model.state_dict(),
                f"{experiments_folder}/model_{i}.pt",
            )

    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
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
        model = VAE(
            GaussianPrior(M),
            GaussianDecoder(new_decoder()),
            GaussianEncoder(new_encoder()),
        ).to(device)
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()

        elbos = []
        with torch.no_grad():
            for x, y in mnist_test_loader:
                x = x.to(device)
                elbo = model.elbo(x)
                elbos.append(elbo)
        mean_elbo = torch.tensor(elbos).mean()
        print("Print mean test elbo:", mean_elbo)
    elif args.mode == "save_data": 
        mnist_test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=64, shuffle=True
        )
        batch = next(iter(mnist_test_loader))
        torch.save({'imgs':batch[0], 'labels':batch[0]}, 'data4later.pt')
        
    elif args.mode == "geodesics":

        if args.num_decoders == 1: 
            model = VAE(
                GaussianPrior(M),
                GaussianDecoder(new_decoder()),
                GaussianEncoder(new_encoder()),
            ).to(device)
            length_calculation_ting = "notmean"
        else: 
            model = VAE(
                GaussianPrior(M),
                EnsembleDecoder([new_decoder() for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)
            length_calculation_ting = "mean"
        
        model.load_state_dict(torch.load(args.experiment_folder + "/model.pt"))
        model.eval()
        
        # Store all datapoints in array : 
        latent_points = torch.zeros((len(mnist_train_loader.dataset), 2), device=device)
        labels = torch.zeros((len(mnist_train_loader.dataset)), device=device)
        curr_pointer = 0
        for batch in mnist_train_loader:
            imgs = batch[0].to(device)
            zs = model.encoder(imgs).base_dist.mean
            labels[curr_pointer:curr_pointer+len(imgs)] = batch[1]
            latent_points[curr_pointer:curr_pointer+len(zs)] = zs.detach()
            curr_pointer += len(imgs) 
        
        # Go through K pairs randomly sampled? 
        K = 25
        N = 10
        paths = torch.zeros((K, N+2, 2), device=device)
        lengths = torch.zeros((K), device=device)
        fixed_data = torch.load('data4later.pt')
        imgs = fixed_data['imgs']
        # labels = fixed_data['labels']
        for i in tqdm(range(K)): 
            start_p = model.encoder(imgs[i*2:i*2+1]).base_dist.mean.detach()
            end_p = model.encoder(imgs[i*2+1:i*2+2]).base_dist.mean.detach()
            
            trajectory, energies = get_geo(start_p, end_p, model.decoder, N=N, device=device, mc_samples=10)
            length = curve_length(trajectory=trajectory, decoder=model.decoder, mode=length_calculation_ting)
            
            paths[i] = trajectory.detach()
            lengths[i] = length.detach()
        
        latent_points = latent_points.to('cpu')
        paths = paths.to('cpu')
        lengths = lengths.to('cpu')
        torch.save({'paths': paths, 'lengths': lengths, 'latent_points': latent_points}, args.experiment_folder + '/results25latentspace.pth')
        # temp = torch.load('results25latentspace.pth')
        # paths = temp['paths']
        # lengths = temp['lengths']
        
        cmap = plt.get_cmap("Paired")
        # cmap = plt.get_cmap("tab10")
        colors = [cmap(int(label)) for label in labels]
        unique_labels = torch.unique(labels)

        # Plot latent representation of all points 
        fig = plt.figure()
        plt.scatter(latent_points[:, 0], latent_points[:,1], c=colors, s=1, marker='o')
        legend = [Line2D([0], [0], color=cmap(int(i)), marker='o', linestyle='None', markersize=10, label=f'{int(i)}') for i in unique_labels]
        plt.legend(handles=legend)

        # Plot trajectories 
        cmap2 = plt.get_cmap("spring")
        norm = mcolors.Normalize(vmin=lengths.min().item(), vmax=lengths.max().item())
        sm = cm.ScalarMappable(norm=norm, cmap=cmap2)
        for i in range(5, len(paths)): 
            #plt.plot(paths[i, :, 0].T, paths[i, :, 1].T, '-', alpha=0.3, color=sm.to_rgba(lengths[i]))
            plt.plot(paths[i, :, 0].T, paths[i, :, 1].T, '-', alpha=0.5, color='k')
            plt.plot(paths[i, 0, 0], paths[i, 0, 1], 'k.')
            plt.plot(paths[i, -1, 0], paths[i, -1, 1], 'k.')

        # Plot special paths 
        list_of_shapes = ["*", "s", "p", "P", "D"]  
        for i in range(0, 5): 
            plt.plot(paths[i, :, 0].T, paths[i, :, 1].T, '-', lw=2, alpha=1, color=sm.to_rgba(lengths[i]))#, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
            plt.plot(paths[i, 0, 0], paths[i, 0, 1], markersize=7, marker=list_of_shapes[i], color=sm.to_rgba(lengths[i]))
            plt.plot(paths[i, -1, 0], paths[i, -1, 1], markersize=7, marker=list_of_shapes[i], color=sm.to_rgba(lengths[i]))
        plt.colorbar(sm, ax=fig.axes[0])
        plt.axis('off')
        # Show the distance for each line 
        # for line in range(len(paths)): 
        #     pos = paths[line, N//2]
        #     plt.text(pos[0], pos[1], f'{lengths[line].item():.1f}', fontsize=8, ha='center', va='center', color='black')
        plt.savefig(args.experiment_folder +'/25path_latent_space.png')
        #TODO grid? 
        plt.show()
        
    elif args.mode == "evaluate_cov":  
        N = 10
        distances = torch.ones((args.num_reruns, args.num_decoders, 10))
        eudist = torch.ones((args.num_reruns, 10))

        fixed_data = torch.load('data4later.pt')
        imgs = fixed_data['imgs'][0:20]

        # Go through models, number of decoders, number of pairs
        for i in tqdm(range(args.num_reruns)): 
            # Load model
            model = VAE(
                GaussianPrior(M),
                EnsembleDecoder([new_decoder() for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)
            
            model.load_state_dict(torch.load(args.experiment_folder + f"/model_{i}.pt"))
            model.eval()
            
            encodings = model.encoder(imgs).base_dist.mean.detach() 
            
            eudist[i] = torch.norm(encodings[::2] - encodings[1::2], dim=1) 
            
            for j in range(args.num_decoders): 
                model.decoder.number_of_decoders = j + 1
                
                for k in range(10): 
                    start_p = encodings[k*2:k*2+1]
                    end_p = encodings[k*2+1:k*2+2]
            
                    trajectory, energies = get_geo(start_p, end_p, model.decoder, N=N, device=device, mc_samples=2*(j+1))
                    distances[i,j,k] = curve_length(trajectory=trajectory, decoder=model.decoder, mode='mean')

        # Save the distances 
        torch.save({'geodesic': distances, 'euclidean': eudist}, args.experiment_folder + "/distances.pt")
        
        means = distances.mean(dim=0)
        stds = distances.std(dim=0)
        
        geo_cov = stds/means 
        eu_cov = eudist.std(dim=0) / eudist.mean(dim=0)
        
        mean_geo_cov = geo_cov.mean(dim=-1)
        mean_eu_cov = eu_cov.mean(dim=-1)
        
        plt.plot([1,2,3], mean_geo_cov.detach(), '-g', label="Geodesics")
        plt.plot([1,2,3], torch.ones(3)*mean_eu_cov.detach(), '-r', label="Euclidean")
        plt.legend()
        plt.savefig('thecovofcovsity.png')
        plt.show()        
        