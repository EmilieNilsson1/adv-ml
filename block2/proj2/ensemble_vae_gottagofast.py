# Code for DTU course 02460 (Advanced Machine Learning Spring) by Jes Frellsen, 2024
# Version 1.0 (2024-01-27)
# Inspiration is taken from:
# - https://github.com/jmtomczak/intro_dgm/blob/main/vaes/vae_example.ipynb
# - https://github.com/kampta/pytorch-distributions/blob/master/gaussian_vae.py
#
# Significant extension by Søren Hauberg, 2024

from typing import Iterable
import os
import argparse

import torch
import torch.nn as nn
import torch.distributions as td
import torch.optim.lbfgs
import torch.utils.data
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import matplotlib as mpl
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D




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


class Ensemble(nn.Module):
    def __init__(self, models: Iterable[nn.Module]):
        """
        Define an ensemble of models.

        Parameters:
        models: [list[torch.nn.Module]]
           A list of models to be used in the ensemble.
        """
        super().__init__()
        self.models = nn.ModuleList(models)


    def forward(self, z: torch.Tensor, model_idx: int = None):
        """
        Sample a model uniformly and use it for the forward pass.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        model_idx: [int]
           Index of the model to use for the forward pass. If None, a model is sampled uniformly.
        """
        if model_idx is None:
            model_idx = torch.randint(0, len(self.models), (1,)).item()

        return self.models[model_idx](z)


    def forward_mean(self, z: torch.Tensor, model_idx: torch.Tensor = None):
        """
        Sample a model uniformly and use it for the forward pass.

        Parameters:
        z: [torch.Tensor]
           A tensor of dimension `(batch_size, M)`, where M is the dimension of the latent space.
        model_idx: [torch.Tensor]
           If a single integer is provided, use this model for the forward pass of the whole batch.
           If a tensor of multiple elements is passed, it must have same length as the batch size. 
           Each element in the batch is then processed by the corresponding model.
           If None, a model is sampled uniformly to use for the whole batch.
        """
        # If no model provided, sample uniformly
        if model_idx is None:
            model_idx = torch.randint(0, len(self.models), (1,)).item()

        # If only one model is specified, use it for the entire batch
        if model_idx.numel() == 1:
            model_idx = model_idx.expand(z.shape[0])

        assert model_idx.shape[0] == z.shape[0], "model_idx must have same length as batch size"


        # Forward pass over the points assigned to each model
        def forward_model(idx):
            return self.models[idx](z[model_idx == idx]).mean

        output_models = [
            forward_model(i)
            for i in range(len(self.models))
        ]

        # Reorder the outputs to match the original order
        output = torch.empty((len(z), *output_models[0].shape[1:]), device=z.device, dtype=z.dtype)
        for i, out in enumerate(output_models):
            output[model_idx == i] = out

        return output


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

    with tqdm(range(num_steps), leave=False) as pbar:
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



def get_pairs(n_pairs: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(12)
    return torch.randperm(len(test_data), generator=rng)[:(n_pairs*2)].view(-1, 2)


def curve(latent_pairs: torch.Tensor, inner_points: torch.Tensor) -> torch.Tensor:
    return torch.concat((latent_pairs[:, 0:1], inner_points, latent_pairs[:, 1:2]), dim=1)
    # return torch.stack([
    #     torch.concat([start[None], inner, end[None]])
    #     for (start, end), inner in zip(latent_pairs, inner_points)
    # ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        default="plot",
        choices=["train", "sample", "eval", "geodesics", "plot", "cov"],
        help="what to do when running the script (default: %(default)s)",
    )
    parser.add_argument(
        "--experiment-folder",
        type=str,
        default="experiments",
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
        default="cuda",
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
        "--num-decoders",
        type=int,
        default=1,
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
        "--plot-rerun-index",
        type=int,
        default=1,
        metavar="N",
        help="index of rerun to plot (default: %(default)s)",
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
    M = 2

    def new_encoder():
        encoder_net = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.Softmax(1),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.Softmax(1),
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
            nn.Softmax(1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 32, 3, stride=2, padding=1, output_padding=0),
            nn.Softmax(1),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.Softmax(1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
        )
        return decoder_net

    # Choose mode to run
    if args.mode == "train":
        os.makedirs(f"{args.experiment_folder}/models/{args.num_decoders}/", exist_ok=True)

        for rerun_idx in tqdm(range(args.num_reruns), desc=f"Reruns for num_decoders: {args.num_decoders}"):
            model = VAE(
                GaussianPrior(M),
                Ensemble([GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

            train(
                model,
                optimizer,
                mnist_train_loader,
                args.epochs_per_decoder * args.num_decoders,
                # # NOTE: Not adjusting by number of decoders, because the file came like this
                # args.epochs_per_decoder,
                args.device,
            )

            torch.save(
                model.state_dict(),
                f"{args.experiment_folder}/models/{args.num_decoders}/{rerun_idx}.pt",
            )


    elif args.mode == "sample":
        model = VAE(
            GaussianPrior(M),
            Ensemble([GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]),
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
            Ensemble([GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]),
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

    elif args.mode == "geodesics":
        # Settings
        n_pairs = 25
        n_curve_points = 20
        n_energy_samples = 2 * args.num_decoders
        learning_rate = 4e-2
        steps_per_decoder = 400 // args.num_decoders
        tolerance = 1e-1
        tolerance_divergence = 1.10
        tolerance_diverged = 10.00
        
        # Make results directory for geodesics
        os.makedirs(f"{args.experiment_folder}/curves/{args.num_decoders}/", exist_ok=True)
        os.makedirs(f"{args.experiment_folder}/latents/{args.num_decoders}/", exist_ok=True)
        os.makedirs(f"{args.experiment_folder}/decoded/{args.num_decoders}/", exist_ok=True)

        # Get important pairs
        rng = torch.Generator(device).manual_seed(1337)
        pair_idx = get_pairs(n_pairs)

        # Get the data
        # NOTE: If using the training, make sure it does NOT shuffle here
        images, labels = test_data.tensors


        for rerun_idx in tqdm(range(args.num_reruns), desc=f"Reruns for num_decoders: {args.num_decoders}"):
            # Use new encoder all the time
            model = VAE(
                GaussianPrior(M),
                Ensemble([GaussianDecoder(new_decoder()) for _ in range(args.num_decoders)]),
                GaussianEncoder(new_encoder()),
            ).to(device)

            model.load_state_dict(torch.load(f"{args.experiment_folder}/models/{args.num_decoders}/{rerun_idx}.pt"))
            model.eval()


            # Use the same encoder for all decoders
            # model = VAE(
            #     GaussianPrior(M),
            #     Ensemble([GaussianDecoder(new_decoder()) for _ in range(10)]),
            #     GaussianEncoder(new_encoder()),
            # ).to(device)

            # model.load_state_dict(torch.load(f"{args.experiment_folder}/models/10/{rerun_idx}.pt"))
            # model.eval()

            # model.decoder.models = model.decoder.models[:args.num_decoders]




            # Shape pipeline
            # ======================================================================
            # Latents
            #  [points, latent_dim]

            # Latent pairs
            #  [point_pairs, 2, latent_dim]

            # Curve
            #  [point_pairs, curve_points, latent_dim]
            
            # Decoded
            #  [point_pairs, curve_points, n_energy_samples, ...decoded_dim]

            # Diff. Norm Squared
            #  [point_pairs, curve_points-1, n_energy_samples]
            
            # Expectation
            #  [point_pairs, curve_points-1]

            # Energy
            #  [point_pairs]

            # Get latents for pairs
            with torch.no_grad():
                latents = torch.concat([
                    model.encoder(img.to(device=device)).mean
                    for (img, _) in tqdm(torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False), desc="Calculating latents", leave=False)
                ])

            # Adjust to correct pair shape
            latent_pairs = latents[pair_idx.view(-1)].view(n_pairs, 2, M)


            # Loop to enable retrying if diverged
            while True:
                # Curve parameterized as a line

                # Initialized as line between start and end
                # params = torch.stack([
                #     torch.stack([
                #         torch.linspace(start[d], end[d], n_curve_points)[1:-1]
                #         for d in range(M)
                #     ], dim=1)
                #     for start, end in latent_pairs
                # ])
                
                # Initialized as line between start and end with noise
                params = torch.stack([
                    torch.stack([
                        torch.linspace(start[d], end[d], n_curve_points, device=device)[1:-1] + (torch.rand(n_curve_points - 2, device=device, generator=rng) - 0.5) * (end[d] - start[d]) / (n_curve_points - 2) * 4
                        for d in range(M)
                    ], dim=1)
                    for start, end in latent_pairs
                ])

                # Initialized as random points between start and end
                # params = torch.stack([
                #     torch.stack([
                #         (torch.rand(n_curve_points - 2, device=device, generator=rng) - 0.5) * (end[d] - start[d]) + (start[d] + end[d]) / 2
                #         for d in range(M)
                #     ], dim=1)
                #     for start, end in latent_pairs
                # ])



                # [point_pairs, parameter_size]
                params = params.to(device).requires_grad_()
                # params = [param.requires_grad_() for param in params.to(device)]

                # optimizer = torch.optim.AdamW(
                #     params=[params],
                #     lr=learning_rate,
                #     weight_decay=0.0,
                # )

                optimizer = torch.optim.LBFGS(
                    params=[params],
                    lr=learning_rate,
                    tolerance_change=tolerance,
                )
                # optimizers = [
                #     torch.optim.AdamW(
                #         params=[param],
                #         lr=learning_rate,
                #         weight_decay=0.0
                #     )
                #     for param in params
                # ]


                def curve_energy() -> torch.Tensor:
                    optimizer.zero_grad()
                    z = curve(latent_pairs, params)
                    # for opt in optimizers:
                    #     opt.zero_grad()
                    # z = curve(latent_pairs, torch.stack(params))
                    z = z[:, :, None, ...].expand(-1, -1, n_energy_samples, -1).reshape(-1, M)

                    model_idx = torch.randint(0, args.num_decoders, (z.shape[0],), device=device, generator=rng)
                    x = model.decoder.forward_mean(z, model_idx)
                    x = x.view(n_pairs, n_curve_points, n_energy_samples, -1)

                    # NOTE: Optimizing the summed energy functions of all curves at the same time
                    #        (               Norm              ) (  MC  ) (segs.) (pairs)
                    energy = ((x[:, :-1] - x[:, 1:])**2).sum(-1).mean(-1).sum(-1).sum(-1)
                    energy.backward()

                    return energy


                # Optimize parameters
                energy_best = curve_energy()
                for _ in (pbar := tqdm(range(steps_per_decoder * args.num_decoders), leave=False)):
                    energy = optimizer.step(curve_energy)
                    # energy = curve_energy()
                    # for opt in optimizers:
                    #     opt.step()
                    
                    
                    pbar.set_postfix(mean_energy=f"{energy/n_pairs:.4f}")
                    if (energy := energy.item()) < energy_best:
                        energy_best = energy

                    if energy > energy_best * tolerance_diverged:
                        break


                # If not diverged, break, else try again
                # NOTE: May get stuck in a loop, if unable to converge within chosen amount of steps
                if energy < energy_best * tolerance_divergence:
                    break


            # Final decoded latents on the geodesic
            with torch.no_grad():
                z = curve(latent_pairs, params)
                # z = curve(latent_pairs, torch.stack(params))
                z = z[:, :, None, ...].expand(-1, -1, args.num_decoders, -1).reshape(-1, M)

                model_idx = torch.arange(args.num_decoders, device=device).repeat(n_pairs*n_curve_points)
                x = model.decoder.forward_mean(z, model_idx)
                x = x.view(n_pairs, n_curve_points, args.num_decoders, *x.shape[1:])


            # Save the parameters, latents, and decoded latents
            torch.save(params.detach().cpu(), f"{args.experiment_folder}/curves/{args.num_decoders}/{rerun_idx}.pt")
            # torch.save(torch.stack(params).detach().cpu(), f"{args.experiment_folder}/curves/{args.num_decoders}/{rerun_idx}.pt")
            torch.save(latents.detach().cpu(), f"{args.experiment_folder}/latents/{args.num_decoders}/{rerun_idx}.pt")
            torch.save(x.cpu(), f"{args.experiment_folder}/decoded/{args.num_decoders}/{rerun_idx}.pt")


    elif args.mode == "plot":
        # Settings
        min_dist, max_dist = 0.0, 20
        
        # Load data
        params = torch.load(f"{args.experiment_folder}/curves/{args.num_decoders}/{args.plot_rerun_index}.pt")
        latents = torch.load(f"{args.experiment_folder}/latents/{args.num_decoders}/{args.plot_rerun_index}.pt")
        decoded = torch.load(f"{args.experiment_folder}/decoded/{args.num_decoders}/{args.plot_rerun_index}.pt")
        labels = test_data.tensors[1]

        # Get important pairs
        n_pairs = params.shape[0]
        pair_idx = get_pairs(n_pairs)

        latent_pairs = latents[pair_idx.view(-1)].view(n_pairs, 2, M)
        curves = curve(latent_pairs, params)
        dists = (decoded[:, :-1] - decoded[:, 1:]).view(n_pairs, curves.shape[1]-1, args.num_decoders, -1).norm(dim=-1, p=2).sum(dim=1).mean(dim=-1)

        # Plot the latent space
        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        
        # Disable axis
        # ax.set_axis_off()

        # Plot all the latents
        cmap_latent = mpl.colormaps["Paired"]
        ax.scatter(*latents.T, c=[cmap_latent(int(l)) for l in labels], marker='o', s=1)
        ax.legend(handles=[Line2D([0], [0], color=cmap_latent(l), marker='o', linestyle="None", label=l) for l in range(3)])

        # Plot lantent pairs
        for (start, end), c in zip(latent_pairs, curves):
            ax.scatter([start[0], end[0]], [start[1], end[1]], c='k', marker='o', s=4)
            ax.plot(*c.T, "k", alpha=0.5, lw=1.0)


        # Plot highlighted curves
        shapes = ["o", "s", "p", "P", "D"]
        # shapes = ["$1$", "$2$", "$3$", "$4$", "$5$"]
        cmap_highlight = mpl.colormaps["spring"]
        cmap_norm = mcolors.Normalize(vmin=min_dist, vmax=max_dist)
        cmap_mapper = ScalarMappable(norm=cmap_norm, cmap=cmap_highlight)
        
        for shape, (start, end), c, d in zip(shapes, latent_pairs, curves, dists):
            color = cmap_mapper.to_rgba(d)
            ax.plot(*c.T, color=color, alpha=0.8, lw=1.5)
            ax.scatter([start[0], end[0]], [start[1], end[1]], color=color, edgecolors="purple", lw=0.5, marker=shape, s=32, zorder=10)

        # Add colorbar
        cbar = plt.colorbar(cmap_mapper, ax=ax)

        plt.show()


    elif args.mode == "cov":
        n_pairs = 25
        dists_all = torch.empty((n_pairs, args.num_decoders, args.num_reruns))
        dists_all_euclidean = torch.empty((n_pairs, args.num_decoders, args.num_reruns))

        for num_decoders in tqdm(range(1, args.num_decoders+1), desc="Num. decoders"):
            for rerun_idx in tqdm(range(args.num_reruns), desc=f"Rerun", leave=False):
                params = torch.load(f"{args.experiment_folder}/curves/{num_decoders}/{rerun_idx}.pt")
                latents = torch.load(f"{args.experiment_folder}/latents/{num_decoders}/{rerun_idx}.pt")
                decoded = torch.load(f"{args.experiment_folder}/decoded/{num_decoders}/{rerun_idx}.pt")

                n_pairs = params.shape[0]
                pair_idx = get_pairs(n_pairs)

                latent_pairs = latents[pair_idx.view(-1)].view(n_pairs, 2, M)
                curves = curve(latent_pairs, params)
                dists = (decoded[:, :-1] - decoded[:, 1:]).view(n_pairs, curves.shape[1]-1, num_decoders, -1).norm(dim=-1, p=2).sum(dim=1).mean(dim=-1)


                dists_all[:, num_decoders-1, rerun_idx] = dists
                dists_all_euclidean[:, num_decoders-1, rerun_idx] = (latent_pairs[:, 1] - latent_pairs[:, 0]).norm(dim=-1, p=2)


        cov = dists_all.std(dim=-1) / dists_all.mean(dim=-1)

        dists_all_euclidean = dists_all_euclidean
        cov_euclidean = dists_all_euclidean.std(dim=-1) / dists_all_euclidean.mean(dim=-1)

        fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
        ax.plot(range(1, args.num_decoders+1), cov_euclidean.mean(0), label="Euclidean")
        # NOTE: Outliers are omitted from the boxplot
        ax.boxplot(cov, label="Geodesics", showfliers=False)
        # ax.plot(range(1, args.num_decoders+1), cov.T, label="Geodesics", color="red")
        ax.plot(range(1, args.num_decoders+1), cov.mean(0), color="orange")
        
        ax.set_ylim(0.0, 0.40)

        ax.legend()
        ax.grid()

        plt.show()