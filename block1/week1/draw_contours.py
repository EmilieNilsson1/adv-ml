import torch as th 
import torch.distributions as td
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D

def project_to_pca_space(mu, sigma, principal_components):
    mu_new = mu @ principal_components
    sigma_new = principal_components.T @ sigma @ principal_components
    return mu_new, sigma_new

def diag_from_std(sigma):
    diag = th.zeros(sigma.shape[0], sigma.shape[1], sigma.shape[1])
    for i in range(sigma.shape[0]):
        diag[i] = th.diag(sigma[i])
    return diag 

def evaluate_distribution_on_grid(distribution, max_v, min_v, n_points=100):
    x = th.linspace(min_v[0],max_v[0],n_points)
    y = th.linspace(min_v[1],max_v[1],n_points)
    X,Y = th.meshgrid(x,y)
    grid = th.stack([X,Y], dim=-1)
    grid = grid.view(-1,2)
    log_probs = distribution.log_prob(grid).detach()
    return th.exp(log_probs).view(n_points,n_points), X, Y

def draw_contours( data_mean, data_std, prior_dist, img_name, multivariate=False):
 
    if multivariate:
        prior_mean = prior_dist.component_distribution.mean
        prior_std = prior_dist.component_distribution.stddev
    else:
        prior_mean = prior_dist.mean.unsqueeze(0)
        prior_std = prior_dist.stddev.unsqueeze(0)
    
    n_datapoints = len(data_mean)
    k = len(prior_mean)
    # Project to 2D space 
    # We should standardize the means before doing PCA on them, but I am not sure what effect that has on the distributions they describe? 
    # Perform PCA
    pca = PCA(n_components=2)  # Set the number of components
    pca.fit_transform(data_mean.detach().numpy())

    principal_components = th.Tensor(pca.components_).T
    
    data_mean_2d, data_std_2d = project_to_pca_space(data_mean, diag_from_std(data_std), principal_components)
    prior_mean_2d, prior_std_2d = project_to_pca_space(prior_mean, diag_from_std(prior_std), principal_components)

    # Re-create distributions 
    prior_components_2d = td.MultivariateNormal(prior_mean_2d, prior_std_2d)
    prior_weights_2d = td.Categorical(probs=th.ones(k)/k)
    prior_distribution_2d = td.MixtureSameFamily(prior_weights_2d, prior_components_2d)

    data_components_2d = td.MultivariateNormal(data_mean_2d, data_std_2d)
    data_weights_2d = td.Categorical(probs=th.ones(n_datapoints)/n_datapoints)
    data_distribution_2d = td.MixtureSameFamily(data_weights_2d, data_components_2d)
    
    # Plot contours 
    data_points_2d = data_mean @ principal_components
    max_data = th.max(data_points_2d, dim=0).values
    min_data = th.min(data_points_2d, dim=0).values

    grid_data, X, Y = evaluate_distribution_on_grid(data_distribution_2d, max_data, min_data)
    grid_prior, _, _ = evaluate_distribution_on_grid(prior_distribution_2d, max_data, min_data)

    # Random points to show : 
    # n_points = 1000
    # random_points = th.random.choice()

    fig, ax = plt.subplots(1,3, figsize=(15,5))

    contour1 = ax[0].contourf(X, Y, grid_data, cmap='viridis')
    ax[0].axis('off')
    ax[0].set_title('Aggregate Posterior')

    contour2 = ax[1].contourf(X, Y, grid_prior, cmap='viridis')
    ax[1].axis('off')
    ax[1].set_title('Prior')

    contour3 = ax[2].contourf(X, Y, grid_data - grid_prior, cmap='viridis')
    ax[2].axis('off')
    ax[2].set_title('Difference')
    fig.colorbar(contour1, ax=ax[0], orientation='horizontal', pad=0.01)
    fig.colorbar(contour2, ax=ax[1], orientation='horizontal', pad=0.01)
    fig.colorbar(contour3, ax=ax[2], orientation='horizontal', pad=0.01)

    fig.tight_layout()
    plt.savefig(img_name)
    
def draw_contours_points(data_mean, data_std, prior_dist, img_name, title, multivariate=False, labels=None):    
 
    if multivariate:
        prior_mean = prior_dist.component_distribution.mean
        prior_std = prior_dist.component_distribution.stddev
        prior_weights = prior_dist.mixture_distribution.probs
    else:
        prior_mean = prior_dist.mean.unsqueeze(0)
        prior_std = prior_dist.stddev.unsqueeze(0)
        prior_weights = th.ones(1)
    
    n_datapoints = len(data_mean)
    k = len(prior_mean)
    # Project to 2D space 
    # We should standardize the means before doing PCA on them, but I am not sure what effect that has on the distributions they describe? 
    # Perform PCA
    pca = PCA(n_components=2)  # Set the number of components
    pca.fit_transform(data_mean.detach().numpy())

    principal_components = th.Tensor(pca.components_).T
    
    data_mean_2d, data_std_2d = project_to_pca_space(data_mean, diag_from_std(data_std), principal_components)
    prior_mean_2d, prior_std_2d = project_to_pca_space(prior_mean, diag_from_std(prior_std), principal_components)

    # Re-create distributions 
    prior_components_2d = td.MultivariateNormal(prior_mean_2d, prior_std_2d)
    prior_weights_2d = td.Categorical(probs=prior_weights)
    prior_distribution_2d = td.MixtureSameFamily(prior_weights_2d, prior_components_2d)

    data_components_2d = td.MultivariateNormal(data_mean_2d, data_std_2d)
    data_weights_2d = td.Categorical(probs=th.ones(n_datapoints)/n_datapoints)
    data_distribution_2d = td.MixtureSameFamily(data_weights_2d, data_components_2d)

    # Calculate KL-divergence 
    samples = prior_distribution_2d.sample((1000,))
    log_p = data_distribution_2d.log_prob(samples)
    log_q = prior_distribution_2d.log_prob(samples)
    kl = th.mean(log_q - log_p)
    
    # Plot contours 
    data_points_2d = data_mean @ principal_components
    max_data = [5, 5] # th.max(data_points_2d, dim=0).values
    min_data = [-5, -5] #th.min(data_points_2d, dim=0).values

    grid_data, X, Y = evaluate_distribution_on_grid(data_distribution_2d, max_data, min_data)
    grid_prior, _, _ = evaluate_distribution_on_grid(prior_distribution_2d, max_data, min_data)
    vmax = th.max(grid_data.max(), grid_prior.max())
    vmin = 0#th.min(grid_data.min(), grid_prior.min())

    # Random points to show : 
    # n_points = 1000
    # random_points = th.random.choice()

    fig, ax = plt.subplots(2,2, figsize=(10,10))

    if labels is not None: 
        for i, label in enumerate(th.arange(10)): 
            idx = labels == label
            ax[0,0].plot(data_points_2d[idx,0], data_points_2d[idx,1], '.', alpha=0.2, color=plt.cm.tab10(i))
        legend_handles = [Line2D([0], [0], color=plt.cm.tab10(i), marker='o', linestyle='None', markersize=10, label=str(i)) for i in range(10)]
        fig.legend(handles=legend_handles,
            loc='center left', bbox_to_anchor=(0.04, 0.51), ncol=10,
            columnspacing=0.1, handletextpad=0.2
        )
        # fig.legend( loc='center left', bbox_to_anchor=(0.04, 0.51), ncol=10, columnspacing=0.1)
    else : 
        ax[0,0].plot(data_points_2d[:,0], data_points_2d[:,1], 'y,', alpha=0.2) 
    dummy_contour = ScalarMappable(cmap='viridis')
    dummy_contour.set_array([])
    dummy_cbar = fig.colorbar(dummy_contour, ax=ax[0,0], orientation='horizontal', pad=0.01)
    dummy_cbar.outline.set_visible(False)
    dummy_cbar.ax.set_xticklabels([])
    dummy_cbar.ax.set_xticks([])
    dummy_cbar.ax.set_yticklabels([])
    dummy_cbar.ax.set_yticks([])
    dummy_cbar.set_alpha(0.0)
    dummy_cbar.ax.set_visible(False)


    ax[0,0].set_ylim(min_data[0], max_data[0])
    ax[0,0].set_xlim(min_data[1], max_data[1])
    ax[0,0].set_title('Aggregate Posterior Mean Samples')

    contour1 = ax[1,0].pcolormesh(X, Y, grid_data, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1,0].contour(X, Y, grid_data, linewidths=1, colors='black', vmin=vmin, vmax=vmax)
    ax[1,0].axis('off')
    ax[1,0].set_title('Aggregate Posterior')

    contour2 = ax[1,1].pcolormesh(X, Y, grid_prior, cmap='viridis', vmin=vmin, vmax=vmax)
    ax[1,1].contour(X, Y, grid_prior, linewidths=1, colors='black', vmin=vmin, vmax=vmax)
    ax[1,1].axis('off')
    ax[1,1].set_title('Prior')

    contour3 = ax[0,1].pcolormesh(X, Y, th.abs(grid_data - grid_prior), cmap='viridis', vmin=vmin, vmax=vmax)
    ax[0,1].contour(X, Y, grid_data - grid_prior, linewidths=1, colors='black')
    ax[0,1].axis('off')
    ax[0,1].set_title(f'Difference. KL(prior, posterior) = {kl:.4f}')
    fig.colorbar(contour1, ax=ax[1,0], orientation='horizontal', pad=0.01)
    fig.colorbar(contour2, ax=ax[1,1], orientation='horizontal', pad=0.01)
    fig.colorbar(contour3, ax=ax[0,1], orientation='horizontal', pad=0.01)
    

    fig.suptitle(title)
    fig.tight_layout()
    plt.savefig(img_name)

def draw_contours_points_2d(data_mean, data_std, prior_dist, img_name, multivariate=False):
 
    if multivariate:
        prior_mean = prior_dist.component_distribution.mean
        prior_std = prior_dist.component_distribution.stddev
        prior_weights = prior_dist.mixture_distribution.probs
    else:
        prior_mean = prior_dist.mean.unsqueeze(0)
        prior_std = prior_dist.stddev.unsqueeze(0)
    
    n_datapoints = len(data_mean)
    k = len(prior_mean)

    prior_std = diag_from_std(prior_std)
    data_std = diag_from_std(data_std)
    # Re-create distributions 
    prior_components_2d = td.MultivariateNormal(prior_mean, prior_std)
    prior_weights_2d = td.Categorical(probs=prior_weights)
    prior_distribution_2d = td.MixtureSameFamily(prior_weights_2d, prior_components_2d)

    data_components_2d = td.MultivariateNormal(data_mean, data_std)
    data_weights_2d = td.Categorical(probs=th.ones(n_datapoints)/n_datapoints)
    data_distribution_2d = td.MixtureSameFamily(data_weights_2d, data_components_2d)
    
    # Plot contours 
    data_points_2d = data_mean
    max_data = th.max(data_points_2d, dim=0).values
    min_data = th.min(data_points_2d, dim=0).values

    grid_data, X, Y = evaluate_distribution_on_grid(data_distribution_2d, max_data, min_data)
    grid_prior, _, _ = evaluate_distribution_on_grid(prior_distribution_2d, max_data, min_data)

    # Random points to show : 
    # n_points = 1000
    # random_points = th.random.choice()

    fig, ax = plt.subplots(1,4, figsize=(15,5))

    ax[0].plot(data_points_2d[:,0], data_points_2d[:,1], 'y.', alpha=0.01)

    contour1 = ax[1].contourf(X, Y, grid_data, cmap='viridis')
    ax[1].axis('off')
    ax[1].set_title('Aggregate Posterior')

    contour2 = ax[2].contourf(X, Y, grid_prior, cmap='viridis')
    ax[2].axis('off')
    ax[2].set_title('Prior')

    contour3 = ax[3].contourf(X, Y, grid_data - grid_prior, cmap='viridis')
    ax[3].axis('off')
    ax[3].set_title('Difference')
    fig.colorbar(contour1, ax=ax[1], orientation='horizontal', pad=0.01)
    fig.colorbar(contour2, ax=ax[2], orientation='horizontal', pad=0.01)
    fig.colorbar(contour3, ax=ax[3], orientation='horizontal', pad=0.01)

    fig.tight_layout()
    plt.savefig(img_name)

    

if __name__ == '__main__':
    k = 10 
    d = 32 

    # Distribution for parameters of the Gaussian components
    mus = th.randn(k, d)
    sigmas = th.rand(k, d)**2

    gaus_distributions = td.Independent(td.Normal(mus, sigmas), 1)
    distribution = td.Categorical(probs=th.ones(k)/k)
    mixture = td.MixtureSameFamily(distribution, gaus_distributions)

    # Datapoints distributions 
    n_datapoints = 60000
    encoded_m = th.randn(n_datapoints, d)
    encoded_s = 0.1*th.ones(n_datapoints, d) #th.randn(n_datapoints, 2)**2
    data_components = td.Independent(td.Normal(encoded_m, encoded_s), 1)
    data_weights = td.Categorical(probs=th.ones(n_datapoints)/n_datapoints)
    data_distribution = td.MixtureSameFamily(data_weights, data_components)
    
