import numpy as np
# import torch
from scipy.linalg import sqrtm
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import cdist
from sklearn.metrics.pairwise import rbf_kernel
from skimage.metrics import structural_similarity as ssim
import logging
from scipy.stats import wasserstein_distance, entropy

def calculate_ssim_tabular(real_data, synthetic_data, window_size=None, data_range=None):
    if data_range is None:
        data_range = real_data.max() - real_data.min()
    
    data_1 = real_data.reshape(real_data.shape[0], 18, 20)
    data_2 = synthetic_data.reshape(synthetic_data.shape[0], 18, 20)
    
    ssim_scores = []
    for i in range(data_1.shape[0]):
        ssim_scores.append(
            ssim(data_1[i], data_2[i], 
                 data_range=data_range, win_size=window_size)
        )
    
    return np.nanmean(ssim_scores)

def calculate_fid_tabular(real_data, synthetic_data):
    data_1 = real_data.reshape(real_data.shape[0], -1)
    data_2 = synthetic_data.reshape(synthetic_data.shape[0], -1)

    # Calculate mean and covariance statistics
    mu1, sigma1 = data_1.mean(axis=0), np.cov(data_1, rowvar=False)
    mu2, sigma2 = data_2.mean(axis=0), np.cov(data_2, rowvar=False)
    
    # Calculate squared difference between means
    ssdiff = np.sum((mu1 - mu2)**2.0)
    
    # Calculate sqrt of product between covariances
    covmean = sqrtm(sigma1.dot(sigma2))
    
    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate score
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def calculate_wasserstein_distance(real_data, synthetic_data):
    data_1 = real_data.reshape(real_data.shape[0], -1)
    data_2 = synthetic_data.reshape(synthetic_data.shape[0], -1)

    n_features = data_1.shape[1]
    w_distances = []
    for i in range(n_features):
        feature1 = data_1[:, i]
        feature2 = data_2[:, i]
        valid_feature1 = feature1[np.isfinite(feature1)]
        valid_feature2 = feature2[np.isfinite(feature2)]

        # Ensure there are still samples left after removing non-finite ones
        if len(valid_feature1) > 0 and len(valid_feature2) > 0:
             w_distances.append(wasserstein_distance(valid_feature1, valid_feature2))
        else:
             # Handle cases where a feature might be all NaN/inf or empty
             w_distances.append(np.nan)

    return np.nanmean(w_distances)

def calculate_kullback_leibler_divergence(real_data, synthetic_data, n_bins=50):
    data_1 = real_data.reshape(real_data.shape[0], -1)
    data_2 = synthetic_data.reshape(synthetic_data.shape[0], -1)

    n_features = data_1.shape[1]
    kl_divergences = []
    # combined_finite_data = np.vstack([
    #     data_1[np.isfinite(data_1).all(axis=1)],
    #     data_2[np.isfinite(data_2).all(axis=1)]
    # ])

    for i in range(n_features):
        feature1 = data_1[:, i]
        feature2 = data_2[:, i]

        hist1, bin_edges = np.histogram(feature1, bins=n_bins, range=(0, 1), density=True)
        hist2, _ = np.histogram(feature2, bins=bin_edges, density=True)

        epsilon = 1e-10
        hist1 = hist1 + epsilon
        hist2 = hist2 + epsilon

        hist1 /= np.sum(hist1)
        hist2 /= np.sum(hist2)

        # Calculate KL divergence
        kl_div = entropy(hist1, hist2)
        if np.isinf(kl_div) or np.isnan(kl_div):
            kl_divergences.append(np.nan)
        else:
            kl_divergences.append(kl_div)

    return np.nanmean(kl_divergences)

def calculate_mmd_rbf(real_data, synthetic_data, gamma=None):
    data_1 = real_data.reshape(real_data.shape[0], -1)
    data_2 = synthetic_data.reshape(synthetic_data.shape[0], -1)

    Kxx = rbf_kernel(data_1, data_1, gamma=gamma)
    Kyy = rbf_kernel(data_2, data_2, gamma=gamma)
    Kxy = rbf_kernel(data_1, data_2, gamma=gamma)

    n1 = data_1.shape[0]
    n2 = data_2.shape[0]
    mmd2 = (np.sum(Kxx) / (n1 * n1) +
            np.sum(Kyy) / (n2 * n2) -
            2 * np.sum(Kxy) / (n1 * n2))
    
    if np.isnan(mmd2):
        return np.nan
    else:
        return np.sqrt(max(0, mmd2))

def calculate_metrics(real_data, synthetic_data, category=''):
    metrics = {
        f'{category}_ssim': calculate_ssim_tabular(real_data, synthetic_data),
        f'{category}_fid': calculate_fid_tabular(real_data, synthetic_data),
        f'{category}_wd': calculate_wasserstein_distance(real_data, synthetic_data),
        f'{category}_kld': calculate_kullback_leibler_divergence(real_data, synthetic_data),
        f'{category}_mmd': calculate_mmd_rbf(real_data, synthetic_data)
    }

    logger = logging.getLogger('eval_dist')
    logger.info(f'[{category}]: {list(metrics.values())}')

    return metrics
