import numpy as np
# import torch
from scipy.linalg import sqrtm
# from sklearn.decomposition import PCA
# from scipy.spatial.distance import cdist
from skimage.metrics import structural_similarity as ssim
import logging

def calculate_ssim_tabular(real_data, synthetic_data, window_size=7, data_range=None):
    """
    Calculate SSIM for tabular data by treating each sample as a "row image"
    
    Args:
        real_data: numpy array of real samples (n_samples, n_steps, n_features)
        synthetic_data: numpy array of synthetic samples (n_samples, n_steps, n_features)
        window_size: size of the SSIM comparison window
        data_range: range of the data (max - min)
        
    Returns:
        ssim_score: mean SSIM score across all samples
    """
    if data_range is None:
        data_range = real_data.max() - real_data.min()
    
    # Reshape each sample to be 2D (as SSIM expects images)
    real_reshaped = real_data.reshape(-1, 18, 20)
    synthetic_reshaped = synthetic_data.reshape(-1, 18, 20)
    
    ssim_scores = []
    for i in range(real_reshaped.shape[0]):
        ssim_scores.append(
            ssim(real_reshaped[i], synthetic_reshaped[i], 
                 data_range=data_range, win_size=window_size)
        )
    
    return np.mean(ssim_scores)

def calculate_fid_tabular(real_data, synthetic_data):
    """
    Calculate FID score for tabular data
    
    Args:
        real_data: numpy array of real samples (n_samples, n_steps, n_features)
        synthetic_data: numpy array of synthetic samples (n_samples, n_steps, n_features)
        
    Returns:
        fid_score: Frechet Inception Distance score
    """
    real_reshaped = real_data.reshape(-1, 5*72)
    synthetic_reshaped = synthetic_data.reshape(-1, 5*72)

    # Calculate mean and covariance statistics
    mu1, sigma1 = real_reshaped.mean(axis=0), np.cov(real_reshaped, rowvar=False)
    mu2, sigma2 = synthetic_reshaped.mean(axis=0), np.cov(synthetic_reshaped, rowvar=False)
    
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

def calculate_metrics(real_data, synthetic_data, num_samples=100, sample_size=50):
    """
    Calculate both SSIM and FID metrics for tabular data (Adapted to GAN tablular)
    
    Args:
        real_data: numpy array of real samples (n_samples, n_steps, n_features)
        synthetic_data: numpy array of synthetic samples (n_samples, n_steps, n_features)
        
    Returns:
        Dictionary containing both metrics
    """

    metrics = {
        'SSIM': [],
        'FID': [],
    }

    for sample in range(num_samples):
        real_samples = np.random.choice(real_data.shape[0], size=sample_size, replace=False)
        synthetic_samples = np.random.choice(synthetic_data.shape[0], size=sample_size, replace=False)

        metrics['SSIM'].append(calculate_ssim_tabular(real_data[real_samples], synthetic_data[synthetic_samples]))
        metrics['FID'].append(calculate_fid_tabular(real_data[real_samples], synthetic_data[synthetic_samples]))
    
    logger = logging.getLogger('eval_dist')
    logger.info('SSIM: {} (mean {})'.format(metrics['SSIM'][:10], np.mean(metrics['SSIM'])))
    logger.info('FID: {} (mean {})'.format(metrics['FID'][:10], np.mean(metrics['FID'])))

    return metrics
