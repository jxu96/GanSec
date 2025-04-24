import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

dnn_res = 'saved/dnn/dist.mat'
rnn_res = 'saved/rnn/dist.mat'
cnn_res = 'saved/cnn/dist.mat'

# baselines
tanogan_res = 'backups/tanogan/dist.mat'
deepsmote_res = 'backups/deepsmote/dist.mat'

dnn_res = loadmat(dnn_res, squeeze_me=True)
rnn_res = loadmat(rnn_res, squeeze_me=True)
cnn_res = loadmat(cnn_res, squeeze_me=True)

tanogan_res = loadmat(tanogan_res, squeeze_me=True)
deepsmote_res = loadmat(deepsmote_res, squeeze_me=True)

keys = [
    'ec-gan_0_fid',
    'ec-gan_1_fid',
    'co-gan_0_fid',
    'co-gan_1_fid',
    'ec-gan_0_mmd',
    'ec-gan_1_mmd',
    'co-gan_0_mmd',
    'co-gan_1_mmd',
]

for alg in ['ec', 'co']:
    for metric in ['fid', 'mmd']:
        n_bins = np.arange(0, 70, 0.1) if metric == 'fid' else np.arange(0, 1, 0.0001)

        plt.figure(figsize=(6, 3))

        results = dnn_res[f'{alg}-gan_0_{metric}'] + dnn_res[f'{alg}-gan_1_{metric}']
        cdf, _ = np.histogram(results, bins=n_bins, density=True)
        cdf = np.cumsum(cdf)
        cdf = cdf.astype(float) / cdf[-1]
        plt.plot(n_bins[0:-1], cdf, '-', zorder=4, linewidth=2, label=f'{alg.upper()}-GAN-DNN')

        results = rnn_res[f'{alg}-gan_0_{metric}'] + rnn_res[f'{alg}-gan_1_{metric}']
        cdf, _ = np.histogram(results, bins=n_bins, density=True)
        cdf = np.cumsum(cdf)
        cdf = cdf.astype(float) / cdf[-1]
        plt.plot(n_bins[0:-1], cdf, '-.', zorder=4, linewidth=2, label=f'{alg.upper()}-GAN-RNN')

        results = cnn_res[f'{alg}-gan_0_{metric}'] + cnn_res[f'{alg}-gan_1_{metric}']
        cdf, _ = np.histogram(results, bins=n_bins, density=True)
        cdf = np.cumsum(cdf)
        cdf = cdf.astype(float) / cdf[-1]
        plt.plot(n_bins[0:-1], cdf, '--', zorder=4, linewidth=2, label=f'{alg.upper()}-GAN-CNN')

        results = tanogan_res[f'tanogan_0_{metric}'] + tanogan_res[f'tanogan_1_{metric}']
        cdf, _ = np.histogram(results, bins=n_bins, density=True)
        cdf = np.cumsum(cdf)
        cdf = cdf.astype(float) / cdf[-1]
        plt.plot(n_bins[0:-1], cdf, ':', zorder=4, linewidth=2, label='TAnoGAN')

        results = deepsmote_res[f'deepsmote_0_{metric}'] + deepsmote_res[f'deepsmote_1_{metric}']
        cdf, _ = np.histogram(results, bins=n_bins, density=True)
        cdf = np.cumsum(cdf)
        cdf = cdf.astype(float) / cdf[-1]
        plt.plot(n_bins[0:-1], cdf, ':', zorder=4, linewidth=2, label='DeepSMOTE')

        plt.grid(linestyle='--', linewidth=0.5, zorder=0)
        plt.ylim(0, 1.)
        plt.xlim(0, 70.0 if metric == 'fid' else 1.)
        plt.xlabel(f'{metric.upper()} on Dataset A', verticalalignment='top')
        plt.ylabel('Cumulative Distribution Function', verticalalignment='bottom')
        plt.legend()  # show legend
        plt.tight_layout()

        plt.savefig(f"benchmark_results_{metric}_{alg}.png", dpi=800, bbox_inches='tight')
        plt.savefig(f"benchmark_results_{metric}_{alg}.pdf", dpi=800, bbox_inches='tight')
