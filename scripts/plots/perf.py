import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.io import loadmat

dnn_res = 'saved/dnn/eval.mat'
rnn_res = 'saved/rnn/eval.mat'
cnn_res = 'saved/cnn/eval.mat'

# baselines
tanogan_res = 'backups/tanogan/eval.mat'
deepsmote_res = 'backups/deepsmote/eval.mat'

dnn_res = loadmat(dnn_res, squeeze_me=True)
rnn_res = loadmat(rnn_res, squeeze_me=True)
cnn_res = loadmat(cnn_res, squeeze_me=True)

tanogan_res = loadmat(tanogan_res, squeeze_me=True)
deepsmote_res = loadmat(deepsmote_res, squeeze_me=True)

x_axis = [.1, .2, .3, .5, .7, 1., 1.5, 2., 3., 5.]
baseline = dnn_res['pre-aug-B'][0]
ec_gan_dnn_acc = [dnn_res[f'ec-gan_B_{int(ratio*100)}'][0] for ratio in x_axis]
co_gan_dnn_acc = [dnn_res[f'co-gan_B_{int(ratio*100)}'][0] for ratio in x_axis]

ec_gan_rnn_acc = [rnn_res[f'ec-gan_B_{int(ratio*100)}'][0] for ratio in x_axis]
co_gan_rnn_acc = [rnn_res[f'co-gan_B_{int(ratio*100)}'][0] for ratio in x_axis]

ec_gan_cnn_acc = [cnn_res[f'ec-gan_B_{int(ratio*100)}'][0] for ratio in x_axis]
co_gan_cnn_acc = [cnn_res[f'co-gan_B_{int(ratio*100)}'][0] for ratio in x_axis]

tanogan_acc = [tanogan_res[f'tanogan_B_{int(ratio*100)}'][0] for ratio in x_axis]
deepsmote_acc = [deepsmote_res[f'deepsmote_B_{int(ratio*100)}'][0] for ratio in x_axis]

x_label = [f'+{int(ratio*100)}%' for ratio in x_axis]

## EC-GAN
plt.figure(figsize=(10, 6))  # Adjust figure size

plt.plot(x_label, ec_gan_dnn_acc, marker='o', label='EC-GAN-DNN')
plt.plot(x_label, ec_gan_rnn_acc, marker='x', label='EC-GAN-RNN')
plt.plot(x_label, ec_gan_cnn_acc, marker='*', label='EC-GAN-CNN')

plt.plot(x_label, tanogan_acc, marker='x', linestyle='--', label='TAnoGAN')
plt.plot(x_label, deepsmote_acc, marker='o', linestyle='--', label='DeepSMOTE')

plt.grid(linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='Pre-augmentation')
plt.ylim(0.7, 1.0)
# plt.xlim(-.1, 5.1)
# plt.xticks(x_axis, x_label)
plt.xlabel("Augmented Data Volume Ratio")  # x-axis label
plt.ylabel("Accuracy on Dataset B")    # y-axis label
plt.title("Benchmark Comparison of Different Methods")  # title
plt.legend()  # show legend

plt.savefig("benchmark_results_acc_ec.png", dpi=800, bbox_inches='tight')
plt.savefig("benchmark_results_acc_ec.pdf", dpi=800, bbox_inches='tight')

## CO-GAN
plt.figure(figsize=(10, 6))  # Adjust figure size

plt.plot(x_label, co_gan_dnn_acc, marker='o', label='CO-GAN-DNN')
plt.plot(x_label, co_gan_rnn_acc, marker='x', label='CO-GAN-RNN')
plt.plot(x_label, co_gan_cnn_acc, marker='*', label='CO-GAN-CNN')

plt.plot(x_label, tanogan_acc, marker='x', linestyle='--', label='TAnoGAN')
plt.plot(x_label, deepsmote_acc, marker='o', linestyle='--', label='DeepSMOTE')

plt.grid(linestyle='--', linewidth=0.5, zorder=0)
plt.axhline(y=baseline, color='red', linestyle='--', linewidth=2, label='Pre-augmentation')
plt.ylim(0.7, 1.0)
# plt.xlim(-.1, 5.1)
# plt.xticks(x_axis, x_label)
plt.xlabel("Augmented Data Volume Ratio")  # x-axis label
plt.ylabel("Accuracy on Dataset B")    # y-axis label
plt.title("Benchmark Comparison of Different Methods")  # title
plt.legend()  # show legend

plt.savefig("benchmark_results_acc_co.png", dpi=800, bbox_inches='tight')
plt.savefig("benchmark_results_acc_co.pdf", dpi=800, bbox_inches='tight')
