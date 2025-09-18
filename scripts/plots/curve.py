import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np

target_model = 'backups/Thu_Sep_18_12:02:02_2025_DNN'
# target_model = 'backups/Thu_Sep_18_12:15:25_2025_DNN'
# target_model = 'backups/Thu_Sep_18_12:02:39_2025_RNN'

def curve_plot(traces, filename):
    epochs = traces['epochs']
    loss_d = traces['loss_d']
    loss_g = traces['loss_g']
    sample_g = traces['sample_g']

    fig, ax1 = plt.subplots(figsize=(6, 3))

    # Plot generator loss
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Generator Loss", color="red")
    sns.lineplot(x=epochs, y=loss_g, color="red", label="Generator", ax=ax1)
    # ax1.plot(epochs, loss_g, color="red", label="Generator")
    ax1.tick_params(axis="y", labelcolor="red")
    ax1.set_ylim(0, 1.35)

    # Plot discriminator loss on right axis
    ax2 = ax1.twinx()
    ax2.set_ylabel("Discriminator Loss", color="blue")
    sns.lineplot(x=epochs, y=loss_d, color="blue", label="Discriminator", ax=ax2)
    # ax2.plot(epochs, loss_d, color="blue", label="Discriminator")
    ax2.tick_params(axis="y", labelcolor="blue")
    ax2.set_ylim(.3, 1.3)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(handles1 + handles2, labels1 + labels2, loc="center right")
    if ax2.get_legend() is not None:
        ax2.get_legend().remove()

    for ep in [1, 1000, 2000, 3000, 4000, 5000]:
        i = np.where(epochs == ep)
        sample = sample_g[i][0][:15]

        x_disp, y_disp = ax1.transData.transform((ep, ax1.get_ylim()[0]))  
        inv = fig.transFigure.inverted()
        x_fig, y_fig = inv.transform((x_disp, y_disp))

        ax_inset = fig.add_axes([x_fig-0.05, .92, 0.11, 0.2])

        img = np.vstack(sample)
        ax_inset.imshow(img, cmap="seismic", aspect="auto")
        ax_inset.set_title(f"epoch={ep}", fontsize=8)
        ax_inset.axis("off")

    fig.savefig(f'{filename}.png', dpi=800, bbox_inches='tight')
    fig.savefig(f'{filename}.pdf', dpi=800, bbox_inches='tight')

traces_ec = loadmat(f'{target_model}/traces_ec.mat', squeeze_me=True)
traces_co = loadmat(f'{target_model}/traces_co.mat', squeeze_me=True)

curve_plot(traces_ec, 'traces_ec')
curve_plot(traces_co, 'traces_co')
