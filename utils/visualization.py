import matplotlib.pyplot as plt
import numpy as np


def plot_raw(clean, adv, file_name, is_norm=False):
    if is_norm:
        max_, min_ = np.max(clean), np.min(clean)
        clean = (clean - min_) / (max_ - min_)
        adv = (adv - min_) / (max_ - min_)

    # chans = [48, 31, 10, 35, 52] # MI109
    chans = [14, 2, 9, 4, 16] # MI4C
    chans_name = ['P3', 'F3', 'Cz', 'F4', 'P4']
    x = np.arange(clean.shape[1]) * 1.0 / 128

    fontsize=10
    fig = plt.figure(figsize=(6, 3))

    ax1 = fig.add_subplot(1, 2, 1)
    l1, = ax1.plot(x, adv[chans[0]] - np.mean(adv[chans[0]]), linewidth=1.0, color='red', label='Unlearnable trial')  # plot adv data
    l2, = ax1.plot(x, clean[chans[0]] - np.mean(adv[chans[0]]), linewidth=1.0, color='dodgerblue',
                   label='Benign trial')  # plot clean data
    for i in range(1, 5):
        ax1.plot(x, adv[chans[i]] + i - np.mean(adv[chans[i]]), linewidth=1.0, color='red')  # plot adv data
        ax1.plot(x, clean[chans[i]] + i - np.mean(adv[chans[i]]), linewidth=1.0, color='dodgerblue')  # plot clean data
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylim([-0.5, 5.5])
    temp_y = np.arange(5)
    y_names = [f'{x}' for x in chans_name]
    plt.yticks(temp_y, y_names, fontsize=fontsize)
    plt.legend(handles=[l2, l1], labels=['Benign trial', 'Unlearnable trial'], loc='upper right', ncol=1,
               fontsize=fontsize-2)

    ax2 = fig.add_subplot(1, 2, 2)
    l1, = ax2.plot(x, (adv[chans[0]] - clean[chans[0]])*50, linewidth=1.0, color='red', label='Noise')
    for i in range(1, 5):
        ax2.plot(x, (adv[chans[i]] - clean[chans[i]])*50 + i, linewidth=1.0, color='red')
    plt.xlabel('Time (s)', fontsize=fontsize)
    plt.ylim([-0.5, 5.5])
    temp_y = np.arange(5)
    y_names = [f'{x}' for x in chans_name]
    plt.yticks(temp_y, y_names, fontsize=fontsize)
    plt.legend(handles=[l1], labels=['Noise (x50)'], loc='upper right', ncol=2,
               fontsize=fontsize-2)

    plt.subplots_adjust(wspace=1.0, hspace=0)
    plt.tight_layout()
    plt.savefig(file_name + '.png', dpi=300)