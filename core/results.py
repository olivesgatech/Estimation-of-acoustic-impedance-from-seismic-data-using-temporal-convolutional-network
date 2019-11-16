# This script generates results from the data stored in the results directory
import os
from os.path import join as pjoin

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import linregress


def evaluate(args):
    """Function makes plots on the results and also calculates pcc and r2 scores on both the validation and training
    data"""

    # Load data
    AI = np.load(pjoin('results', 'AI.npy'))
    AI_inv = np.load(pjoin('results', 'AI_inv.npy')) * AI.std() + AI.mean()

    # Get training and validation indices
    train_indices = np.linspace(0, 1946, args.n_wells).astype(int)
    val_indices = np.setdiff1d(np.arange(0, 1946).astype(int), train_indices)

    # Make AI, predicted AI plots
    make_plots(AI, AI_inv)

    # Make scatter plot
    scatter_plot(AI, AI_inv)

    # Make trace plot
    trace_plot(AI, AI_inv, val_indices)

    # Print r2 and PCC scores
    r2_pcc_scores(train_indices, val_indices, AI, AI_inv)


def plot(img, cmap='rainbow', cbar_label=r'AI ($m/s\times g/cm^3$)', vmin=None, vmax=None):
    """Makes seaborn style plots"""
    dt = 0.001165582791395698
    dx = 6.247687564234327
    Y, X = np.mgrid[slice(0.47, 2.8 + dt, dt), slice(2824, 14982 + dx, dx)]

    fig = plt.figure(figsize=(20, 6))
    ax = fig.add_subplot(1, 1, 1)
    if (vmin is None or vmax is None):
        plt.pcolormesh(X, Y, img.T, cmap=cmap)
    else:
        plt.pcolormesh(X, Y, img.T, cmap=cmap, vmin=vmin, vmax=vmax)

    cbar = plt.colorbar()
    plt.ylabel("Depth (Km)", fontsize=30)
    plt.xlabel("Distance (m)", fontsize=30, labelpad=15)
    ax.invert_yaxis()
    ax.xaxis.set_label_position('top')
    ax.xaxis.set_ticks_position("top")
    plt.gca().set_xticks(np.arange(2800, 15000 + 1, 1700 * 2))
    plt.tick_params(axis='both', which='major', labelsize=30)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label(cbar_label, rotation=270, fontsize=30, labelpad=40)
    return fig


def make_plots(AI, AI_inv):
    """Generate and save true and predicted AI plots"""
    #vmin = min([AI.min(), AI_inv.min()])
    vmin = AI.min()
    #vmax = max([AI.max(), AI_inv.max()])
    vmax = AI.max()
    fig = plot(AI, vmin=vmin, vmax=vmax)
    fig.savefig('AI.png', bbox_inches='tight')
    fig = plot(AI_inv, vmin=vmin, vmax=vmax)
    fig.savefig('AI_inv.png', bbox_inches='tight')
    fig = plot(abs(AI_inv - AI), vmin=vmin, vmax=vmax)
    fig.savefig('difference.png', bbox_inches='tight')


def scatter_plot(AI, AI_inv):
    """Generate scatter plot between true and predicted AI"""
    AI = np.expand_dims(AI, axis=1)
    AI_inv = np.expand_dims(AI_inv, axis=1)
    sns.set(style="whitegrid")
    fig = plt.figure()
    np.random.seed(30)
    inds = np.random.choice(AI.shape[0], 30)
    x = np.reshape(AI[inds, 0], -1)
    y = np.reshape(AI_inv[inds, 0], -1)

    std = AI[:, 0].std()

    max = np.max([AI[:, 0].max(), AI_inv[:, 0].max()])
    min = np.min([AI[:, 0].min(), AI_inv[:, 0].min()])

    d = {'True AI': x, 'Estimated AI': y}
    df = pd.DataFrame(data=d)

    fig = plt.figure(figsize=(15, 15))
    g = sns.jointplot("Estimated AI", "True AI", data=df, kind="reg",
                      xlim=(min, max), ylim=(min, max), color="k", scatter_kws={'s': 10}, label='big', stat_func=None)

    plt.xlabel(r"Estimated AI", fontsize=30)
    plt.ylabel(r"True AI", fontsize=30)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.savefig('Scatter.png', bbox_inches='tight')
    plt.show()


def trace_plot(AI, AI_inv, val_indices):
    """
    Generates a figure showing handpicked traces from true and predicted AI models superimposed on each other.

    Parameters
    ----------
    AI: numpy array
       The true AI model
    AI_inv: numpy array
           The predicted AI model
    """

    inds = val_indices[::500].astype(int)

    x = AI[inds].squeeze()
    y = AI_inv[inds].squeeze()
    time = np.linspace(0.47, 2.8, 2000)
    fig, ax = plt.subplots(1, x.shape[0], figsize=(10, 12), sharey=True)

    max = np.max([y.max(), x.max()]) * 1.2
    min = np.min([y.min(), x.min()]) * 0.8

    for i in range(len(inds)):
        p1 = ax[i].plot(x[i], time, 'k')
        p2 = ax[i].plot(y[i], time, 'r')
        ax[i].set_xlabel(r'AI($m/s \times g/cm^3$)' + '\n' + r'$trace={}m$'.format(inds[i]), fontsize=15)
        if i == 0:
            ax[i].set_ylabel('Depth (Km)', fontsize=20)
            ax[i].yaxis.set_tick_params(labelsize=20)

        ax[i].set_ylim(time[0], time[-1])
        ax[i].set_xlim(min, max)
        ax[i].invert_yaxis()
        ax[i].xaxis.set_tick_params(labelsize=10)

    fig.legend([p1[0], p2[0]], ["True AI", "Estimated AI for SVR"], loc="upper center", fontsize=20,
               bbox_to_anchor=(0.5, 1.07))
    plt.show()
    fig.savefig('AI_traces_svr.png'.format(inds), bbox_inches='tight')


def r2_pcc_scores(train_indices, val_indices, AI, AI_inv):
    """Function computes and prints the r2 and pcc on both the training and validation sets"""
    pcc_train = 0
    r2_train = 0
    for i in range(len(train_indices)):
        trace_pred = AI_inv[train_indices[i]]
        trace_actual = AI[train_indices[i]]
        pcc_train += np.corrcoef(trace_actual, trace_pred)[0, 1]
        slope, intercept, r_value, p_value, std_err = linregress(trace_actual, trace_pred)
        r2_train += r_value ** 2
    pcc_train = pcc_train / len(train_indices)
    r2_train = r2_train / len(train_indices)

    pcc_val = 0
    r2_val = 0
    for i in range(len(val_indices)):
        trace_pred = AI_inv[val_indices[i]]
        trace_actual = AI[val_indices[i]]
        pcc_val += np.corrcoef(trace_actual, trace_pred)[0, 1]
        slope, intercept, r_value, p_value, std_err = linregress(trace_actual, trace_pred)
        r2_val += r_value ** 2
    pcc_val = pcc_val / len(val_indices)
    r2_val = r2_val / len(val_indices)

    print('r2_training: {:0.4f} | r2_validation: {:0.4f} | pcc_training: {:0.4f} | pcc_val: {:0.4f}'.format(r2_train,
                                                                                                            r2_val,
                                                                                                            pcc_train,
                                                                                                            pcc_val))





