import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def poisson_plot(poisson_lamda_x_prob):
    plt.style.use("seaborn-v0_8")
    lamdas = list(poisson_lamda_x_prob.keys())
    n = len(lamdas)
    row = n / 2
    fig, ax = plt.subplots(row, 2)
    i = 0

    def _axis_plot_lamda():
        ax[i].set_title("λ = " + lamdas[i])
        plt.vlines(ax=ax[i], x=record[0], ymin=0, ymax=record[1])

    while i < n:
        record = poisson_lamda_x_prob[lamdas[i]]
        _axis_plot_lamda()
        i = i + 1
        record = poisson_lamda_x_prob[lamdas[i]]
        _axis_plot_lamda()
        i = i + 1

    fig.tight_layout()
    plt.show()


def plot_density_comparison_for_rvs(x_1, x_2, density_name=""):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(1, 2)
    sns.histplot(ax=ax[0], x=x_1, stat="density", kde=True)
    sns.histplot(ax=ax[1], x=x_2, stat="density", kde=True)
    ax[0].set_title("Inverse method")
    ax[1].set_title("Original density")
    fig.suptitle(density_name, fontsize=15)
    plt.show()


def plot_target_vs_proposal(rvs_f, probs_f, rvs_g, probs_g):
    plt.style.use("seaborn-v0_8")
    target = pd.DataFrame(
        {
            "Sample": rvs_f,
            "Probability": probs_f,
            "Type": np.repeat("Target Density", len(rvs_f)),
        }
    )
    proposal = pd.DataFrame(
        {
            "Sample": rvs_g,
            "Probability": probs_g,
            "Type": np.repeat("Proposal Density", len(rvs_g)),
        }
    )
    sns.lineplot(
        data=pd.concat([target, proposal]), x="Sample", y="Probability", hue="Type"
    )
    plt.show()


def plot_jump_diffusion_process_path(path_details):
    plt.style.use("seaborn-v0_8")
    n_fig = len(path_details)
    rows = int(n_fig / 2)
    fig, ax = plt.subplots(nrows=rows, ncols=2)
    i = 0
    k = 0
    title = "Drift: {drift}, Diffusion: {diffusion}, Jump(Rate: {jump_rate},Mean: {jump_mean}, Deviation: {jump_deviation})"
    while i < rows:
        path_details[k][5].plot(ax=ax[i, 0], x="time", y="state")
        f_title = str(k + 1) + ") " + title
        ax[i, 0].set_title(
            f_title.format(
                drift=str(np.round(path_details[k][0])),
                diffusion=str(np.round(path_details[k][1])),
                jump_rate=str(path_details[k][2]),
                jump_mean=str(path_details[k][3]),
                jump_deviation=str(path_details[k][4]),
            ),
            fontsize=8,
        )

        path_details[k + 1][5].plot(ax=ax[i, 1], x="time", y="state")
        f_title = str(k + 2) + ") " + title
        ax[i, 1].set_title(
            f_title.format(
                drift=str(np.round(path_details[k + 1][0])),
                diffusion=str(np.round(path_details[k + 1][1])),
                jump_rate=str(path_details[k + 1][2]),
                jump_mean=str(path_details[k + 1][3]),
                jump_deviation=str(path_details[k + 1][4]),
            ),
            fontsize=8,
        )
        i = i + 1
        k = k + 2

    fig.tight_layout()
    plt.show()


def plot_density_and_process(path, density_scores):
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].set_title("Sample path")
    path.plot(ax=ax[0], x="time", y="state")
    ax[1].set_title("Estimated Probability Density")
    ax[1].scatter(path["state"], density_scores, marker="x")
    fig.tight_layout()
    plt.show()
