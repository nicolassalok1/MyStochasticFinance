import matplotlib.pyplot as plt


def plot_all_mean_paths_for_bm(dis_r_1, dis_r_2, dis_r_3, dis_r_4):
    params = "(μ={0},σ={1})"
    _, ax = plt.subplots(nrows=2, ncols=2)

    dis_r_1[0].plot_mean_path(ax=ax[0, 0])
    ax[0, 0].legend([params.format(dis_r_1[1], dis_r_1[2])])

    dis_r_2[0].plot_mean_path(ax=ax[0, 1])
    ax[0, 1].legend([params.format(dis_r_2[1], dis_r_2[2])])

    dis_r_3[0].plot_mean_path(ax=ax[1, 0])
    ax[1, 0].legend([params.format(dis_r_3[1], dis_r_3[2])])

    dis_r_4[0].plot_mean_path(ax=ax[1, 1])
    ax[1, 1].legend([params.format(dis_r_4[1], dis_r_4[2])])

    plt.show()


def plot_all_sample_paths_for_bm(dis_r_1, dis_r_2, dis_r_3, dis_r_4):
    params = "(μ={0},σ={1})"
    _, ax = plt.subplots(nrows=2, ncols=2)

    dis_r_1[0].plot_sample_paths(ax=ax[0, 0])
    ax[0, 0].legend([params.format(dis_r_1[1], dis_r_1[2])])

    dis_r_2[0].plot_sample_paths(ax=ax[0, 1])
    ax[0, 1].legend([params.format(dis_r_2[1], dis_r_2[2])])

    dis_r_3[0].plot_sample_paths(ax=ax[1, 0])
    ax[1, 0].legend([params.format(dis_r_3[1], dis_r_3[2])])

    dis_r_4[0].plot_sample_paths(ax=ax[1, 1])
    ax[1, 1].legend([params.format(dis_r_4[1], dis_r_4[2])])

    plt.show()


def plot_all_sample_paths_for_pp(dis_r_1, dis_r_2, dis_r_3, dis_r_4):
    params = "(λ={0})"
    _, ax = plt.subplots(nrows=2, ncols=2)

    dis_r_1[0].plot_sample_paths(ax=ax[0, 0])
    ax[0, 0].legend([params.format(dis_r_1[1])])

    dis_r_2[0].plot_sample_paths(ax=ax[0, 1])
    ax[0, 1].legend([params.format(dis_r_2[1])])

    dis_r_3[0].plot_sample_paths(ax=ax[1, 0])
    ax[1, 0].legend([params.format(dis_r_3[1])])

    dis_r_4[0].plot_sample_paths(ax=ax[1, 1])
    ax[1, 1].legend([params.format(dis_r_4[1])])

    plt.show()
