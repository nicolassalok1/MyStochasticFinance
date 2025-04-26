import matplotlib.pyplot as plt


def plot_kde_with_hs(dens_1, dens_2, dens_3, dens_4):
    plt.style.use("seaborn-v0_8")
    _, ax = plt.subplots(nrows=2, ncols=2)

    ax[0, 0].plot(dens_1[0], dens_1[1])
    ax[0, 0].legend(['h=' + str(dens_1[2])])

    ax[0, 1].plot(dens_2[0], dens_2[1])
    ax[0, 1].legend(['h=' + str(dens_2[2])])

    ax[1, 0].plot(dens_3[0], dens_3[1])
    ax[1, 0].legend(['h=' + str(dens_3[2])])

    ax[1, 1].plot(dens_4[0], dens_4[1])
    ax[1, 1].legend(['h=' + str(dens_4[2])])

    plt.show()


def plot_merton_process_paths_for_single_set_params(forecast_display):
    _, ax = plt.subplots(nrows=1, ncols=3)
    forecast_display.plot_sample_paths(ax=ax[0])
    ax[0].legend(['All Sample Paths'])
    forecast_display.plot_mean_path(ax=ax[1])
    ax[1].legend(['Mean Path'])
    forecast_display.plot_uncertainity_bounds(ax=ax[2])
    plt.show()


def plot_kou_process_paths_for_single_set_params(forecast_display):
    _, ax = plt.subplots(nrows=1, ncols=3)
    forecast_display.plot_sample_paths(ax=ax[0])
    ax[0].legend(['All Sample Paths'])
    forecast_display.plot_mean_path(ax=ax[1])
    ax[1].legend(['Mean Path'])
    forecast_display.plot_uncertainity_bounds(ax=ax[2])
    plt.show()


def plot_model_forecasting_results(forecast_display, ap_back_testing):
    _, ax = plt.subplots(nrows=1, ncols=3)
    forecast_display.plot_sample_paths(ax=ax[0])
    ax[0].legend(['All Sample Paths'])
    forecast_display.plot_uncertainity_bounds(ax=ax[1])
    ax[2].legend(['Forecast Comparison'])
    ap_back_testing.plot_comparison(ax=ax[2])
    plt.show()


def plot_merton_process_parameter_famailiarization(dis_r_1, dis_r_2, dis_r_3, dis_r_4):
    params = "(r={0}, \nσ={1}, \nλ={2}, \nμ_j={3}, \nσ_j={4})"
    _, ax = plt.subplots(nrows=2, ncols=2)

    dis_r_1[0].plot_mean_path(ax=ax[0, 0])
    ax[0, 0].legend(
        [params.format(dis_r_1[1], dis_r_1[2], dis_r_1[3], dis_r_1[4], dis_r_1[5])])

    dis_r_2[0].plot_mean_path(ax=ax[0, 1])
    ax[0, 1].legend(
        [params.format(dis_r_2[1], dis_r_2[2], dis_r_2[3], dis_r_2[4], dis_r_2[5])])

    dis_r_3[0].plot_mean_path(ax=ax[1, 0])
    ax[1, 0].legend(
        [params.format(dis_r_3[1], dis_r_3[2], dis_r_3[3], dis_r_3[4], dis_r_3[5])])

    dis_r_4[0].plot_mean_path(ax=ax[1, 1])
    ax[1, 1].legend(
        [params.format(dis_r_4[1], dis_r_4[2], dis_r_4[3], dis_r_4[4], dis_r_4[5])])

    plt.show()


def plot_kou_process_parameter_famailiarization(dis_r_1, dis_r_2):
    params = "(r={0}, \nσ={1}, \nλ={2}, \np={3}, \nα_1={4}, \nα_2={5})"
    _, ax = plt.subplots(nrows=1, ncols=2)

    dis_r_1[0].plot_mean_path(ax=ax[0])
    ax[0].legend(
        [params.format(dis_r_1[1], dis_r_1[2], dis_r_1[3], dis_r_1[4], dis_r_1[5], dis_r_1[6])])

    dis_r_2[0].plot_mean_path(ax=ax[1])
    ax[1].legend(
        [params.format(dis_r_2[1], dis_r_2[2], dis_r_2[3], dis_r_2[4], dis_r_2[5], dis_r_2[6])])

    plt.show()


def plot_gaussian_kernel_process_parameter_famailiarization(dis_r_1, dis_r_2, dis_r_3, dis_r_4):
    params = "(h={0},σ_2_J={1})"
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


def plot_back_testing_results(bkt_1, bkt_2, bkt_3, bkt_4):
    params = "(h={0},σ_2_J={1})"
    _, ax = plt.subplots(nrows=2, ncols=2)

    bkt_1[0].plot_comparison(ax=ax[0, 0])
    ax[0, 0].set_title(params.format(bkt_1[1], bkt_1[2]))

    bkt_2[0].plot_comparison(ax=ax[0, 1])
    ax[0, 1].set_title(params.format(bkt_2[1], bkt_2[2]))

    bkt_3[0].plot_comparison(ax=ax[1, 0])
    ax[1, 0].set_title(params.format(bkt_3[1], bkt_3[2]))

    bkt_4[0].plot_comparison(ax=ax[1, 1])
    ax[1, 1].set_title(params.format(bkt_4[1], bkt_4[2]))

    plt.show()


def plot_full_testing_results(forecast_result_disp, back_testing, h, σ_2_J):
    params = "(h={0},σ_2_J={1})".format(h, σ_2_J)
    fig, ax = plt.subplots(nrows=2, ncols=2)

    forecast_result_disp.plot_sample_paths(ax=ax[0, 0])
    forecast_result_disp.plot_mean_path(ax=ax[0, 1])
    forecast_result_disp.plot_uncertainity_bounds(ax=ax[1, 0])
    back_testing.plot_comparison(ax=ax[1, 1])

    fig.suptitle(params, fontsize=8)
    plt.show()
