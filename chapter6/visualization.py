import matplotlib.pyplot as plt


def plot_gbm_parameter_famailiarization(dis_r_1, dis_r_2, dis_r_3, dis_r_4):
    params = "(r={0},σ={1})"
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


def plot_model_forecasting_results(forecast_display, ap_back_testing):
    _, ax = plt.subplots(nrows=1, ncols=3)
    forecast_display.plot_sample_paths(ax=ax[0])
    ax[0].legend(['All Sample Paths'])
    forecast_display.plot_uncertainity_bounds(ax=ax[1])
    ax[2].legend(['Backtesting Result'])
    ap_back_testing.plot_comparison(ax=ax[2])
    plt.show()
