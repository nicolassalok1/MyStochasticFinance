from array import array
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

plt.style.use('seaborn-v0_8')


def plot_returns_for_different_periods(ticker, periodic_returns: List[tuple]):
    fig, ax = plt.subplots(len(periodic_returns), 1)

    for index, t in enumerate(periodic_returns):

        t[1].plot(ax=ax[index], x='time', y='Return')
        ax[index].set_title(ticker + ' - ' + t[0] + ' Returns')

    fig.tight_layout()
    plt.show()


def plot_portfolios(portfolios: array):
    fig, ax = plt.subplots(len(portfolios), 1)
    for index, entry in enumerate(portfolios):
        ax[index].set_title(
            'Weights distrubtion of assets - ' + entry[0] + ' Returns Portfolio')
        ax[index].text(0.02, 0.98, 'Volatility Returns: ' + str(round(entry[1].optimal_variance, 4)),
                       ha="left", va="top", transform=ax[index].transAxes)
        record = pd.DataFrame([{'symbol': k, 'weight': entry[1].asset_allocation_distribution[k]}
                               for k in entry[1].asset_allocation_distribution])
        sns.barplot(ax=ax[index], data=record, x='symbol',
                    y='weight', width=0.1, dodge=False)

    fig.tight_layout()
    plt.show()


def plot_returns_for_different_assets(periodic_returns: dict):
    fig, ax = plt.subplots(len(periodic_returns), 1)

    for index, ticker in enumerate(periodic_returns):
        periodic_returns[ticker].plot(ax=ax[index], x='time', y='Return')
        ax[index].set_title(ticker + ' - Returns')

    fig.tight_layout()
    plt.show()


def plot_scatter(data, x_name, y_name, title):
    sns.scatterplot(data=data, x=x_name, y=y_name)
    plt.title(title)
    plt.show()


def plot_line(data, x_name, y_name, title):
    sns.lineplot(data=data, x=x_name, y=y_name)
    plt.title(title)
    plt.show()
