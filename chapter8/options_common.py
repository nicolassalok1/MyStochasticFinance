import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

from chapter5.base_forecasting import ForecastResult
import matplotlib.pyplot as plt
import seaborn as sns


@dataclass
class OptionsResult:

    value_at_0: float = None  # premium paid for call & put options

    all_values: ForecastResult = None  # simulation result

    underlying_s_values: np.ndarray = None

    label: str = None

    for_call_option: bool = True


class OptionsTemplate(ABC):

    '''
    Interface for all type of options - Vanilla, Asian, or Barrier
    '''

    @abstractmethod
    def estimate_call(self, expiry_time_T, strike_price_K,
                      **kwargs) -> tuple: ...
    '''
    Function to compute call option. It returns tuple of
    forecast results that may contain greeks.
    '''

    @abstractmethod
    def estimate_put(self, expiry_time_T, strike_price_K,
                     **kwargs) -> tuple: ...
    '''
    Function to compute put option. It returns tuple of
    forecast results that may contain greeks.
    '''


def plot_options_surface(options_result: OptionsResult):
    plt.style.use("seaborn-v0_8")
    fig = plt.figure(figsize=(10, 7))
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122)
    underlyings_values = options_result.underlying_s_values[:, 1:]
    rows, cols = underlyings_values.shape
    option_values = options_result.all_values.sample_paths.values.T

    sns.scatterplot(ax=ax2, x=underlyings_values.flatten(),
                    y=option_values.flatten())

    ts = [[t for t in range(cols)] for _ in range(rows)]
    if (options_result.for_call_option):
        ax1.plot_surface(
            ts,
            underlyings_values,
            option_values, rstride=5, cstride=5,
            cmap=plt.cm.gnuplot2,
            edgecolor="black")

        ax1.set_xlabel('t')
        ax1.set_ylabel('S')
    else:
        ax1.plot_surface(
            underlyings_values,
            ts,
            option_values, rstride=5, cstride=5,
            cmap=plt.cm.gnuplot2,
            edgecolor="black")

        ax1.set_xlabel('S')
        ax1.set_ylabel('t')

    ax1.set_zlabel(options_result.label)

    ax2.set_xlabel('S')
    ax2.set_ylabel(options_result.label)

    fig.tight_layout()
    plt.show()
