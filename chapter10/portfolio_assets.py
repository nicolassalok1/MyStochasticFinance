from chapter2.stock_price_dataset_adapters import YahooFinancialsAdapter, Frequency, MarketStackAdapter
from abc import ABC, abstractmethod, ABCMeta
from typing import List
import pandas as pd
import numpy as np


class PortfolioAssets(metaclass=ABCMeta):

    """
     An abstract meta-class (interface) for a portfolio 
     holding a collection of assets. This should be extended 
     to support different dataset adapters

    """
    @property
    @abstractmethod
    def ticker_symbols(self): ...
    """ List of stock ticker symbols 
         in this portfolio  """

    @property
    @abstractmethod
    def weights(self): ...
    """ Assset weights """

    @weights.setter
    @abstractmethod
    def weights(self, w): ...

    @property
    @abstractmethod
    def expected_return(self): ...
    """
     Expected return of portfolio with adjusted weights
     """

    @property
    @abstractmethod
    def volatility(self): ...
    """Volatility of returns of the portfolio with adjusted weights"""

    @property
    @abstractmethod
    def unweighted_mean_returns(self): ...
    """Mean of returns of all assets in
         the portfolio"""

    @property
    @abstractmethod
    def covariance_of_returns(self): ...
    """Covariance of returns of all assets in
         the portfolio"""

    @property
    @abstractmethod
    def periodic_returns_for_different_assets(self): ...
    """ Periodic returns as per the frequency set in dataset adapter """


class BasePortfolioAssets(PortfolioAssets, ABC):

    def __init__(self, tickers: List[str], date_range: tuple, frequency: Frequency = None,
                 weights=None):
        self._frequency = frequency
        self._tickers = tickers
        self._date_range = date_range
        self._adapter = {}
        self._weights = np.array(weights)
        self._compute_unweighted_returns()

    @property
    def weights(self):
        return self._weights.tolist()

    @weights.setter
    def weights(self, w):
        self._weights = np.array(w)

    @property
    def ticker_symbols(self):
        return self._tickers

    @property
    def expected_return(self):
        """
        Expected return of portfolio with adjusted weights
        """
        # Applies formulae wT𝝁 and returns a float
        return np.dot(self._weights.T, self._returns.mean())

    @property
    def volatility(self):
        """Volatility of returns of the portfolio with adjusted weights"""
        # Applies formulae wTCw and returns a float
        return np.dot(self._weights.T,
                      np.dot(self._returns.cov(), self._weights))

    @property
    def covariance_of_returns(self):
        """Covariance of returns of all assets in
            the portfolio"""
        # Returns a dataframe with columns as ticker symbols
        return self._returns.cov()

    @property
    def unweighted_mean_returns(self):
        # Returns a dataframe with columns as ticker symbols
        return self._returns.mean()

    @property
    def periodic_returns_for_different_assets(self):
        return self._periodic_returns

    @property
    @abstractmethod
    def _asset_adapter(self): ...

    def _compute_unweighted_returns(self):
        """
        Computes simple returns of all assets. 

        Returns a dataframe having returns in rows and 
        asset names in columns.

        """
        self._returns = pd.DataFrame()
        self._periodic_returns = {}
        updated_tickers = []
        for ts in self._tickers:
            s = self._asset_adapter[ts].training_set
            if s is not None:

                # Applying R_t = (S_t-S_(t-τ))/S_(t-τ)
                self._returns[ts] = s['stock price'] / \
                    s['stock price'].shift(1) - 1

                self._periodic_returns[ts] = s
                self._periodic_returns[ts]['Return'] = self._returns[ts]
                updated_tickers.append(ts)

        self._tickers = updated_tickers


class YahooFinancialsPortfolioAssets(BasePortfolioAssets):

    """
    Asssets are fetched from Yahoo Data source adapter with a
    provided frequency, list of ticker symbols and date range.

    Parameters
   ----------
    frequency: enum,
               Frequency of stock values. Should be DAILY, WEEKLY or MONTHLY
    tickers: List of strings,
               Stock symbols of assets in the portfolio
    date_range: tuple of strings
               Date range for which stock values to be considered

    """

    def __init__(self, frequency: Frequency, tickers: List[str], date_range: tuple):
        super().__init__(frequency=frequency, tickers=tickers, date_range=date_range)

    @property
    def _asset_adapter(self):
        if len(self._adapter) == 0:
            for ticker in self._tickers:
                self._adapter[ticker] = YahooFinancialsAdapter(ticker=ticker,
                                                               frequency=self._frequency,
                                                               training_set_date_range=self._date_range)
        return self._adapter


class MarketStackPortfolioAssets(BasePortfolioAssets):

    def __init__(self, date_range: tuple, tickers: List[str] = None):
        self._adapter = None
        self._tickers = None
        self._date_range = date_range
        if tickers is None:
            tickers = MarketStackAdapter.get_samples_of_available_tickers()[
                :100]
        super().__init__(tickers=tickers, date_range=date_range)

    @property
    def _asset_adapter(self):
        if len(self._adapter) == 0:
            for ticker in self._tickers:
                self._adapter[ticker] = MarketStackAdapter(ticker=ticker,
                                                           training_set_date_range=self._date_range)
        return self._adapter
