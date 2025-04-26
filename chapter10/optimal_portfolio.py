from chapter10.portfolio_assets import PortfolioAssets
from typing import List
from abc import ABC, abstractmethod, ABCMeta


class MinVariancePortfolioOptimizer(metaclass=ABCMeta):
    """
    Meta-class for minimum variance portfolio optimizer 
    strategy. It should be implemented fully by the any 
    specific algorithm of portfolio optimization
    """
    @abstractmethod
    def fit(portfolio_assets: PortfolioAssets): ...
    """
     Runs the portfolio optimization algorithm or any 
     formulae provided by this specific optimizer.  
     Optimizer should set the _weights variable accordingly 
     as output.

     Parameters
     ----------
     portfolio_assets: Instance of meta-class PortfolioAssets
    """

    @property
    @abstractmethod
    def asset_allocation_distribution(self): ...
    """
    Returns the dictionary of asset names & corresponding weights
    """

    @property
    @abstractmethod
    def optimal_variance(self): ...
    """
      Returns the minimum variance of portfolio as 
      determined by the optimization algorithm or formulae
    """


class BaseMinVariancePortfolioOptimizer(MinVariancePortfolioOptimizer, ABC):

    def __init__(self, expected_mean_return=None):
        self._expected_mean_return = expected_mean_return
        self._weights = None
        self._asset_allocation = {}
        self._optimal_var = None

    @property
    def asset_allocation_distribution(self):
        return self._asset_allocation

    @property
    def optimal_variance(self):
        return self._optimal_var

    def _prepare_asset_allocation(self, asset_names: List[str]):
        self._asset_allocation = {}
        for index, asset in enumerate(asset_names):
            self._asset_allocation[asset] = self._weights[index]
