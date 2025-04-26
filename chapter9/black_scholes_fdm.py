import numpy as np
from chapter9.finite_difference_methods import HeatEquationImplicitFDMSolverTemplate
from chapter6.diffusion_model import DiffusionProcessAssetPriceModel
import matplotlib.pyplot as plt
import seaborn as sns


class BlackScholesPutOptionsFDMSolver(HeatEquationImplicitFDMSolverTemplate):

    """
     Class to solve Black Scholes PSDE for Put options with FDM.
     It is integrated with Diffusion Model to find the bounds on asset
     price S.
    """

    def __init__(self,
                 diffusion_asset_model: DiffusionProcessAssetPriceModel,
                 M,  # Number of time data points
                 N,  # Number of asset values data points
                 strike_price_K
                 ):
        self._strike_price_K = strike_price_K
        self._diffusion_asset_model = diffusion_asset_model
        self._σ = diffusion_asset_model.parameters_['σ']
        self._r = diffusion_asset_model.parameters_['r']
        self._M = M
        self._mean_path, self._s_min, self._s_max = self._extract_bounds()

        super().__init__(x_min=self._s_min,
                         x_max=self._s_max,
                         T=1.0,
                         M=M,
                         N=N,
                         func_name="Put V",
                         space_var_name='S',
                         time_var_name='t',
                         terminal_condition_ind=True)

    def _extract_bounds(self):
        mean_path = np.array(self._diffusion_asset_model.forecast(
            T=self._M).mean_path.values).flatten()
        return mean_path, np.min(mean_path), np.max(mean_path)

    def plot_asset_grid(self):
        plt.style.use("seaborn-v0_8")
        ax = sns.lineplot(y=self._mean_path, x=self._t)
        ax.hlines(y=self._x,
                  xmin=0, xmax=np.max(self._t))
        ax.vlines(ymin=self._s_min, ymax=self._s_max+10,
                  x=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        plt.show()

    def _a(self, i=None, j=None):
        return 0.5 * self._δt * ((self._r*(self._x[i]/self._δx)) - (self._σ / (self._x[i]/self._δx))**2)

    def _b(self, i=None, j=None):
        return 1 + self._δt * ((self._σ / (self._x[i]/self._δx))**2 + self._r)

    def _c(self, i=None, j=None):
        return -0.5 * self._δt * ((self._r*(self._x[i]/self._δx)) + (self._σ / (self._x[i]/self._δx))**2)

    def _initial_condition(self, x):
        return max(self._strike_price_K-x, 0)

    def _first_boundary_condition(self, t):
        return self._strike_price_K * np.exp(-self._r*(1.0-t)) - self._s_min

    def _second_boundary_condition(self, t):
        return 0

    @ property
    def premium_(self): return self._u[:, 0]
