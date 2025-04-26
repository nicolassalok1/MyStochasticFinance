from chapter10.portfolio_assets import PortfolioAssets
import numpy as np

from qpsolvers import solve_qp


from chapter10.optimal_portfolio import BaseMinVariancePortfolioOptimizer


class MarkowitzMinVariancePortfolioOptimizer(BaseMinVariancePortfolioOptimizer):

    def __init__(self, expected_mean_return=None):
        super().__init__(expected_mean_return)

    def fit(self, portfolio_assets: PortfolioAssets):
        means = portfolio_assets.unweighted_mean_returns
        cov = portfolio_assets.covariance_of_returns

        self._optimize_portfolio(means, cov, portfolio_assets.ticker_symbols)

    def _optimize_portfolio(self, means, cov, asset_names):
        """
         It computes weight distribution by the formulae,
            w = (rA-B)/D C^(-1) μ +  (M-rB)/D C^(-1) 1_N
        """
        n_assets = len(asset_names)
        cov_inverse = np.linalg.inv(cov)  # C^(-1)
        u_N = np.ones(n_assets)  # 1_N
        C_inv_mu = np.dot(cov_inverse, means)  # C^(-1) μ
        C_inv_1_N = np.dot(cov_inverse, u_N)  # C^(-1) 1_N

        A = np.dot(u_N.T, C_inv_1_N)  # 〖1_N〗^T C^(-1) 1_N
        B = np.dot(means.T, C_inv_1_N)  # μ^T C^(-1) 1_N
        M = np.dot(means.T, C_inv_mu)  # μ^T C^(-1) μ
        D = A * M - (B ** 2)

        factor_1 = ((self._expected_mean_return * A) - B) / D  # (rA-B)/D
        factor_2 = (M - (self._expected_mean_return * B)) / D  # (M-rB)/D

        self._weights = (factor_1 * C_inv_mu) + (factor_2 * C_inv_1_N)

        if self._weights is not None:
            self._prepare_asset_allocation(asset_names)
            self._optimal_var = np.dot(
                self._weights.T, np.dot(cov, self._weights))


class ExtendedMarkowitzMinVariancePortfolioOptimizer(MarkowitzMinVariancePortfolioOptimizer):

    def __init__(self, expected_mean_return=None):
        super().__init__(expected_mean_return)

    def _optimize_portfolio(self, means, cov, asset_names):
        n_assets = len(asset_names)

        # P of x^T Px. In this case wTCw
        P = np.array(2.0 * cov)

        #  q of q^Tx. q should be zero vector.
        q = np.zeros(n_assets)

        # A of Ax = b. Two constraints wT μ = r & w = 1 are combined here
        A = np.array([means, [1] * n_assets])

        # b of Ax = b. Two constraints wT μ = r & w = 1 are combined here
        b = np.array([self._expected_mean_return, 1.0])

        # G of Gx ≤ h. It is an identity matrix. It is for w > 0
        G = np.identity(n_assets) * (-1)

        # h of Gx ≤ h
        h = np.zeros(n_assets)

        self._weights = solve_qp(P=P, q=q, A=A, b=b, G=G, h=h, solver="osqp")

        if self._weights is not None:  # If optimal solution found with given constraints
            self._prepare_asset_allocation(asset_names)
            self._optimal_var = np.dot(
                self._weights.T, np.dot(cov, self._weights))
