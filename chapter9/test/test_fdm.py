from chapter9.finite_difference_methods import HeatEquationExplicitFDMSolver, HeatEquationImplicitFDMSolver, HeatEquationCrankNicolsonFDMSolver
from chapter9.black_scholes_fdm import BlackScholesPutOptionsFDMSolver
from dateparser import parse
from chapter6.diffusion_model import DiffusionProcessAssetPriceModel, IndexedTimeTransformer
from chapter2.stock_price_dataset_adapters import Frequency


def test_heat_equation_explicit_fdm_solver():
    HeatEquationExplicitFDMSolver(x_min=0,
                                  x_max=1, T=0.2,
                                  M=1000,
                                  N=10).solve().plot_solution()


def test_heat_equation_implicit_fdm_solver():
    HeatEquationImplicitFDMSolver(x_min=0,
                                  x_max=1, T=0.2,
                                  M=1000,
                                  N=10, terminal_condition_ind=True).solve().plot_solution()


def test_heat_equation_crank_nicolson_fdm_solver():
    HeatEquationCrankNicolsonFDMSolver(x_min=0,
                                       x_max=1, T=0.2,
                                       M=1000,
                                       N=10, terminal_condition_ind=True).solve().plot_solution()


def test_BS_fdm_solver():
    params = {'s0': 2058.89990234375,
              'r': 0.002476615753153449, 'σ': 0.02042995488960794,
              't0': parse('2015-01-01', date_formats=['YYYY-mm-dd'])}

    model = DiffusionProcessAssetPriceModel.load(parameters=params,
                                                 time_unit_transformer=IndexedTimeTransformer(
                                                     time_freq=Frequency.WEEKLY),
                                                 n_sample_paths=5)

    bs_put_solver = BlackScholesPutOptionsFDMSolver(diffusion_asset_model=model,
                                                    M=250,
                                                    N=10,
                                                    strike_price_K=4000)
    bs_put_solver.solve()
    bs_put_solver.plot_asset_grid()
    print(bs_put_solver.premium_)


test_BS_fdm_solver()
