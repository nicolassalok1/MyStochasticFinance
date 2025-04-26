
from numpy import exp
import pandas as pd
from chapter4.monte_carlo_simulation import MonteCarloSimulation, AntitheticSampling
from chapter4.gaussian_mc_simulation import GaussianImportanceSampling, StandardNormalTargetSamplingDensity
import matplotlib.pyplot as plt
import seaborn as sns


def test_gaussian_mc_simulation_no_var_reduction():
    mcs = MonteCarloSimulation(h_x_fun=lambda x: exp(-x),
                               n_vars=4, n_sample_paths=7,
                               target_sampling_density=StandardNormalTargetSamplingDensity())
    e = mcs.new_estimate()
    print(e.mean)
    print(e.standard_error)
    '''
    samples = pd.DataFrame(e.samples, columns=[
                           'X(t)', 'X(t+1)', 'X(t+2)', 'X(t+3)'])
                           '''
    samples = pd.DataFrame([e.mean], columns=[
                           1, 2, 3, 4])
    print(samples)
    print(samples.transpose())
    # samples.transpose().plot(xlabel='t', ylabel='X(t)')
    uncern_b_1 = e.mean + e.standard_error
    uncern_b_2 = e.mean - e.standard_error
    print(uncern_b_1)
    print(uncern_b_2)
    plt.show()


def test_gaussian_mc_simulation_imortance_sampling():
    mcs = MonteCarloSimulation(h_x_fun=lambda x: exp(-x),
                               n_vars=2, n_sample_paths=500,
                               var_reduction=GaussianImportanceSampling(),
                               target_sampling_density=StandardNormalTargetSamplingDensity())
    e = mcs.new_estimate()
    print(e.mean)
    print(e.standard_error)
    print(e.samples)


def test_gaussian_mc_simulation_antithetic_sampling():
    mcs = MonteCarloSimulation(h_x_fun=lambda x: exp(-x),
                               n_vars=10,
                               var_reduction=AntitheticSampling(),
                               target_sampling_density=StandardNormalTargetSamplingDensity())
#    debugpy.breakpoint()
    e = mcs.new_estimate()
    print(e.mean)
    print(e.standard_error)
    print(e.samples)


# test_gaussian_mc_simulation_imortance_sampling()
test_gaussian_mc_simulation_no_var_reduction()
# test_gaussian_mc_simulation_antithetic_sampling()
