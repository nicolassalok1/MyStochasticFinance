from chapter4.random_number_gen import (
    inverse_transform_method_rvs,
    generate_poisson_rv)

from chapter4.random_number_gen_accept_reject import DefaultSupremumEstimator

from chapter4.accept_reject_method_densities import (
    GaussianVariateGeneratorWithExponential,
    BetaVariateGeneratorWithGaussian,
    CosineVariateGeneratorWithUniform)

from numpy import pi
import chapter4.visualization as vs


from scipy.stats import expon
from numpy import log


def test_inverse_transform_method_exponen():
    lamda = 2.0

    # Random variables from the original Exponential distribution
    xs = expon.rvs(scale=1 / lamda, size=1000)

    # Inverse of the Expoential distrubition function
    def F_inverse_exponen(p): return -(log(1 - p)) / lamda

    # Generated random variables from inverse method
    xinvs = inverse_transform_method_rvs(F_inverse=F_inverse_exponen)

    # Compare and plot the densities from two generated random variable sets
    vs.plot_density_comparison_for_rvs(
        x_1=xs, x_2=xinvs, density_name="Exponential Density"
    )


def test_generate_poisson_rv():
    print(generate_poisson_rv(5, 10))


def test_acceptance_rejection_gaussian_gen_with_exponential():
    # Estimator designed for Exponential density used as proposal desnity g
    exponential_supremum_estimator = DefaultSupremumEstimator(
        x0=[0.01], x0_bounds=[(0.01, None)], θ0=[0.01], θ0_bounds=[(0.001, None)])
    _, sample_trace = GaussianVariateGeneratorWithExponential(μ=10, σ=4,
                                                              hat_func_optimizer=exponential_supremum_estimator).sample(n_rv=1000)
    sample_trace.plot()


def test_acceptance_rejection_beta_gen_with_gaussian():
    # Estimator designed for Gaussian density used as proposal desnity g
    gaussian_supremum_estimator = DefaultSupremumEstimator(
        x0=[0.00001], x0_bounds=[(0.00001, None)], θ0=[0.00001, 0.00001], θ0_bounds=[(0.00001, None), (0.00001, None)])
    _, sample_trace = BetaVariateGeneratorWithGaussian(a=5, b=1,
                                                       hat_func_optimizer=gaussian_supremum_estimator).sample(n_rv=1000)
    sample_trace.plot()


def test_acceptance_rejection_cosine_gen_with_uniform():
    # Estimator designed for Uniform density used as proposal desnity g
    uniform_supremum_estimator = DefaultSupremumEstimator(
        x0=[-3*pi], x0_bounds=[(-10*pi, 10*pi)], θ0=[-3*pi, 3*pi], θ0_bounds=[(-10*pi, None), (10*pi, None)])
    _, sample_trace = CosineVariateGeneratorWithUniform(
        hat_func_optimizer=uniform_supremum_estimator).sample(n_rv=1000)
    sample_trace.plot()


def test_acceptance_rejection_markov_log_normal_gen_with_exponential():
    # Estimator designed for Exponential density used as proposal desnity g
    exponential_supremum_estimator = DefaultSupremumEstimator(
        x0=[1000.01], x0_bounds=[(1000.01, None)], θ0=[0.01], θ0_bounds=[(0.001, None)])
    _, sample_trace = MarkovLogNormalVariateGeneratorWithExponential(r=0.009, σ=0.149, x_t_1=2300.78,
                                                                     hat_func_optimizer=exponential_supremum_estimator).sample(n_rv=5000)
    sample_trace.plot()


# test_acceptance_rejection_beta_gen_with_gaussian()
# test_acceptance_rejection_gaussian_gen_with_exponential()
# test_acceptance_rejection_cosine_gen_with_uniform()
# test_generate_poisson_rv()
test_acceptance_rejection_markov_log_normal_gen_with_exponential()
