from scipy.stats import uniform
from numpy import repeat
from scipy.optimize import minimize
from abc import ABC, abstractmethod
from typing import List
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sys import float_info


class HatFunctionEstimator(ABC):

    """
    Abstract template for estimating Hat function H(x).
    It should be implemented to feed a strategy of constructing H(x)
    in the acceptance/rejection method. 
    """

    @abstractmethod
    def estimate_parameters(self, f: callable, g: callable): ...
    '''
          Function to construct H(x) and estimate
          any parameters inside it.

    '''

    @abstractmethod
    def H_x(self, x): ...
    '''
     Return H(x) for a given x.
    '''


class SupremumEstimatorTemplate(HatFunctionEstimator, ABC):
    '''
    Template implementation of HatFunctionOptimizer
    which is of the form H(x) = Cg(x,θ).
    It uses supremum of f(x)/g(x,θ) to find out C & θ.
    However, it does not provide maxima & mimnima functions.

    This can be extended with a suitable choice of 
    function optimization algorithm.
    '''

    def __init__(self):
        self._f: callable = None
        self._g: callable = None
        self._C: float = None
        self._θ: float = None

    @abstractmethod
    def _maximize_wrt_x(self, ratio_f: callable) -> tuple: ...

    @abstractmethod
    def _minimize_wrt_θ(self, ratio_f: callable) -> tuple: ...

    def _max_f_g_ratio(self, θ: tuple):
        '''
          Objective function to maximize f(x)/g(x, θ) w.r.t x
        '''
        def _c(x):
            c = 1.0
            proposal_prob = self._g(x, θ)
            if proposal_prob > 0:
                c = self._f(x) / proposal_prob

            return -c

        # Maximize f/g w.r.t x
        return self._maximize_wrt_x(_c)[0]

    def estimate_parameters(self, f: callable, g: callable):
        self._f = f
        self._g = g

        # Minimize maximum of f/g w.r.t θ
        result = self._minimize_wrt_θ(self._max_f_g_ratio)

        self._C = result[0]
        self._θ = result[1]
        print("Estimated C", self._C, " θ = ", self._θ)

    def H_x(self, x):
        '''
         Hat function H(x) = C * g(x)
        '''
        return self._C * self._g(x, self._θ)

    @property
    def θ_optimal_for_g(self): return self._θ


class DefaultSupremumEstimator(SupremumEstimatorTemplate):

    '''
      Default implemenation of SupremumEstimatorTemplate.
      It leverage scipy.optimize module to find maxima & minima of 
      f/g ratio.
    '''

    def __init__(self,
                 x0: List,  # Initial value of RV X of proposal density
                 # Sequence of (min, max) pairs of RV X
                 x0_bounds: List[tuple],
                 θ0: List,  # Initial value of parameter  of θ proposal density

                 # Sequence of (min, max) pairs of parameter  of θ
                 θ0_bounds: List[tuple]
                 ):
        self._x0 = x0
        self._x0_bounds = x0_bounds
        self._θ0 = θ0
        self._θ0_bounds = θ0_bounds
        super().__init__()

    def _maximize_wrt_x(self, ratio_f: callable) -> tuple:
        res = minimize(ratio_f,  # Minimize maximum f/g w.r.t x
                       x0=self._x0,   # Range & initial values of x
                       bounds=self._x0_bounds)
        return -res.fun, res.x

    def _minimize_wrt_θ(self, ratio_f: callable) -> tuple:
        res = minimize(ratio_f,  # Minimize maximum f/g w.r.t θ
                       x0=self._θ0,  # Range & initial values of θ
                       bounds=self._θ0_bounds)
        return res.fun, res.x


class SamplesTraceDisplay:

    """
    Class to provide comparative scatter plots of accepted &
    rejected samples.
    """

    def __init__(self, accepted, rejected, accepted_prob, rejected_prob):
        self.__accepted = accepted
        self.__rejected = rejected
        self.__accepted_prob = accepted_prob
        self.__rejected_prob = rejected_prob

    def plot(self):
        plt.style.use("seaborn-v0_8")
        accepted_samples_df = pd.DataFrame(
            {
                "Sample": self.__accepted,
                "Probability": self.__accepted_prob,
                "Type": repeat("Accepted Samples", len(self.__accepted)),
            }
        )
        rejected_samples_df = pd.DataFrame(
            {
                "Sample": self.__rejected,
                "Probability": self.__rejected_prob,
                "Type": repeat("Rejected Samples", len(self.__rejected)),
            }
        )

        sns.scatterplot(
            data=pd.concat([accepted_samples_df, rejected_samples_df]), x="Sample", y="Probability", hue="Type", style="Type"
        )
        plt.show()


class AcceptanceRejectionMethod(ABC):

    """
    Base tempplate for acceptance-rejection method. This should be
    extended to three functions should be overriden to implement
    drawing of samples from a target distrubution
    """

    def __init__(self, hat_func_estimator: HatFunctionEstimator):
        self._hat_func_estimator = hat_func_estimator

    @abstractmethod
    def target_pdf_f(self, x):
        '''
        Funbction to provide PDF of target density. This should be overidden by 
        the concrete class extending AcceptanceRejectionMethod
        '''
        ...

    @abstractmethod
    def _proposal_pdf_g(self, x, θ: tuple):
        '''
        Function to provide PDF of proposal density. This should be overidden by 
        the concrete class extending AcceptanceRejectionMethod
        '''
        ...

    @abstractmethod
    def _sample_from_proposal_g_with_θ_optimal(self, n_rv):
        '''
        Function to samples from the proposal density. This should be overidden by 
        the concrete class extending AcceptanceRejectionMethod
        '''
        ...

    def sample(self, n_rv):
        """
          Function to sample n_rv number of RV from target density f
        """

        self._hat_func_estimator.estimate_parameters(f=self.target_pdf_f,
                                                     g=self._proposal_pdf_g)
        count_accepted = 0
        rv_f_accepted = []
        rv_f_rejected = []

        def _filter_sample(x_u):
            x, u = x_u
            return u <= self.target_pdf_f(x)/self._hat_func_estimator.H_x(x)

        while count_accepted < n_rv:
            remaining_count = n_rv - count_accepted
            rv_g = self._sample_from_proposal_g_with_θ_optimal(remaining_count)
            u = uniform.rvs(0, 1, remaining_count)

            '''
            Reject samples based on criteria:
            uniform random variate < f(x)/H(x)
            '''
            accepted_samples = map(lambda x: x[0],
                                   filter(_filter_sample, zip(rv_g, u)))

            rv_f_accepted.extend(accepted_samples)

            # Collect rejected samples
            rv_f_rejected.extend(set(rv_g).difference(rv_f_accepted))

            count_accepted = len(rv_f_accepted)

        samples_trace = SamplesTraceDisplay(
            rv_f_accepted, rv_f_rejected,
            self.target_pdf_f(rv_f_accepted),
            self._hat_func_estimator.H_x(rv_f_rejected))

        return rv_f_accepted, samples_trace
