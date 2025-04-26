from typing import TypedDict, List
import numpy as np
from abc import ABC, abstractmethod
from scipy.stats import norm as Φ
from enum import Enum

from chapter6.diffusion_model import GeometricBrownianMotionProcess, DiffusionProcessParameters
from chapter5.base_forecasting import TimeUnitTransformer
from chapter8.options_common import OptionsResult, OptionsTemplate


class _BlackScholesOptionsGBMProcess(GeometricBrownianMotionProcess, ABC):

    '''
    Abstract class for the stochastic process of Options & Greeks.
    This should be extended by Call, Put & corresponding 5 greeks
    to create their respective processes.

    Options & Greeks should follow the GBM process to leverage
    this class.
    '''

    def __init__(self, r, σ,
                 strike_price_K=None,
                 expiry_time_T=None,
                 initial_state=1.0,
                 n_sample_paths=5):

        super().__init__(r=r,
                         σ=σ,
                         initial_state=initial_state,
                         n_sample_paths=n_sample_paths)

        self._underlying_s_values = np.ndarray(
            shape=(n_sample_paths, expiry_time_T+1), dtype=float)
        self._current_sample_path = 0
        self._strike_price_K = strike_price_K
        self._expiry_time_T = expiry_time_T

        # V(S,0;K,T)
        self._initial_v_state = self._pay_off_and_greek_f(
            self._initial_state, self._t)

        # V(S,t;K,T)
        self._v_state = self._pay_off_and_greek_f(
            self._initial_state, self._t)

    def _update_current_state(self, z):
        '''
        Function H(x) of the Monte-Carlo framework for simulating sample paths
        of the stochastic process. It computes the current V(S,t;K,T) 
        and stores underlying asset values of the current sample path.
        '''
        super()._update_current_state(z)
        self._underlying_s_values[self._current_sample_path][self._t] = self._state_t
        self._v_state = self._pay_off_and_greek_f(self._state_t, self._t)
        return self._v_state

    def _reset_new_sample_path_state(self):
        if self._t >= self._T:
            self._state_t = self._initial_state
            self._v_state = self._pay_off_and_greek_f(
                self._initial_state, self._t)
            self._t = 0
            self._current_sample_path = self._current_sample_path + 1
        self._t = self._t + 1

    def _d1(self, s_t, t):
        '''
        Term d1 in the expression for both call & put
            V_C (S_t,t;K,T)= S_t Φ(d_1 )- Ke^(-r(T-t) ) Φ(d_2)
            V_P (S_t,t;K,T)= Ke^(-r(T-t) ) Φ(〖-d〗_2 )- S_t Φ(〖-d〗_1 )

        It is given by
            d_1 =  ( log⁡〖S_t/K〗-(r+1/2 σ^2 )(T-t))/(σ√(T-t)),
        '''
        return (np.log(s_t/self._strike_price_K) +
                (self._r + 0.5*(self._σ ** 2)) *
                (self._expiry_time_T-t))/(self._σ * np.sqrt(self._expiry_time_T-t))

    def _d2(self, s_t, t):
        '''
        Term d1 in the expression for both call & put
            V_C (S_t,t;K,T)= S_t Φ(d_1 )- Ke^(-r(T-t) ) Φ(d_2)
            V_P (S_t,t;K,T)= Ke^(-r(T-t) ) Φ(〖-d〗_2 )- S_t Φ(〖-d〗_1 )

        It is given by
            d_2 = d_1-σ√(T-t)
        '''
        return self._d1(s_t, t) - self._σ * np.sqrt(self._expiry_time_T - t)

    @property
    def value_at_0(self):
        '''
        This is the fair value of the premium paid to the
        option seller. This is a present value of the price
        at expiry time i.e., measured at t=0

        '''
        return self._initial_v_state

    @property
    def underlying_s_values(self):
        '''
        All values of sample paths of the underlying asset.
        '''
        return self._underlying_s_values

    @abstractmethod
    def _pay_off_and_greek_f(self, s_t, t): ...
    '''
       Payoff function of call, put & greeks. This should be 
       overriden by the implenting class.
    '''

    @property
    @abstractmethod
    def label(self): ...


class _CallOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    '''
     Class for call options
    '''

    def _pay_off_and_greek_f(self, s_t, t):
        d1 = self._d1(s_t, t)
        d2 = self._d2(s_t, t)
        '''
         It computes the call value as given by,
            V_C (S_t,t;K,T)= S_t Φ(d_1 )- Ke^(-r(T-t) ) Φ(d_2)           
        '''
        return (s_t * Φ.cdf(d1)) - (self._strike_price_K * np.exp(-self._r*(self._expiry_time_T-t)) * Φ.cdf(d2))

    @property
    def label(self):
        return "Call Option V(S,t)"


class _PutOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    '''
     Class for put options
    '''

    def _pay_off_and_greek_f(self, s_t, t):
        d1 = self._d1(s_t, t)
        d2 = self._d2(s_t, t)
        '''
         It computes the put value as given by,
            V_P (S_t,t;K,T)= Ke^(-r(T-t) ) Φ(〖-d〗_2 )- S_t Φ(〖-d〗_1 )        
        '''
        return (self._strike_price_K * np.exp(-self._r*(self._expiry_time_T-t)) * Φ.cdf(-d2)) - (s_t * Φ.cdf(-d1))

    @property
    def label(self):
        return "Put Option V(S,t)"


'''   Delta '''


class _DeltaForCallOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return Φ.cdf(self._d1(s_t, t))

    @property
    def label(self):
        return OptionGreeks.Δ.value.greek_label + " Call Option " + OptionGreeks.Δ.value.greek_label + "(S,t)"


class _DeltaForPutOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return Φ.cdf(self._d1(s_t, t)) - 1

    @property
    def label(self):
        return OptionGreeks.Δ.value.greek_label + " Put Option " + OptionGreeks.Δ.value.greek_label + "(S,t)"


''' Gamma '''


class _GammaForCallOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return Φ.pdf(self._d1(s_t, t))/(self._σ * s_t * np.sqrt(self._expiry_time_T-t))

    @property
    def label(self):
        return OptionGreeks.Γ.value.greek_label + " Call Option " + OptionGreeks.Γ.value.greek_label + "(S,t)"


class _GammaForPutOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return Φ.pdf(self._d1(s_t, t))/(self._σ * s_t * np.sqrt(self._expiry_time_T-t))

    @property
    def label(self):
        return OptionGreeks.Γ.value.greek_label + " Put Option " + OptionGreeks.Γ.value.greek_label + "(S,t)"


''' Vega '''


class _VegaForCallOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return Φ.pdf(self._d1(s_t, t))*(self._σ * s_t * np.sqrt(self._expiry_time_T-t))

    @property
    def label(self):
        return OptionGreeks.Κ.value.greek_label + " Call Option " + OptionGreeks.Κ.value.greek_label + "(S,t)"


class _VegaForPutOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return Φ.pdf(self._d1(s_t, t)) * (self._σ * s_t * np.sqrt(self._expiry_time_T-t))

    @property
    def label(self):
        return OptionGreeks.Κ.value.greek_label + " Put Option " + OptionGreeks.Κ.value.greek_label + "(S,t)"


''' Theta '''


class _ThetaForCallOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return -((self._σ * s_t * Φ.pdf(self._d1(s_t, t)))/(2.0*np.sqrt(self._expiry_time_T-t))) \
            - (self._r*self._strike_price_K *
                np.exp(-self._r*(self._expiry_time_T-t))*Φ.cdf(self._d2(s_t, t)))

    @property
    def label(self):
        return OptionGreeks.Θ.value.greek_label + " Call Option " + OptionGreeks.Θ.value.greek_label + "(S,t)"


class _ThetaForPutOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return -((self._σ * s_t * Φ.pdf(-self._d1(s_t, t)))/(2.0*np.sqrt(self._expiry_time_T-t))) \
            + (self._r*self._strike_price_K *
                np.exp(-self._r*(self._expiry_time_T-t))*Φ.cdf(-self._d2(s_t, t)))

    @property
    def label(self):
        return OptionGreeks.Θ.value.greek_label + " Put Option " + OptionGreeks.Θ.value.greek_label + "(S,t)"


''' Rho '''


class _RhoForCallOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return self._strike_price_K * np.exp(-self._r*(self._expiry_time_T-t)) * Φ.cdf(self._d2(s_t, t))

    @property
    def label(self):
        return OptionGreeks.Ρ.value.greek_label + " Call Option " + OptionGreeks.Ρ.value.greek_label + "(S,t)"


class _RhoForPutOptionsGBMProcess(_BlackScholesOptionsGBMProcess):

    def _pay_off_and_greek_f(self, s_t, t):
        return -1.0 * self._strike_price_K * np.exp(-self._r*(self._expiry_time_T-t)) * Φ.cdf(-self._d2(s_t, t))

    @property
    def label(self):
        return OptionGreeks.Ρ.value.greek_label + " Put Option " + OptionGreeks.Ρ.value.greek_label + "(S,t)"


class OptionGreeks(Enum):

    class _Greek:

        def __init__(self, greek_label, call_process, put_process):
            self.greek_label = greek_label
            self.call_process = call_process
            self.put_process = put_process

        def __hash__(self):
            return hash(self.greek_label)

        def __eq__(self, obj):
            return self.greek_label == obj.greek_label

    Δ = _Greek('Delta', _DeltaForCallOptionsGBMProcess.__name__,
               _DeltaForPutOptionsGBMProcess.__name__)
    Γ = _Greek('Gamma', _GammaForCallOptionsGBMProcess.__name__,
               _GammaForPutOptionsGBMProcess.__name__)
    Θ = _Greek('Theta', _ThetaForCallOptionsGBMProcess.__name__,
               _ThetaForPutOptionsGBMProcess.__name__)
    Κ = _Greek('Vega', _VegaForCallOptionsGBMProcess.__name__,
               _VegaForPutOptionsGBMProcess.__name__)
    Ρ = _Greek('Rho', _RhoForCallOptionsGBMProcess.__name__,
               _RhoForPutOptionsGBMProcess.__name__)


class BlackScholesOptionsRiskNeutralGBMModel(OptionsTemplate):

    '''
    Class for computing call, put & greeks based on  
    BlackScholes model.
    '''

    def __init__(self, parameters: DiffusionProcessParameters,
                 time_unit_transformer: TimeUnitTransformer,
                 n_sample_paths=5):
        self._parameters = parameters
        self._time_u = time_unit_transformer
        self._time_u._t0 = parameters['t0']

        self._n_sample_paths = n_sample_paths

    def estimate_call(self, expiry_time_T, strike_price_K, greeks: List[OptionGreeks] = None) -> tuple:
        call_process = _CallOptionsGBMProcess(
            r=self._parameters['r'],
            σ=self._parameters['σ'],
            initial_state=self._parameters['s0'],
            expiry_time_T=expiry_time_T,
            strike_price_K=strike_price_K,
            n_sample_paths=self._n_sample_paths)

        # Forecast call options & greeks till expiry time T
        return self._forecast_options(expiry_time_T,
                                      call_process), self._forecast_greeks(expiry_time_T,
                                                                           strike_price_K, greeks)

    def estimate_put(self, expiry_time_T, strike_price_K, greeks: List[OptionGreeks] = None) -> tuple:
        put_process = _PutOptionsGBMProcess(
            r=self._parameters['r'],
            σ=self._parameters['σ'],
            initial_state=self._parameters['s0'],
            expiry_time_T=expiry_time_T,
            strike_price_K=strike_price_K,
            n_sample_paths=self._n_sample_paths)

        # Forecast put options & greeks till expiry time T
        return self._forecast_options(expiry_time_T,
                                      put_process), self._forecast_greeks(expiry_time_T,
                                                                          strike_price_K, greeks, for_call_option=False)

    def _forecast_options(self, expiry_time_T, options_process: _BlackScholesOptionsGBMProcess, for_call_option=True):
        '''
        Leverage the stochastic process of options (call or put) and 
        return the forecast result wrapped inside OptionsResult.
        '''
        result = options_process.forecast(
            T=expiry_time_T, time_unit_transformer=self._time_u)
        return OptionsResult(all_values=result,
                             value_at_0=options_process.value_at_0,
                             underlying_s_values=options_process.underlying_s_values,
                             label=options_process.label, for_call_option=for_call_option)

    class _GreeksProcessOutputDict(TypedDict):
        '''
        Dictionary holding output from Greeks processes.
        '''
        OptionGreeks.Δ: OptionGreeks
        OptionGreeks.Γ: OptionGreeks
        OptionGreeks.Θ: OptionGreeks
        OptionGreeks.Κ: OptionGreeks
        OptionGreeks.Ρ: OptionGreeks

    def _forecast_greeks(self, expiry_time_T, strike_price_K, greeks: List[OptionGreeks] = None, for_call_option=True) -> _GreeksProcessOutputDict:

        greeks_process_result: _GreeksProcessOutputDict = None

        if greeks:
            glb = globals()
            greeks_process_result = {}
            option_greek_process = None
            for greek in greeks:
                if for_call_option:
                    option_greek_process = greek.value.call_process
                else:
                    option_greek_process = greek.value.put_process

                greeks_process_result[greek] = self._forecast_options(
                    expiry_time_T, glb[option_greek_process](r=self._parameters['r'],
                                                             σ=self._parameters['σ'],
                                                             initial_state=self._parameters['s0'],
                                                             expiry_time_T=expiry_time_T,
                                                             strike_price_K=strike_price_K,
                                                             n_sample_paths=self._n_sample_paths), for_call_option=for_call_option)

        return greeks_process_result
