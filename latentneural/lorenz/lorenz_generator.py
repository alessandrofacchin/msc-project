from __future__ import annotations
from typing import Tuple, Callable, List, Optional
from scipy.integrate import solve_ivp
from numpy import arange
import numpy as np
from scipy import stats

from .utils import trunc_exp
from .initial_conditions import constant


class LorenzGenerator(object):

    _sigma: float = 10
    _rho: float = 28
    _beta: float = 8/3

    def __init__(self, sigma: float=None, rho: float=None, beta: float=None):
        """Lorenz Generator

        Args:
            sigma (float, optional): Lorenz attractor's sigma. Defaults to 10, as in LFADS.
            rho (float, optional): Lorenz attractor's rho. Defaults to 28, as in LFADS.
            beta (float, optional): Lorenz attractor's beta. Defaults to 2.667, as in LFADS.
        """
        self.sigma: float = sigma if sigma is not None else self._sigma
        self.rho: float = rho if rho is not None else self._rho
        self.beta: float = beta if beta is not None else self._beta
        
    def step(self, t: float, point: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """Lorenz System single step

        Args:
            point (Tuple[float, float, float]): coordinates of the point

        Returns:
            Tuple[float, float, float]: The next point
        """
        x, y, z = point
        x_dot = self.sigma * (y - x)
        y_dot = self.rho * x - y - x * z
        z_dot = x * y - self.beta * z
        return (x_dot, y_dot, z_dot)

    def generate_latent(self, x0: float=0, y0: float=1, z0: float=1.05, 
    start: float=0, stop: float=1, step: float=0.006, warmup: int=0) -> Tuple[np.ndarray, np.ndarray]:
        """Generates latent variables
        
        It uses the Lorenz system and integrates with the Explicit Runge-Kutta method of order 5(4).
        The output latent variables are shuffled before being returned.

        Args:
            x0 (float, optional): Initial point X coordinate. Defaults to 0.
            y0 (float, optional): Initial point Y coordinate. Defaults to 1.
            z0 (float, optional): Initial point Z coordinate. Defaults to 1.05.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.
            warmup (int, optional): Steps skipped. Defaults to 0.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time vector (t,) and matrix of latent variables (t,3).
        """
        
        soln = solve_ivp(lambda t, point: self.step(t, point), (start, stop), (x0, y0, z0),
                 dense_output=True)
        t = list(arange(start, stop + warmup * step, step))
        x, y, z = soln.sol(t)

        lorenz = np.array([x[warmup:], y[warmup:], z[warmup:]])
        np.random.shuffle(lorenz) # Mixing columns

        return np.array(t[warmup:]) - t[warmup], lorenz.reshape(lorenz.shape).transpose()

    def normalise_latent(self, z: np.ndarray) -> np.ndarray:
        """Normalise latent variables
       
        Args:
            z (np.ndarray): Latent variables (k,t,3).

        Returns:
            np.ndarray: normalised latent variables (k,t,3).
        """

        z -= np.mean(z, axis=None)
        z /= np.max(np.abs(z))
        return z

    def generate_rates(self, n: int=30, base: float=5, initial_conditions: Callable[..., Tuple[float, float, float]]=constant(), l: int=3,
    start: float=0, stop: float=1, step: float=0.006, warmup: int=0, seed: int=None, trials: int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate firing rates
        It converts latent variables generated by the Lorenz system into firing rates

        Adapted from: https://github.com/catniplab/vlgp

        Args:
            n (int, optional): Total number of neurons. Defaults to 30, as in LFADS.
            base (float, optional): Baseline firing rate (Hz). Defaults to 5, as in LFADS.
            initial_conditions (Callable[..., Tuple[float, float, float]], optional): 
                Generator for initial coordinates. Defaults to constant().
            l (int, optional): number of latent variables. The first l variables from
                the Lorenz system are used. Defaults to 3.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.
            warmup (int, optional): Steps skipped. Defaults to 0.
            seed (int, optional): if provided, random number seed
            trials (int, optional): number of trials k. Defaults to 1

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Time vector (t,), 
            matrix of firing rates (k,t,n), weight matrix (l,n) and matrix of latent variables (k,t,3).
        """
        # Set seed
        if seed is not None:
            np.random.seed(seed)

        if (l > 3) or (l <1):
            raise ValueError('Latent variables must be between 1 and 3')
            
        z_list: List[np.ndarray] = []
        for _ in range(trials):
            x0, y0, z0 = initial_conditions()
            t, z_tmp = self.generate_latent(x0=x0, y0=y0, z0=z0, start=start, stop=stop, step=step, warmup=warmup)
            z_list.append(z_tmp)

        z: np.ndarray = np.asarray(z_list)
        z = self.normalise_latent(z)

        # Cast to type and size
        if z.ndim < 3:
            z = np.atleast_3d(z)
            z = np.rollaxis(z, axis=-1)

        ntrial, ntime, _ = z.shape
        weights: np.ndarray = np.random.uniform(1, 2, (l, n)) * np.sign(np.random.randn(l, n)) # As in https://github.com/colehurwitz/plfads/blob/bcf02b3d94fb1204f72836958acb21d60af96a15/generate_lorenz_data.py#L111
        nchannel = weights.shape[1]

        # Initialise
        y = np.empty((ntrial, ntime, nchannel), dtype=float)
        f = np.empty_like(y, dtype=float)

        for m in range(ntrial):
            for i_t in range(ntime):
                eta = z[m, i_t, :l] @ weights + np.log(base) # As in https://github.com/colehurwitz/plfads/blob/bcf02b3d94fb1204f72836958acb21d60af96a15/generate_lorenz_data.py#L113
                f[m, i_t, :] = trunc_exp(eta)

        return t, f, weights, z

    def generate_spikes(self, n: int=30, base: float=5, initial_conditions: Callable[..., Tuple[float, float, float]]=constant(), l: int=3,
    start: float=0, stop: float=1, step: float=0.006, warmup: int=0, seed: int=None, 
    encoding: Callable[[np.ndarray], np.ndarray]=lambda x: stats.poisson.rvs(x).clip(0,1).reshape(x.shape),
    trials: int=1, conditions: int=1) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate spikes
        It converts latent variables generated by the Lorenz system into spike trains

        Adapted from: https://github.com/catniplab/vlgp

        Args:
            n (int, optional): Total number of neurons. Defaults to 30, as in LFADS.
            base (float, optional): Baseline firing rate (Hz). Defaults to 5, as in LFADS.
            initial_conditions (Callable[..., Tuple[float, float, float]], optional): 
                Generator for initial coordinates. Defaults to constant().
            l (int, optional): number of latent variables. The first l variables from
                the Lorenz system are used. Defaults to 3.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.
            warmup (int, optional): Steps skipped. Defaults to 0.
            seed (int, optional): if provided, random number seed
            encoding (Callable[[np.ndarray], np.ndarray], optional): function to convert rates into 
                spike count. Default to Poisson clipped between 1 and 0. It is equivalent to 
                Bernoulli P(1) = (1 - e^-(lam_t))
            trials (int, optional): number of trials k. Defaults to 1
            conditions (int, optional): number of conditions to try c. Defaults to 1

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Time vector (t,),
            matrix of spikes (c,k,t,n), matrix of firing rates (c,k,t,n), weight matrix (c,l,n) 
            and matrix of latent variables (c,k,t,3).
        """
        # Set seed
        if seed is not None:
            np.random.seed(seed)

        f_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        z_list: List[np.ndarray] = []
        for _ in range(conditions):
            t, f_tmp, w_tmp, z_tmp = self.generate_rates(
                n=n,
                base=base,
                initial_conditions=initial_conditions,
                start=start,
                stop=stop,
                step=step,
                warmup=warmup,
                seed=np.random.randint(1e6),
                trials=trials, 
                l=l
            )
            f_list.append(f_tmp)
            w_list.append(w_tmp)
            z_list.append(z_tmp)

        f: np.ndarray = np.asarray(f_list)
        w: np.ndarray = np.asarray(w_list)
        z: np.ndarray = np.asarray(z_list)

        return t, encoding(f * step), f, w, z

    def generate_spikes_and_behaviour(self, n: int=30, base: float=5, initial_conditions: Callable[..., Tuple[float, float, float]]=constant(),
    l: int=3, b: int=3, y: int=1, start: float=0, stop: float=1, step: float=0.006, warmup: int=0, seed: int=None, 
    encoding: Callable[[np.ndarray], np.ndarray]=lambda x: stats.poisson.rvs(x).clip(0,1).reshape(x.shape),
    trials: int=1, conditions: int=1, behaviour_overlay: Optional[Callable[..., np.ndarray]]=None) -> \
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate spikes and behaviour
        It converts latent variables generated by the Lorenz system into spikes and behaviour

        Adapted from: https://github.com/catniplab/vlgp

        Args:
            n (int, optional): Total number of neurons. Defaults to 30, as in LFADS.
            base (float, optional): Baseline firing rate (Hz). Defaults to 5, as in LFADS.
            initial_conditions (Callable[..., Tuple[float, float, float]], optional): 
                Generator for initial coordinates. Defaults to constant().
            l (int, optional): number of latent variables used in neural activity. The first l variables from
                the Lorenz system are used. Defaults to 3.
            b (int, optional): number of latent variables used in behaviour. The last l variables from
                the Lorenz system are used. Defaults to 3.
            y (int, optional): number behavioural channels. Defaults to 1.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.
            warmup (int, optional): Steps skipped. Defaults to 0.
            seed (int, optional): if provided, random number seed
            encoding (Callable[[np.ndarray], np.ndarray], optional): function to convert rates into 
                spike count. Default to Poisson clipped between 1 and 0. It is equivalent to 
                Bernoulli P(1) = (1 - e^-(lam_t))
            trials (int, optional): number of trials k. Defaults to 1
            conditions (int, optional): number of conditions to try c. Defaults to 1

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 
            Time vector (t,), matrix of behaviour (c,k,t,y), matrix of spikes (c,k,t,n), 
            matrix of firing rates (c,k,t,n), behavioural weights matrix (c,b,y), weight matrix (c,l,n) 
            and matrix of latent variables (c,k,t,3).
        """
        # Set seed
        if seed is not None:
            np.random.seed(seed)

        f_list: List[np.ndarray] = []
        s_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        z_list: List[np.ndarray] = []
        b_list: List[np.ndarray] = []
        bw_list: List[np.ndarray] = []
        for _ in range(conditions):
            t, f_tmp, w_tmp, z_tmp = self.generate_rates(
                n=n,
                base=base,
                initial_conditions=initial_conditions,
                start=start,
                stop=stop,
                step=step,
                warmup=warmup,
                seed=np.random.randint(1e6),
                trials=trials, 
                l=l
            )
            behavioural_weights: np.ndarray = np.random.normal(0, 5, (b, y)) # As in https://github.com/colehurwitz/plfads/blob/bcf02b3d94fb1204f72836958acb21d60af96a15/generate_lorenz_data.py#L135
            behaviours_normal = z_tmp[:,:,-b:] @ behavioural_weights

            if behaviour_overlay:
                overlay = behaviour_overlay(step=step, start=start, stop=stop) # add sine overlay as in https://github.com/colehurwitz/plfads/blob/bcf02b3d94fb1204f72836958acb21d60af96a15/generate_lorenz_data.py#L139
                behaviours_noiseless = behaviours_normal + overlay[None, :, None]
            else:
                behaviours_noiseless = behaviours_normal

            behaviour = behaviours_noiseless + np.random.normal(0, 1, behaviours_noiseless.shape) # As in https://github.com/colehurwitz/plfads/blob/bcf02b3d94fb1204f72836958acb21d60af96a15/generate_lorenz_data.py#L147

            f_list.append(f_tmp)
            s_list.append(encoding(f_tmp * step))
            w_list.append(w_tmp)
            z_list.append(z_tmp)
            b_list.append(behaviour)
            bw_list.append(behavioural_weights)

        f: np.ndarray = np.asarray(f_list)
        s: np.ndarray = np.asarray(s_list)
        w: np.ndarray = np.asarray(w_list)
        z: np.ndarray = np.asarray(z_list)
        b: np.ndarray = np.asarray(b_list)
        bw: np.ndarray = np.asarray(bw_list)

        return t, b, s, f, bw, w, z