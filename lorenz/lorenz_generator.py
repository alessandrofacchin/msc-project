from __future__ import annotations
from typing import Tuple, Callable, List
from scipy.integrate import solve_ivp
from numpy import arange
import numpy as np
from scipy import stats

from .utils import trunc_exp


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
    start: float=0, stop: float=1, step: float=0.006) -> Tuple[np.ndarray, np.ndarray]:
        """Generates latent variables
        It uses the Lorenz system and integrates with the Explicit Runge-Kutta method of order 5(4).

        Args:
            x0 (float, optional): Initial point X coordinate. Defaults to 0.
            y0 (float, optional): Initial point Y coordinate. Defaults to 1.
            z0 (float, optional): Initial point Z coordinate. Defaults to 1.05.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time vector (T,) and matrix of latent variables (T,3).
        """
        
        soln = solve_ivp(lambda t, point: self.step(t, point), (start, stop), (x0, y0, z0),
                 dense_output=True)
        t = list(arange(start, stop, step))
        x, y, z = soln.sol(t)
        return np.array(t), np.array([x,y,z]).transpose()

    def generate_rates(self, n: int=30, bias: float=5, x0: float=0, y0: float=1, z0: float=1.05, 
    start: float=0, stop: float=1, step: float=0.006, seed: int=None, trials: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate firing rates
        It converts latent variables generated by the Lorenz system into firing rates

        Adapted from: https://github.com/catniplab/vlgp

        Args:
            n (int, optional): Total number of neurons. Defaults to 30, as in LFADS.
            bias (float, optional): Baseline firing rate (Hz). Defaults to 5, as in LFADS.
            x0 (float, optional): Initial point X coordinate. Defaults to 0.
            y0 (float, optional): Initial point Y coordinate. Defaults to 1.
            z0 (float, optional): Initial point Z coordinate. Defaults to 1.05.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.
            seed (int, optional): if provided, random number seed
            trials (int, optional): if provided, number of trials k. Defaults to 1

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time vector (T,), matrix of firing rates (k,T,n), 
            weight matrix (3,n) and matrix of latent variables (k,T,3).
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)

        z_list: List[np.ndarray] = []
        for _ in range(trials):
            t, z_tmp = self.generate_latent(x0=x0, y0=y0, z0=z0, start=start, stop=stop, step=step)
            z_list.append(z_tmp)

        z: np.ndarray = np.asarray(z_list)
        # Cast to type and size
        if z.ndim < 3:
            z = np.atleast_3d(z)
            z = np.rollaxis(z, axis=-1)

        ntrial, ntime, nlatent = z.shape
        weights: np.ndarray = (np.random.rand(nlatent, n) + 1) * np.sign(np.random.randn(nlatent, n))
        nchannel = weights.shape[1]

        # Initialise
        y = np.empty((ntrial, ntime, nchannel), dtype=float)
        f = np.empty_like(y, dtype=float)

        for m in range(ntrial):
            for i_t in range(ntime):
                eta = z[m, i_t, :] @ weights + bias
                f[m, i_t, :] = trunc_exp(eta)

        return t, f, weights, z


    def generate_spikes(self, n: int=30, bias: float=5, x0: float=0, y0: float=1, z0: float=1.05, 
    start: float=0, stop: float=1, step: float=0.006, seed: int=None, 
    encoding: Callable[[np.ndarray], np.ndarray]=lambda x: stats.poisson.rvs(x).clip(0,1),
    trials: int=1, conditions: int=1) -> Tuple[np.ndarray, np.ndarray]:
        """Generate firing rates
        It converts latent variables generated by the Lorenz system into firing rates

        Adapted from: https://github.com/catniplab/vlgp

        Args:
            n (int, optional): Total number of neurons. Defaults to 30, as in LFADS.
            bias (float, optional): Baseline firing rate (Hz). Defaults to 5, as in LFADS.
            x0 (float, optional): Initial point X coordinate. Defaults to 0.
            y0 (float, optional): Initial point Y coordinate. Defaults to 1.
            z0 (float, optional): Initial point Z coordinate. Defaults to 1.05.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 1, as in LFADS.
            step (float, optional): Time step. Defaults to 0.006, as in LFADS.
            seed (int, optional): if provided, random number seed
            encoding (Callable[[np.ndarray], np.ndarray], optional): function to convert rates into 
                spike count. Default to Poisson clipped between 1 and 0. It is equivalent to 
                Bernoulli P(1) = (1 - e^-(lam_t))
            trials (int, optional): if provided, number of trials k. Defaults to 1
            conditions (int, optional): if provided, number of conditions to try c. Defaults to 1

        Returns:
            Tuple[np.ndarray, np.ndarray]: Time vector (T,) and matrix of spikes (c,k,T,n), 
            matrix of firing rates (c,k,T,n), weight matrix (c,3,n) and matrix of latent variables (k,T,3).
        """
        
        f_list: List[np.ndarray] = []
        w_list: List[np.ndarray] = []
        z_list: List[np.ndarray] = []
        for _ in range(conditions):
            t, f_tmp, w_tmp, z_tmp = self.generate_rates(
                n=n,
                bias=bias,
                x0=x0,
                y0=y0,
                z0=z0,
                start=start,
                stop=stop,
                step=step,
                seed=seed,
                trials=trials
            )
            f_list.append(f_tmp)
            w_list.append(w_tmp)
            z_list.append(z_tmp)

        f: np.ndarray = np.asarray(f_list)
        w: np.ndarray = np.asarray(w_list)
        z: np.ndarray = np.asarray(z_list)

        return t, encoding(f), f, w, z