from __future__ import annotations
from typing import Tuple
from scipy.integrate import solve_ivp
from numpy import arange
import pandas as pd


class LorenzGenerator(object):

    _sigma: float = 10
    _rho: float = 28
    _beta: float = 2.667

    def __init__(self, sigma: float=None, rho: float=None, beta: float=None):
        """Lorenz Generator

        Args:
            sigma (float, optional): Lorenz attractor's sigma. Defaults to 10.
            rho (float, optional): Lorenz attractor's rho. Defaults to 28.
            beta (float, optional): Lorenz attractor's beta. Defaults to 2.667.
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

    def generate_data(self, x0: float=0, y0: float=1, z0: float=1.05, start: float=0, stop: float=100, step: float=0.01) -> pd.DataFrame:
        """[summary]

        Args:
            x0 (float, optional): Initial point X coordinate. Defaults to 0.
            y0 (float, optional): Initial point Y coordinate. Defaults to 1.
            z0 (float, optional): Initial point Z coordinate. Defaults to 1.05.
            start (float, optional): Starting time. Defaults to 0.
            stop (float, optional): Terminal time. Defaults to 100.
            step (float, optional): Time step. Defaults to 0.01.

        Returns:
            pd.DataFrame: Table containing x, y and x for columns and t on the index
        """
        
        soln = solve_ivp(lambda t, point: self.step(t, point), (start, stop), (x0, y0, z0),
                 dense_output=True)
        t = list(arange(start, stop, step))
        x, y, z = soln.sol(t)
        return pd.DataFrame(dict(x=x, y=y, z=z), index=t)