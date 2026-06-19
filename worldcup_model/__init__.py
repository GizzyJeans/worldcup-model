"""Expert model for World Cup betting.

A Dixon-Coles (time-weighted Poisson) goals model blended with a World
Football Elo rating system, turned into calibrated market probabilities and
compared against bookmaker odds to surface positive expected-value bets.
"""

from .model import ExpertModel

__all__ = ["ExpertModel"]
