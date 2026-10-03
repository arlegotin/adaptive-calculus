"""Storage-coordinate update geometry; optimizer state remains caller-owned."""
from .actions import assist_boundaries, compose_action, multiplicative_proposal

__all__ = ['assist_boundaries', 'compose_action', 'multiplicative_proposal']
