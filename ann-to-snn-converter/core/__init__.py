"""
Autonomous SNN Framework - Core Package
"""
from .evolution_engine import EvolutionEngine, EvolvingSNN
from .motivation import (
    IntrinsicMotivation,
    CuriosityModule,
    MasteryModule,
    EfficacyModule
)
from .self_modifier import SelfModifier, GoalEngine

__all__ = [
    'EvolutionEngine',
    'EvolvingSNN',
    'IntrinsicMotivation',
    'CuriosityModule',
    'MasteryModule',
    'EfficacyModule',
    'SelfModifier',
    'GoalEngine'
]

__version__ = '1.0.0'
