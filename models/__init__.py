"""
Models Package
==============
Core simulation and control models for XAI-MPC distillation project.

Modules:
--------
- distillation_model: Binary distillation column dynamics
- mpc_controller: Model Predictive Control implementation
- ml_surrogate: Machine learning surrogate with XAI
"""

from .distillation_model import BinaryDistillationColumn
from .mpc_controller import SimplifiedMPC, DistillationMPC
from .ml_surrogate import SurrogateMLModel

__version__ = '1.0.0'
__author__ = 'BTP Student, IIT Patna'

__all__ = [
    'BinaryDistillationColumn',
    'SimplifiedMPC',
    'DistillationMPC',
    'SurrogateMLModel'
]