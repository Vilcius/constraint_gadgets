"""
constraint_qaoa.py -- Backward-compatibility shim.

The class has been renamed to VCG (Variable Constraint Gadget).
Import from core.vcg going forward.
"""

from .vcg import VCG as ConstraintQAOA

__all__ = ["ConstraintQAOA"]
