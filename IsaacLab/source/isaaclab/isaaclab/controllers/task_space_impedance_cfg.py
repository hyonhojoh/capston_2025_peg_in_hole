from collections.abc import Sequence
from dataclasses import MISSING

from isaaclab.utils import configclass

@configclass
class TaskSpaceImpedanceControllerCfg:
    """Task Space Impedance Controller Configuration."""

    # Restrict rx, ry
    is_restricted: bool = False

    # Stiffness coefficients (list of 6 values or single value)
    stiffness: Sequence[float] = MISSING
    # Damping ratios (list of 6 values or single value)
    damping_ratio: Sequence[float] = MISSING

    # Enable gravity compensation
    gravity_compensation: bool = True

    # Null space
    default_dof_pos_tensor: Sequence[float] = MISSING
    kp_null: float = MISSING
    kd_null: float = MISSING
