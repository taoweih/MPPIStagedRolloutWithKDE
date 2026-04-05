"""Standard MPPI controller — all logic inherited from MPPIBase."""

from mppi_mjwarp.algs.mppi_base import MPPIBase, extract_data_np  # noqa: F401


class MPPI(MPPIBase):
    """Model-predictive path integral control using mujoco_warp.

    Samples control sequences around a mean, rolls out in parallel via
    mujoco_warp, and updates the mean with an exponentially weighted average.
    """

    pass
