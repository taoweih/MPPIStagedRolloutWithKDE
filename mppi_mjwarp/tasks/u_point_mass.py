"""U-shaped point mass task for mujoco_warp."""

import numpy as np
import mujoco

from mppi_mjwarp.tasks.task_base import Task, ROOT


class UPointMass(Task):
    """Point mass navigation task."""

    # running_cost uses only control — no GPU data extraction needed
    running_cost_fields = ()
    terminal_cost_fields = ("xpos",)

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/u_point_mass/scene.xml"
        )
        super().__init__(mj_model, trace_bodies=("point_mass",))

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "point_mass"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

    def running_cost(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        # Cost based on control magnitude (distance travelled)
        return 100.0 * np.sqrt(np.sum(control ** 2, axis=1))

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        xpos = data_np["xpos"]  # (nworld, nbody, 3)
        ee_pos = xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))

    def terminal_cost(self, data_np: dict) -> np.ndarray:
        xpos = data_np["xpos"]  # (nworld, nbody, 3)
        ee_pos = xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return 100.0 * np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))
