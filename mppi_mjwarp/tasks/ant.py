"""Ant locomotion task for mujoco_warp."""

import numpy as np
import mujoco

from mppi_mjwarp.tasks.task_base import Task, ROOT


class Ant(Task):
    """Ant locomotion task."""

    running_cost_fields = ("qvel", "xpos", "xquat")
    terminal_cost_fields = ("xpos",)

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ant/scene.xml")
        mj_model.opt.timestep = 0.01
        super().__init__(mj_model, trace_sites=("torso_site",))

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

    def running_cost(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        qvel = data_np["qvel"]  # (nworld, nv)
        xpos = data_np["xpos"]  # (nworld, nbody, 3)
        xquat = data_np["xquat"]  # (nworld, nbody, 4)

        # Speed cost
        xy_velocity = qvel[:, :2]
        speed = np.sqrt(np.sum(xy_velocity ** 2, axis=1))

        # Orientation cost (torso upright)
        q = xquat[:, self.end_effector_pos_id, :]  # (nworld, 4) as (w,x,y,z)
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        upright = 1.0 - 2.0 * (x * x + y * y)
        orientation_cost = (1.0 - upright) ** 2

        return 10.0 * speed + 5.0 * orientation_cost

    def terminal_cost(self, data_np: dict) -> np.ndarray:
        xpos = data_np["xpos"]  # (nworld, nbody, 3)
        ee_pos = xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return 10.0 * np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))
