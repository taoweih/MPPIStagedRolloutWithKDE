"""UR5e reach task for mujoco_warp."""

import numpy as np
import mujoco

from mppi_mjwarp.tasks.task_base import Task, ROOT


class UR5e(Task):
    """Reach task for the UR5e robot arm."""

    running_cost_fields = ("sensordata",)
    terminal_cost_fields = ("site_xpos", "xpos")

    def __init__(self) -> None:
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ur5e/scene.xml")
        mj_model.opt.timestep = 0.01
        # mj_model.opt.ccd_iterations = 1000  # UR5e complex geometry requires high CCD iterations
        super().__init__(mj_model, trace_sites=("attachment_site",))

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

        self.ee_vel_sensor_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "ee_linvel"
        )
        self.ee_vel_sensor_adr = self.mj_model.sensor_adr[self.ee_vel_sensor_id]

    def running_cost(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        sensordata = data_np["sensordata"]  # (nworld, nsensordata)
        ee_vel = sensordata[:, self.ee_vel_sensor_adr:self.ee_vel_sensor_adr + 3]
        speed = np.sqrt(np.sum(ee_vel ** 2, axis=1))
        return 10.0 * speed

    def terminal_cost(self, data_np: dict) -> np.ndarray:
        site_xpos = data_np["site_xpos"]  # (nworld, nsite, 3)
        xpos = data_np["xpos"]  # (nworld, nbody, 3)
        ee_pos = site_xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return 10.0 * np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))

    def success_function(self, data_np: dict, control: np.ndarray) -> np.ndarray:
        site_xpos = data_np["site_xpos"]
        xpos = data_np["xpos"]
        ee_pos = site_xpos[:, self.end_effector_pos_id, :]
        goal_pos = xpos[:, self.goal_pos_id, :]
        return np.sqrt(np.sum((ee_pos - goal_pos) ** 2, axis=1))