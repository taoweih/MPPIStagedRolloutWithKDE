from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class UDrone(Task):
    """Standup task for the Unitree G1 humanoid."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/u_drone/scene.xml")
        mj_model.opt.timestep = 0.01
        super().__init__(
            mj_model,
            trace_sites= ["tracking_site"] #["imu_in_torso", "left_foot", "right_foot"],
        )

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "tracking_site"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

        self.body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "x2"
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        end_effector_pos = state.site_xpos[self.end_effector_pos_id]
        goal_pos = state.xpos[self.goal_pos_id]
        pos_cost = jnp.sum(jnp.square(end_effector_pos - goal_pos),axis=0)

        v_world = state.qvel[:3]
        w_world = state.qvel[3:6]
        vel_cost     = jnp.sum(v_world ** 2)
        angvel_cost  = jnp.sum(w_world ** 2)

        R = state.xmat[self.body_id].reshape(3, 3)
        tilt_err = 1.0 - R[2, 2]
        tilt_cost = tilt_err ** 2

        u = control
        u_hover = jnp.full_like(u, self.mj_model.keyframe("hover").ctrl)
        effort_cost = jnp.sum((u - u_hover) ** 2)

        z = state.xpos[self.body_id, 2]
        alt_cost = jnp.maximum(0.0, 0.5 - z) ** 2

        cost = 50*pos_cost + 50000 * tilt_cost + 10*effort_cost + 10*alt_cost + 0.1*vel_cost + 0.01 * angvel_cost
        return cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        end_effector_pos = state.site_xpos[self.end_effector_pos_id]
        goal_pos = state.xpos[self.goal_pos_id]
        pos_cost = jnp.sum(jnp.square(end_effector_pos - goal_pos),axis=0) ** 4
        # return self.running_cost(state, jnp.zeros(self.model.nu))
        return 100*pos_cost
    
    def success_function(self, state, control):
        end_effector_pos = state.site_xpos[self.end_effector_pos_id]
        goal_pos = state.xpos[self.goal_pos_id]
        pos_cost = jnp.sum(jnp.square(end_effector_pos - goal_pos),axis=0) 
        return pos_cost

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the friction parameters."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.5, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Randomly perturb the measured base position and velocities."""
        rng, q_rng, v_rng = jax.random.split(rng, 3)
        q_err = 0.01 * jax.random.normal(q_rng, (7,))
        v_err = 0.01 * jax.random.normal(v_rng, (6,))

        qpos = data.qpos.at[0:7].set(data.qpos[0:7] + q_err)
        qvel = data.qvel.at[0:6].set(data.qvel[0:6] + v_err)

        return {"qpos": qpos, "qvel": qvel}
