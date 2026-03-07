from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class UDrone(Task):
    """Quadrotor navigation task for Skydio X2."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/u_drone/scene.xml"
        )
        mj_model.opt.timestep = 0.01
        super().__init__(
            mj_model,
            trace_sites=["tracking_site"],
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

        # save hover control for effort penalty
        self.hover_ctrl = jnp.array(mj_model.keyframe("hover").ctrl)

    def _tilt_cost(self, state: mjx.Data) -> jax.Array:
        R = state.xmat[self.body_id].reshape(3, 3)
        tilt_err = 1.0 - R[2, 2] 
        return tilt_err ** 2

    def _altitude_cost(self, state: mjx.Data) -> jax.Array:
        z = state.xpos[self.body_id, 2]
        return jnp.maximum(0.0, 0.15 - z) ** 2

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        speed = jnp.sqrt(jnp.sum(jnp.square(state.qvel[:3])))

        tilt_cost = self._tilt_cost(state)
        alt_cost = self._altitude_cost(state)

        cost = 10.0 * speed + 10.0 * tilt_cost + 5.0 * alt_cost
        return cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        ee = state.site_xpos[self.end_effector_pos_id]
        goal = state.xpos[self.goal_pos_id]
        distance = jnp.sqrt(jnp.sum(jnp.square(ee - goal)))
        return 10.0 * distance

    def success_function(self, state, control):
        ee = state.site_xpos[self.end_effector_pos_id]
        goal = state.xpos[self.goal_pos_id]
        return jnp.sqrt(jnp.sum(jnp.square(ee - goal)))

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
