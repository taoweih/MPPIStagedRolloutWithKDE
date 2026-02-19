from typing import Dict

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import Task


class Ant(Task):
    """Ant task."""

    def __init__(self) -> None:
        """Load the MuJoCo model and set task parameters."""
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/ant/scene.xml")
        mj_model.opt.timestep = 0.01

        super().__init__(
            mj_model,
            trace_sites= ["torso_site"],
        )

        self.end_effector_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "torso"
        )
        self.goal_pos_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        xy_velocity = state.qvel[:2]
        speed = jnp.sqrt(jnp.sum(jnp.square(xy_velocity)))
        distance_cost = speed

        ctrl_cost = 0.5 * jnp.sum(jnp.square(control))

        contact_forces = jnp.clip(state.qfrc_constraint, -1.0, 1.0)
        contact_cost = 5e-4 * jnp.sum(jnp.square(contact_forces))

        z_pos = state.xpos[self.end_effector_pos_id, 2]
        is_healthy = jnp.logical_and(z_pos >= 0.4, z_pos <= 0.8)
        healthy_reward = 5.0 * is_healthy.astype(jnp.float32)

        q = state.xquat[self.end_effector_pos_id]  # (w,x,y,z)
        w, x, y, z = q
        upright = 1.0 - 2.0 * (x*x + y*y)         
        orientation_cost = jnp.square(1.0 - upright) 

        cost = 10*distance_cost #+ 5*contact_cost + 20*orientation_cost  #+ 0.01 * ctrl_cost # - healthy_reward

        return cost
    
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T)."""
        end_effector_pos = state.xpos[self.end_effector_pos_id]
        goal_pos = state.xpos[self.goal_pos_id]
        distance_cost = jnp.sqrt(jnp.sum(jnp.square(end_effector_pos - goal_pos), axis=0))
        return 10*distance_cost

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
