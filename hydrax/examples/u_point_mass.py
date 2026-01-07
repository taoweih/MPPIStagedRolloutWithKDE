import argparse

import mujoco
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64', False)

from mujoco import mjx

from hydrax.algs import MPPI, MPPIStagedRollout, MPPIMemory
from hydrax.simulation.deterministic import run_interactive
from hydrax.tasks.u_point_mass import UPointMass


import time
from typing import Sequence
import os

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import mujoco.viewer
import numpy as np

from hydrax.alg_base import SamplingBasedController
from hydrax import ROOT
from hydrax.utils.video import VideoRecorder

from tqdm import tqdm

def _sync_tree(x):
    return jax.tree_util.tree_map(
        lambda a: a.block_until_ready() if hasattr(a, "block_until_ready") else a, x
    )


def run_interactive_visualize_discrete(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
) -> None:
    
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # bench_planner(controller, mj_model,  mj_data)

    # Initialize the controller
    mjx_model = controller.task.model
    mjx_data  = mjx.make_data(mjx_model)
    mjx_data = mjx_data.replace(
        mocap_pos=mj_data.mocap_pos, mocap_quat=mj_data.mocap_quat
    )
    policy_params = controller.init_params(initial_knots=initial_knots)
    ### Wrap this to update param stored in controller
    _jit_optimize = jax.jit(controller.optimize)
    def jit_optimize(mjx_data, policy_params, global_memory=None):
        policy_params, rollouts, rollout_states, global_memory= _jit_optimize(mjx_data, policy_params, global_memory)
        if hasattr(controller, 'params'):
            controller.params = policy_params
        return policy_params, rollouts, rollout_states, global_memory
    jit_interp_func = jax.jit(controller.interp_func)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts, _, _= jit_optimize(mjx_data, policy_params)
    policy_params, rollouts, _, _= jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
        )
        # Ensure model visual offscreen buffer is compatible with video recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)


    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )
        cost_array = []

        global_memory = 0
        if hasattr(controller, "sizes"):
            global_memory = jnp.zeros(controller.sizes())

            # def world_to_grid(x, y):
            #     j = (x - bounds[0,0]) / grid_width[0]
            #     i = (_sizes[1] - 1) - (y - bounds[1,0]) / grid_width[1]
            #     return j, i
            
            bounds = jnp.array([[-1, 1],   # x
                    [-1, 1]],  # y 
                   dtype=jnp.float32)
            
            grid_width = jnp.array([0.05, 0.05], dtype=jnp.float32) 

            _sizes = jnp.ceil((bounds[:,1] - bounds[:,0]) / grid_width).astype(int)

            end_effector_pos_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, "point_mass"
            )
            goal_pos_id = mujoco.mj_name2id(
                mj_model, mujoco.mjtObj.mjOBJ_BODY, "goal"
            ) 
            
            def grid_to_world(i, j):
                x = bounds[0,0] + (j + 0.5) * grid_width[0]
                y = bounds[1,0] + ((_sizes[1] - 1 - i) + 0.5) * grid_width[1]
                return x, y
            
            for i in range(global_memory.shape[0]):
                for j in range(global_memory.shape[1]):
                    new_data = mujoco.MjData(mj_model)
                    x, y = grid_to_world(i, j)
                    new_data.qpos[:] = [x, y]
                    mujoco.mj_forward(mj_model, new_data)
                    global_memory = global_memory.at[i, j].set(controller.task.terminal_cost(new_data))

            # hard code goal to 0
            # idx = jnp.floor((jnp.array([0, 0.8]) - bounds[:,0]) / grid_width)
            # idx = jnp.array([(_sizes[1] - 1) - idx[1],idx[0]])
            # idx = jnp.clip(idx, 0, _sizes - 1).astype(jnp.int32)
            # global_memory = global_memory.at[idx[0], idx[1]].set(0)
            # global_memory = global_memory.at[idx[0]+1, idx[1]+1].set(0)
            # global_memory = global_memory.at[idx[0]-1, idx[1]-1].set(0)

        # while viewer.is_running():
        for iter in tqdm(range(101)):
        
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )
            
            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts, rollout_states, new_global_memory= jit_optimize(mjx_data, policy_params, global_memory=global_memory)


            if iter%10 == 0 or iter in range(30,50):
                fig, ax = plt.subplots(figsize=(40,40), dpi=400)

                im = ax.imshow(global_memory, cmap="Blues", vmin=-2, vmax=200)

                for i in range(global_memory.shape[0]):
                    for j in range(global_memory.shape[1]):
                        ax.text(j, i, f"{global_memory[i, j]:.2f}",
                                ha="center", va="center", fontsize=16, color="black")

                ax.set_xticks(np.arange(-.5, global_memory.shape[1], 1), minor=True)
                ax.set_yticks(np.arange(-.5, global_memory.shape[0], 1), minor=True)
                ax.grid(which="minor", color="black", linestyle='-', linewidth=0.2)

                ax.set_xticks([])
                ax.set_yticks([])

                #param for u point
                bounds = jnp.array([[-1, 1],   # x
                    [-1, 1]],  # y 
                   dtype=jnp.float32)
                grid_width = jnp.array([0.05, 0.05], dtype=jnp.float32) 
                _sizes = jnp.ceil((bounds[:,1] - bounds[:,0]) / grid_width).astype(int) 

                def world_to_grid(x, y):
                    i = (x - bounds[0,0]) / grid_width[0] -0.5
                    j = (_sizes[1] - 1) - (((y - bounds[1,0]) / grid_width[1]) -0.5)
                    return i, j

                ## draw goal
                x, y = 0.025, 0.775
                cx, cy = world_to_grid(x, y)
                circle = patches.Circle((cx, cy), radius=0.2,
                                    facecolor="red")
                ax.add_patch(circle)

                ### walls

                # MuJoCo wall geoms:
                #   <geom name="frontwall" size="0.2 0.01 0.2" type="box" pos="0 0.20 0" rgba="0.6 0.6 0.6 0.8"/>
                #   <geom name="leftwall"  size="0.01 0.2 0.2" type="box" pos="0.21 0.01 0"  rgba="0.6 0.6 0.6 0.8"/>
                #   <geom name="rightwall" size="0.01 0.2 0.2" type="box" pos="-0.21 0.01 0" rgba="0.6 0.6 0.6 0.8"/>

                # # front
                # x, y = 0, 0.2,
                # sx, sy = 0.2, 0.01
                # cx, cy = world_to_grid(x, y)
                # width = (2 * sx) / grid_width[0]
                # height = (2 * sy) / grid_width[1]
                # front_rect = patches.Rectangle(
                #     (cx - width / 2, cy - height / 2),
                #     width, height,
                #     facecolor=(0.6, 0.6, 0.6, 0.8),
                # )
                # ax.add_patch(front_rect)

                # # left
                # x, y = 0.21, 0.01
                # sx, sy = 0.01, 0.2
                # cx, cy = world_to_grid(x, y)
                # width = (2 * sx) / grid_width[0]
                # height = (2 * sy) / grid_width[1]
                # left_rect = patches.Rectangle(
                #     (cx - width / 2, cy - height / 2),
                #     width, height,
                #     facecolor=(0.6, 0.6, 0.6, 0.8),
                # )
                # ax.add_patch(left_rect)

                # # right
                # x, y = -0.21, 0.01
                # sx, sy = 0.01, 0.2
                # cx, cy = world_to_grid(x, y)
                # width = (2 * sx) / grid_width[0]
                # height = (2 * sy) / grid_width[1]
                # right_rect = patches.Rectangle(
                #     (cx - width / 2, cy - height / 2),
                #     width, height,
                #     facecolor=(0.6, 0.6, 0.6, 0.8),
                # )
                # ax.add_patch(right_rect)

                rollout_states, nominal_trajectory_states = rollout_states
                
                # draw start location
                heuristic_states, all_states = rollout_states
                jnp_location_start = heuristic_states.squeeze(0).squeeze(0)[0]

                cx, cy = world_to_grid(jnp_location_start[0],jnp_location_start[1])

                circle = patches.Circle((cx, cy), radius=0.2,
                                    facecolor="green")
                ax.add_patch(circle)

                ## draw exploring path for updating heuristic
                states = heuristic_states.squeeze(0).squeeze(0)

                path_points = []
                for s in states:
                    cx, cy = world_to_grid(s[0],s[1])
                    path_points.append((cx, cy))

                path_points = np.array(path_points)

                ax.plot(
                    path_points[:, 0],
                    path_points[:, 1],
                    color="orange",
                    linewidth=5,
                    alpha=0.8
                )
                
                ## draw nominal (actual) trajectory
                nominal_trajectory_states = nominal_trajectory_states

                path_points = []
                for s in nominal_trajectory_states:
                    cx, cy = world_to_grid(s[0],s[1])
                    path_points.append((cx, cy))

                path_points = np.array(path_points)

                ax.plot(
                    path_points[:, 0],
                    path_points[:, 1],
                    color="purple",
                    linewidth=5,
                    alpha=0.8
                )
                
                ax.plot(path_points[:, 0],
                        path_points[:, 1],
                        'ro-', 
                        linewidth=1, markersize=3)

                ## draw all sampled paths
                all_states = all_states.squeeze(0).squeeze(0)
                for i in range(all_states.shape[0]):
                    one_trajectory_states = all_states[i]
                    path_points = []
                    for s in one_trajectory_states:
                        cx, cy = world_to_grid(s[0],s[1])
                        path_points.append((cx, cy))

                    path_points = np.array(path_points)

                    if i % 20 == 0:

                        ax.plot(
                            path_points[:, 0],
                            path_points[:, 1],
                            color="black",
                            linewidth=2,
                            alpha=0.5
                        )


                # draw path

                plt.tight_layout()
                plt.savefig(f"global_memory_grid_iteration_{iter}.png", bbox_inches="tight")
                plt.close()
            # _sync_tree(rollout_states)
   
            _sync_tree(policy_params)

            global_memory = new_global_memory

            plan_time = time.time() - plan_start

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Capture frame if recording
                if record_video and recorder.is_recording:
                    renderer.update_scene(mj_data, viewer.cam)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

            cost = controller.task.success_function(mj_data, mj_data.ctrl[:])
            cost_array.append(cost)

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            # print(
            #     f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
            #     end="\r",
            # )
        plt.figure()
        plt.plot(cost_array)
        # plt.ylim(9,13)
        plt.xlabel("Iterations")
        plt.ylabel("Current objective")
        plt.title(f'Horizon: {controller.plan_horizon}s')
        plt.show()

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()

# Need to be wrapped in main loop for async simulation
if __name__ == "__main__":
    # jax.config.update('jax_platform_name', 'cpu')

    # Define the task (cost and dynamics)
    task = UPointMass()

    def state_selection_function(state: mjx.Data) -> jax.Array:
        jnp_state = state.qpos[...,0:2]
        # jnp_state = state.xpos[...,1,0:2]
        # jnp_state = jnp_state.reshape(jnp_state.shape[0], -1)
        return jnp_state


    # Set up the controller
    # ctrl = MPPIStagedRollout(
    #     task,
    #     num_samples=512,
    #     noise_level=2.0,
    #     temperature=0.01,
    #     num_randomizations=1,
    #     plan_horizon=1.5,
    #     spline_type="zero",
    #     num_knots=16,
    #     kde_bandwidth=0.1,
    #     # state_weight=jnp.array([1,1,0])
    #     state_selection_function= state_selection_function,
    # )

    # ctrl = MPPI(
    #     task,
    #     num_samples=512,
    #     noise_level=2.0,
    #     temperature=0.01,
    #     num_randomizations=1,
    #     plan_horizon=0.2,
    #     spline_type="zero",
    #     num_knots=16,
    # )

    ctrl = MPPIMemory(
        task,
        num_samples=512,
        noise_level=2.0,
        temperature=0.0001,
        num_randomizations=1,
        plan_horizon=0.2,
        spline_type="zero",
        num_knots=16,
        kde_bandwidth=0.1,
        # state_weight=jnp.array([1,1,0])
        state_selection_function= state_selection_function,
    )
        


    # Define the model used for simulation
    mj_model = task.mj_model
    mj_model.opt.timestep = 0.01

    mj_data = mujoco.MjData(mj_model)

    run_interactive_visualize_discrete(
            ctrl,
            mj_model,
            mj_data,
            frequency=50,
            show_traces=False,
            record_video=False,
        )
