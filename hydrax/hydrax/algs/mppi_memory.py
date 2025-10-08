from typing import Literal, Tuple, Any

import jax
import jax.numpy as jnp
import math
# from jax.scipy.stats import gaussian_kde
from hydrax.utils.kde import gaussian_kde
import numpy as np

from flax.struct import dataclass

from functools import partial
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

from typing import Callable, Optional

@dataclass
class MPPIMemoryParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


class MPPIMemory(SamplingBasedController):
    """Model-predictive path integral control.

    Addition to MPPI algorithm by sampling state space during rollout and encourage 
    diverse state space exploration

    """
    def __init__(
        self,
        task: Task,
        num_samples: int,
        noise_level: float,
        temperature: float,
        num_knots_per_stage: int = 4,
        kde_bandwidth: float = 1.0,
        state_weight: jax.Array = None,
        num_randomizations: int = 1,
        risk_strategy: RiskStrategy = None,
        seed: int = 0,
        plan_horizon: float = 1.0,
        spline_type: Literal["zero", "linear", "cubic"] = "zero",
        num_knots: int = 4,
        iterations: int = 1,
        state_selection_function: Optional[Callable[[mjx.Data], jax.Array]] = None,
        ab_testing_flag: int = None,
    ) -> None:
        """Initialize the controller.

        Args:
            task: The dynamics and cost for the system we want to control.
            num_samples: The number of control sequences to sample.
            noise_level: The scale of Gaussian noise to add to sampled controls.
            temperature: The temperature parameter λ. Higher values take a more
                         even average over the samples.
            num_randomizations: The number of domain randomizations to use.
            risk_strategy: How to combining costs from different randomizations.
                           Defaults to average cost.
            seed: The random seed for domain randomization.
            plan_horizon: The time horizon for the rollout in seconds.
            spline_type: The type of spline used for control interpolation.
                         Defaults to "zero" (zero-order hold).
            num_knots: The number of knots in the control spline.
            iterations: The number of optimization iterations to perform.
            state_selection_function: Function used to select which data from mjx.Data is used in density estimation
            ab_testing_flag: Temporary flag for testing
        """
        super().__init__(
            task,
            num_randomizations=num_randomizations,
            risk_strategy=risk_strategy,
            seed=seed,
            plan_horizon=plan_horizon,
            spline_type=spline_type,
            num_knots=num_knots,
            iterations=iterations,
        )
        self.noise_level = noise_level
        self.num_samples = num_samples
        self.temperature = temperature

        self.params = None

        self.num_knots_per_stage = num_knots_per_stage
        self.kde_bandwidth = kde_bandwidth

        if state_weight is None:
            self.state_weight = jnp.array([1])
        else:
            self.state_weight = state_weight

        if state_selection_function is None:
            self.state_selection_function = lambda data: data.qpos
        else:
            self.state_selection_function = state_selection_function

        self.bounds = jnp.array([[-1.0, 1.0],   # x
                    [-1.0, 1.0]],  # y 
                   dtype=jnp.float32)
        self.grid_width = jnp.array([0.005, 0.005], dtype=jnp.float32) 

        self._sizes = jnp.ceil((self.bounds[:,1] - self.bounds[:,0]) / self.grid_width).astype(int) 

        self.ab_testing_flag = ab_testing_flag

    def sizes(self):
        return self._sizes

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIMemoryParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        self.params = MPPIMemoryParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)
        return MPPIMemoryParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def state_selection_function(self, data: mjx.Data):
        return self.state_selection_function(data)
    
    def terminal_cost(self, state: mjx.Data, global_memory):
        if global_memory is None:
            return self.task.terminal_cost(state), global_memory
        else:
            jnp_state = self.state_selection_function(state)
            heuristic_cost = self.heuristic_cost(jnp_state, global_memory)
            default_cost = self.task.terminal_cost(state)
            # jax.debug.print("hcost:{}",heuristic_cost)
            new_cost = jnp.where(heuristic_cost==0, default_cost,heuristic_cost)
            global_memory_updated = self.update_heuristic(global_memory, jnp_state, new_cost)
            heuristic_cost = self.heuristic_cost(jnp_state, global_memory_updated)
            return heuristic_cost, global_memory_updated
            
    
    def heuristic_cost(self, state:jax.Array, global_memory):
        if global_memory is None:
            return 0
        else:
            idx = jnp.floor((state - self.bounds[:,0]) / self.grid_width)
            idx = jnp.clip(idx, 0, self._sizes - 1).astype(jnp.int32)
            heuristic_cost = global_memory[idx[0],idx[1]]
            return heuristic_cost
        
    def update_heuristic(self, global_memory, state:jax.Array, value):
        if global_memory is None:
            return None
        else:
            idx = jnp.floor((state - self.bounds[:,0]) / self.grid_width)
            idx = jnp.clip(idx, 0, self._sizes - 1).astype(jnp.int32)

            _ = jax.lax.cond(
                global_memory[idx[0], idx[1]].astype(jnp.int32) != value.astype(jnp.int32),
                # true branch: print, then return a dummy JAX scalar
                lambda op: (jax.debug.print("updated from {} to {}", op[0], op[1]), jnp.array(0, dtype=jnp.int32))[1],
                # false branch: just return the same-shaped dummy
                lambda op: jnp.array(0, dtype=jnp.int32),
                (global_memory[idx[0], idx[1]], value),
            )   

            global_memory = global_memory.at[idx[0],idx[1]].max(value)
            return global_memory

    def sample_knots(self, params: MPPIMemoryParams) -> Tuple[jax.Array, MPPIMemoryParams]:
        """Sample a control sequence."""
        rng, sample_rng = jax.random.split(params.rng)
        noise = jax.random.normal(
            sample_rng,
            (
                self.num_samples,
                #self.num_knots,
                params.mean.shape[0],
                self.task.model.nu,
            ),
        )
        controls = params.mean + self.noise_level * noise
        return controls, params.replace(rng=rng)

    def update_params(
        self, params: MPPIMemoryParams, rollouts: Trajectory
    ) -> MPPIMemoryParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
    
    def optimize(self, state: mjx.Data, params: Any, global_memory: jax.Array = None, valid_count:int = None) -> Tuple[Any, Trajectory]:
        """Perform an optimization step to update the policy parameters.

        Args:
            state: The initial state x₀.
            params: The current policy parameters, U ~ π(params).

        Returns:
            Updated policy parameters
            Rollouts used to update the parameters
        """
        # Warm-start spline by advancing knot times by sim dt, then recomputing
        # the mean knots by evaluating the old spline at those times
        tk = params.tk
        new_tk = (
            jnp.linspace(0.0, self.plan_horizon, self.num_knots) + state.time
        )
        new_mean = self.interp_func(new_tk, tk, params.mean[None, ...])[0]
        params = params.replace(tk=new_tk, mean=new_mean)

        def _optimize_scan_body(carry, _):
            # Sample random control sequences from spline knots
            params, global_memory = carry
            knots, params = self.sample_knots(params)
            knots = jnp.clip(
                knots, self.task.u_min, self.task.u_max
            )  # (num_rollouts, num_knots, nu)

            # Roll out the control sequences, applying domain randomizations and
            # combining costs using self.risk_strategy.
            rng, dr_rng = jax.random.split(params.rng)
            rollouts, rollout_states, global_memory= self.rollout_with_randomizations(
                state, new_tk, knots, dr_rng, global_memory, valid_count
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(params, rollouts)

            return (params,global_memory), (rollouts, rollout_states)

        (params,global_memory), (rollouts, rollout_states) = jax.lax.scan(
            f=_optimize_scan_body, init=(params,global_memory), xs=jnp.arange(self.iterations)
        )

        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        # if global_memory is not None:
        #     jax.debug.print("all memory: {}", jnp.sum(global_memory))

        return params, rollouts_final, rollout_states, global_memory

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
        global_memory: jax.Array = None,
        valid_count: int = None
    ) -> Trajectory:
        """Compute rollout costs, applying domain randomizations.

        Args:
            state: The initial state x₀.
            tk: The knot times of the control spline, (num_knots,).
            knots: The control spline knots, (num rollouts, num_knots, nu).
            rng: The random number generator key for randomizing initial states.

        Returns:
            A Trajectory object containing the control, costs, and trace sites.
            Costs are aggregated over domains using the given risk strategy.
        """
        # Set the initial state for each rollout.
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )

        if self.num_randomizations > 1:
            # Randomize the initial states for each domain randomization
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        # compute the control sequence from the knots
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)  # (num_rollouts, H, nu)

        # Apply the control sequences, parallelized over both rollouts and
        # domain randomizations.
        rollout_states, rollouts, global_memory_batch = jax.vmap(
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None, None, None)
        )(self.model, states, controls, knots, global_memory, valid_count)

        if global_memory is not None:
            global_memory = jnp.max(global_memory_batch,axis=0)

        # Combine the costs from different domain randomizations using the
        # specified risk strategy.
        costs = self.risk_strategy.combine_costs(rollouts.costs)
        controls = rollouts.controls[0]  # identical over randomizations
        knots = rollouts.knots[0]  # identical over randomizations
        trace_sites = rollouts.trace_sites[0]  # visualization only, take 1st
        return rollouts.replace(
            costs=costs, controls=controls, knots=knots, trace_sites=trace_sites
        ), rollout_states, global_memory
    
    def eval_rollouts(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
        global_memory: jax.Array = None,
        valid_count: int = None,
    ) -> Tuple[mjx.Data, Trajectory]:
        """Rollout control sequences (in parallel) and compute the costs.

        Args:
            model: The mujoco dynamics model to use.
            state: The initial state x₀.
            controls: The control sequences, (num rollouts, H, nu).
            knots: The control spline knots, (num rollouts, num_knots, nu).

        Returns:
            The states (stacked) experienced during the rollouts.
            A Trajectory object containing the control, costs, and trace sites.
        """

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)  # step model + compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            # cost = cost + self.density_cost(x, kde_memory, valid_count)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)
        
        @partial(jax.vmap, in_axes=(0, 0))
        def _rollout_fn(
           x: mjx.Data, u: jax.Array
        )-> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            '''Batched version of _scan_fn'''
            final_state, (states, costs, trace_sites) =jax.lax.scan(
                _scan_fn,  x, u
            )
            return final_state, (states, costs, trace_sites)
        
        #### rollout and resample start ####

        ## Initilize full states and costs that will be updated after each stage
        states = jax.tree_util.tree_map(lambda x: jnp.zeros((self.num_samples, self.ctrl_steps)+x.shape, dtype=x.dtype),state)
        costs = jnp.zeros((self.num_samples, self.ctrl_steps))
        trace_sites = jnp.zeros((self.num_samples, self.ctrl_steps) + self.task.get_trace_sites(state).shape)

        # Calculate some parameters for ease of use
        num_stages = int(math.floor(self.num_knots / self.num_knots_per_stage))
        timesteps_per_stage = int(math.floor(self.ctrl_steps / self.num_knots))*self.num_knots_per_stage

        # batch init state 
        curr_state = jax.tree_util.tree_map((lambda x: jnp.repeat(x[None, ...], self.num_samples, axis=0)), state)

        for n in range(num_stages-1):
            # partial rollout
            partial_controls = controls[:,n*timesteps_per_stage:(n+1)*timesteps_per_stage,:]
            latest_state, (partial_states, partial_costs, partial_trace_sites) = _rollout_fn(curr_state, partial_controls)
            costs = costs.at[:,n*timesteps_per_stage:(n+1)*timesteps_per_stage].set(partial_costs)
            trace_sites = trace_sites.at[:,n*timesteps_per_stage:(n+1)*timesteps_per_stage].set(partial_trace_sites)
            states = jax.tree_util.tree_map(lambda x, new: x.at[:, n*timesteps_per_stage:(n+1)*timesteps_per_stage,...].set(new),states, partial_states)

            # resampling indices
            jnp_latest_state = jax.vmap(self.state_selection_function)(latest_state)
            weight = self.state_weight
            jnp_latest_state = weight * jnp_latest_state
            kde = gaussian_kde(jnp_latest_state.T,bw=self.kde_bandwidth) # scipy kde expect data dimension to be first and batch dimension to be second

            p_x = kde.pdf(jnp_latest_state.T)
            epsilon = 1e-6
            inv_px = (1.0 / p_x + epsilon)
            inv_px = inv_px / inv_px.sum()
            
            indices = jax.random.categorical(jax.random.PRNGKey(0),jnp.log(inv_px),shape=(self.num_samples,))

            # reorder things around (only need to reorder up to current steps but won't matter since the later ones will be overwritten)
            states = jax.tree_util.tree_map(lambda x: x[indices,...], states)
            controls = controls[indices,...]
            knots = knots[indices,...]
            costs = costs[indices,...]
            trace_sites = trace_sites[indices,...]

            curr_state = jax.tree_util.tree_map(lambda x: x[:,-1,...], partial_states)
            curr_state = jax.tree_util.tree_map(lambda x: x[indices,...], curr_state)

            # sample new knots, update controls
            partial_param = self.params.replace(mean= self.params.mean[(n+1)*self.num_knots_per_stage:,:])
            sampled_partial_knots, _ = self.sample_knots(partial_param)
            sampled_partial_knots = jnp.clip(
                sampled_partial_knots, self.task.u_min, self.task.u_max
            )
            knots = knots.at[:,(n+1)*self.num_knots_per_stage:,:].set(sampled_partial_knots)
            tk = partial_param.tk
            tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
            controls = self.interp_func(tq, tk, knots)

        # rollout remaining control
        partial_controls = controls[:,(num_stages-1)*timesteps_per_stage:,:]
        final_state, (partial_states, partial_costs, partial_trace_sites) = _rollout_fn(curr_state, partial_controls)
        costs = costs.at[:,(num_stages-1)*timesteps_per_stage:].set(partial_costs)
        trace_sites = trace_sites.at[:,(num_stages-1)*timesteps_per_stage:].set(partial_trace_sites)
        states = jax.tree_util.tree_map(lambda x, new: x.at[:,(num_stages-1)*timesteps_per_stage:,...].set(new),states, partial_states)

        #### rollout and resample end ####
        final_cost, global_memory_batch= jax.vmap(self.terminal_cost,in_axes=[0,None])(final_state, global_memory)
        if global_memory is not None:
            global_memory = jnp.max(global_memory_batch,axis=0)

        # jnp_final_state = jax.vmap(self.state_selection_function)(final_state)
        # final_cost = final_cost + jax.vmap(self.heuristic_cost,in_axes=(0, None))(jnp_final_state, global_memory) # add heristic to final cost

        final_trace_sites = jax.vmap(self.task.get_trace_sites)(final_state)

        costs = jnp.append(costs, final_cost[:,None], axis=1)
        trace_sites = jnp.append(trace_sites, final_trace_sites[:,None], axis=1)

        def _fori_fn(
            i, carry
        ):
            global_memory, states, cumsum_costs = carry
            assert cumsum_costs.shape[0] == states.shape[0]
            new_h_value = cumsum_costs[i]
            global_memory = self.update_heuristic(global_memory, states[i], new_h_value)
            return (global_memory, states, cumsum_costs)

        sum_cost = jnp.sum(costs, axis=1)
        min_idx = jnp.argmin(final_cost)
        new_h_value = sum_cost[min_idx]
        global_memory = self.update_heuristic(global_memory,self.state_selection_function(state),new_h_value)

        jnp_states = jax.vmap(self.state_selection_function)(states)
        best_trajectory_states = jnp_states[min_idx]
        best_trajectory_costs = costs[min_idx][:-1]

        cumsum_costs = jnp.cumsum(best_trajectory_costs[::-1])[::-1]

        global_memory, _, _ = jax.lax.fori_loop(0,cumsum_costs.shape[0], _fori_fn, (global_memory, best_trajectory_states, cumsum_costs))

        return states, Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        ), global_memory
