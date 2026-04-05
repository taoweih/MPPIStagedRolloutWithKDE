from typing import Literal, Tuple, Any

import jax
import jax.numpy as jnp
import math

from hydrax.utils.kde import gaussian_kde
import numpy as np

from flax.struct import dataclass

from functools import partial
from mujoco import mjx

from hydrax.alg_base import SamplingBasedController, SamplingParams, Trajectory
from hydrax.risk import RiskStrategy
from hydrax.task_base import Task

from typing import Callable, Optional

from flax import nnx

class NeuralNet(nnx.Module):
    def __init__(self, 
                 din=2, 
                 dmid=64, 
                 dout=1, 
                 use_hash_grid=True, 
                 grid_min = -10.0,
                 grid_max = 10.0,
                 rngs: nnx.Rngs = nnx.Rngs(0)):
        
        self.din = din
        self.use_hash_grid = use_hash_grid

        self.grid_min = grid_min
        self.grid_max = grid_max

        #### Positional encoding ####
        self.num_freqs = 12
        
        #### Hash Grid Config ########
        self.num_levels = 16
        self.table_size = 4096
        self.features_per_level = 2

        if self.use_hash_grid:
            self.resolutions = jnp.exp(
                jnp.linspace(jnp.log(16), jnp.log(2048), self.num_levels)
            )

            self.embeddings = nnx.Param(
                jax.random.uniform(rngs.params(), (self.num_levels, self.table_size, self.features_per_level)) * 1e-4
            )

            if din == 2:
                self.primes = jnp.array([1, 2654435761], dtype=jnp.uint32)
            elif din == 3:
                self.primes = jnp.array([1, 2654435761, 805459861], dtype=jnp.uint32)
            else:
                raise ValueError(f"din={din} not supported; use 2 or 3.")

            self.offsets = jnp.array(
                [[int(b) for b in format(i, f'0{din}b')] for i in range(2 ** din)],
                dtype=jnp.int32,
            )
            
            input_dim = self.num_levels * self.features_per_level
            
        else:
            input_dim = self.din * self.num_freqs * 2

        self.linear1 = nnx.Linear(input_dim, dmid, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.linear3 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.linear4 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.linear_out = nnx.Linear(dmid, dout, rngs=rngs)

    def __call__(self, x):
        is_unbatched = x.ndim == 1
        if is_unbatched:
            x = x[None, ...] 
        
        if self.use_hash_grid:
            # x_norm = (x + 1.0) * 0.5 # shift grid from [-1,1] to [0,1]
            x_norm = (x - self.grid_min) / (self.grid_max - self.grid_min)
            
            def process_level(embedding_subtable, resolution, x_in):
                x_grid = x_in * resolution
                x0 = jnp.floor(x_grid).astype(jnp.int32)
                w = x_grid - x0
                
                grid_coords = x0[:, None, :] + self.offsets[None, :, :]
                hashed_indices = ((grid_coords.astype(jnp.uint32) * self.primes).sum(axis=-1)) % self.table_size
                corners = embedding_subtable[hashed_indices]

                off_f = self.offsets[None, :, :].astype(x_in.dtype)
                per_dim = 1.0 - off_f + w[:, None, :] * (2.0 * off_f - 1.0)
                corner_weights = jnp.prod(per_dim, axis=-1)
                value = jnp.einsum('bc,bcf->bf', corner_weights, corners)

                return value

            features = jax.vmap(process_level, in_axes=(0, 0, None))(
                self.embeddings.value, 
                self.resolutions, 
                x_norm
            )
            features = jnp.transpose(features, (1, 0, 2))
            batch_size = features.shape[0]
            encoded_input = features.reshape(batch_size, -1)
            
        else:
            freqs = 2.0 ** jnp.arange(self.num_freqs)
        
            inputs = x[..., None] * freqs # (B, din, num_freqs)
        
            sin_inputs = jnp.sin(inputs * jnp.pi)
            cos_inputs = jnp.cos(inputs * jnp.pi)
        
            encoded_input = jnp.concatenate([sin_inputs, cos_inputs], axis=-1)
            encoded_input = encoded_input.reshape(x.shape[0], -1)


        ##### MLP pass #####
        x_out = nnx.swish(self.linear1(encoded_input))
        x_out = nnx.swish(self.linear2(x_out))
        # x_out = nnx.swish(self.linear3(x_out))
        # x_out = nnx.swish(self.linear4(x_out))
        output = self.linear_out(x_out)

        if is_unbatched:
            return output.squeeze(0)
            
        return output

@dataclass
class MPPIMemoryContinuousParams(SamplingParams):
    """Policy parameters for model-predictive path integral control.

    Same as SamplingParams, but with a different name for clarity.

    Attributes:
        tk: The knot times of the control spline.
        mean: The mean of the control spline knot distribution, μ = [u₀, ...].
        rng: The pseudo-random number generator key.
    """


class MPPIMemoryContinuous(SamplingBasedController):
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
        din: int = 2,
        use_hash_grid: bool = True,
        grid_min = -10.0,
        grid_max = 10.0,
        online_learning_rate = 1e-3,
        goal_position = jnp.array([[0, 0]]),
        goal_weight   = 200000.0,
        num_anchors = 10000,
        new_weight  = 1000.0,
        heuristic_discount_factor: float = 0.99,
        use_staged_rollout: bool = False,
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

        self.din = din
        self.grid_min = grid_min
        self.grid_max = grid_max
        self.use_hash_grid = use_hash_grid
        model = NeuralNet(din=self.din, dout=1, use_hash_grid=self.use_hash_grid, grid_min=self.grid_min, grid_max=self.grid_max)
        self.graphdef, self.nn_weight_template, self.static_state= nnx.split(model, nnx.Param,...)
        self.nn_learning_rate = online_learning_rate
        self.num_anchors = num_anchors
        self.goal_weight = goal_weight
        self.goal_position = goal_position
        self.new_weight = new_weight
        self.heuristic_discount_factor = heuristic_discount_factor
        self.use_staged_rollout = use_staged_rollout

    def sizes(self):
        return self._sizes

    def init_params(
        self, initial_knots: jax.Array = None, seed: int = 0
    ) -> MPPIMemoryContinuousParams:
        """Initialize the policy parameters."""
        _params = super().init_params(initial_knots, seed)
        self.params = MPPIMemoryContinuousParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)
        return MPPIMemoryContinuousParams(tk=_params.tk, mean=_params.mean, rng=_params.rng)

    def state_selection_function(self, data: mjx.Data):
        return self.state_selection_function(data)
    
    def terminal_cost(self, state: mjx.Data, global_memory):
        if global_memory is None:
            return self.task.terminal_cost(state)#, global_memory
        else:
            jnp_state = self.state_selection_function(state)
            heuristic_cost = self.heuristic_cost(jnp_state, global_memory)
            default_cost = self.task.terminal_cost(state)
            return heuristic_cost
            # return default_cost      
    
    def heuristic_cost(self, state:jax.Array, global_memory):
        if global_memory is None:
            return 0
        else:
            model = nnx.merge(self.graphdef, global_memory, self.static_state)
            return model(state).squeeze() 
        
    def update_heuristic(self, global_memory, state: jax.Array, value: jax.Array):
        if global_memory is None:
            return None
        
        model = nnx.merge(self.graphdef, global_memory, self.static_state)

        if state.ndim == 1:
            state = state[None, ...]
            value = value[None, ...] 

        rng = jax.random.PRNGKey(42) 
        num_anchors = self.num_anchors
        anchor_states = jax.random.uniform(rng, (num_anchors, self.din), minval=self.grid_min, maxval=self.grid_max)
        anchor_targets = model(anchor_states).squeeze()

        # hard set goal state
        goal_state = self.goal_position
        goal_target = jnp.array([0.0]) 
        goal_weight = jnp.array([self.goal_weight])

        all_states = jnp.concatenate([state, anchor_states, goal_state], axis=0)
        all_targets = jnp.concatenate([value, anchor_targets, goal_target], axis=0)
        # all_states = state
        # all_targets = value

        B_new = state.shape[0]
        B_anchor = num_anchors
        weights = jnp.concatenate([jnp.ones(B_new) * self.new_weight, jnp.ones(B_anchor) * 1.0, goal_weight], axis=0)
        # weights = jnp.ones(B_new)

        def loss_fn(m, x, y, w):
            pred = m(x).squeeze()
            error =  jnp.maximum(y - pred, 0.0) ** 2#(pred - y) ** 2
            return jnp.mean(error * w)
        
        grad_fn = nnx.grad(loss_fn)
        grads = grad_fn(model, all_states, all_targets, weights)
        _, weight_grads, _ = nnx.split(grads, nnx.Param, ...)

        def update_rule(path, weight, grad):
            # using positional encoding
            if not self.use_hash_grid:
                return weight - self.nn_learning_rate*0.01 * grad

            # using hashgrid 
            def get_name(k):
                return getattr(k, 'name', getattr(k, 'key', str(k)))

            is_embedding = any(get_name(k) == 'embeddings' for k in path)
            
            if is_embedding:
                return weight - self.nn_learning_rate * grad 
            else:
                return weight - (self.nn_learning_rate * 0) * grad

        new_global_memory = jax.tree_util.tree_map_with_path(
            update_rule, 
            global_memory, 
            weight_grads
        )

        return new_global_memory
        # return global_memory

    def sample_knots(self, params: MPPIMemoryContinuousParams) -> Tuple[jax.Array, MPPIMemoryContinuousParams]:
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
        self, params: MPPIMemoryContinuousParams, rollouts: Trajectory
    ) -> MPPIMemoryContinuousParams:
        """Update the mean with an exponentially weighted average."""
        costs = jnp.sum(rollouts.costs, axis=1)  # sum over time steps
        # N.B. jax.nn.softmax takes care of details like baseline subtraction.
        weights = jax.nn.softmax(-costs / self.temperature, axis=0)
        mean = jnp.sum(weights[:, None, None] * rollouts.knots, axis=0)
        return params.replace(mean=mean)
    
    def optimize(self, state: mjx.Data, params: Any, global_memory: nnx.State = None) -> Tuple[Any, Trajectory]:
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
                state, new_tk, knots, dr_rng, global_memory
            )
            params = params.replace(rng=rng)

            # Update the policy parameters based on the combined costs
            params = self.update_params(params, rollouts)

            return (params,global_memory), (rollouts, rollout_states)

        (params,global_memory), (rollouts, rollout_states) = jax.lax.scan(
            f=_optimize_scan_body, init=(params,global_memory), xs=jnp.arange(self.iterations)
        )

        ## rollout once for visualization of current control trajectory
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)

        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, params.mean[None, ...])[0]

        def _scan_fn(
            x: mjx.Data, u: jax.Array
        ) -> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            """Compute the cost and observation, then advance the state."""
            x = x.replace(ctrl=u)
            x = mjx.step(self.model, x)  # step model + compute site positions
            cost = self.dt * self.task.running_cost(x, u)
            # cost = cost + self.density_cost(x, kde_memory, valid_count)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, sites)
        
        def _rollout_fn(
           x: mjx.Data, u: jax.Array
        )-> Tuple[mjx.Data, Tuple[mjx.Data, jax.Array, jax.Array]]:
            '''Batched version of _scan_fn'''
            final_state, (states, costs, trace_sites) =jax.lax.scan(
                _scan_fn,  x, u
            )
            return final_state, (states, costs, trace_sites)
        
        _, (nominal_trajectory_states, _, _) = _rollout_fn(state, controls)

        return params, rollouts_final, (rollout_states, self.state_selection_function(nominal_trajectory_states)), global_memory

    def rollout_with_randomizations(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
        global_memory: nnx.State = None,
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
            self.eval_rollouts, in_axes=(self.randomized_axes, 0, None, None, None)
        )(self.model, states, controls, knots, global_memory)

        if global_memory is not None:
            # global_memory = jnp.max(global_memory_batch,axis=0)
            # jax.debug.print("global_memory_batch shape:{}", global_memory_batch.shape)
            global_memory = jax.tree_util.tree_map(lambda x: jnp.max(x,axis=0), global_memory_batch)
            # global_memory = global_memory_batch

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
        global_memory: nnx.State = None,
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
        
        if self.use_staged_rollout:
        
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
            final_cost = jax.vmap(self.terminal_cost,in_axes=[0,None])(final_state, global_memory)
            final_trace_sites = jax.vmap(self.task.get_trace_sites)(final_state)

            costs = jnp.append(costs, final_cost[:,None], axis=1)
            trace_sites = jnp.append(trace_sites, final_trace_sites[:,None], axis=1)

        else:
            states = jax.tree_util.tree_map((lambda x: jnp.repeat(x[None, ...], self.num_samples, axis=0)), state)

            final_state, (states, costs, trace_sites) = _rollout_fn(states,controls)
            
            final_cost = jax.vmap(self.terminal_cost,in_axes=[0,None])(final_state, global_memory)
            final_trace_sites = jax.vmap(self.task.get_trace_sites)(final_state)
            costs = jnp.append(costs, final_cost[:,None],axis=1)
            trace_sites = jnp.append(trace_sites, final_trace_sites[:,None], axis=1)

        B = states.xpos.shape[0]
        # conver state to (B, T, state.sahpe)
        state = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (B,) + x.shape), state)
        state = jax.tree_util.tree_map(lambda x: x[:, None, ...],  state)

        states = jax.tree_util.tree_map(lambda x0, x1: jnp.concatenate([x0, x1], axis=1), state, states) #append initial state to full states

        # helper function for updating heuristic
        def _fori_fn( 
            i, carry
        ):
            global_memory, states, cumsum_costs = carry
            assert cumsum_costs.shape[0] == states.shape[0]
            new_h_value = cumsum_costs[i]
            global_memory = self.update_heuristic(global_memory, states[i], new_h_value)
            return (global_memory, states, cumsum_costs)

        # find idx of trajectory with minimal costs
        sum_cost = jnp.sum(costs, axis=1)
        min_idx = jnp.argmin(sum_cost)

        # only update heuristic for initial state (true RTAA* style)
        jnp_states = jax.vmap(self.state_selection_function)(states)
        new_h_value = sum_cost[min_idx]
        initial_state = jnp_states[min_idx][0]  # first state of best trajectory
        global_memory = self.update_heuristic(global_memory, initial_state, new_h_value)

        # update heuristic along lowest cost trajectory
        # jnp_states = jax.vmap(self.state_selection_function)(states)
        # best_trajectory_states = jnp_states[min_idx]
        # best_trajectory_costs = costs[min_idx]
        # Original undiscounted cumsum:
        # cumsum_costs = jnp.cumsum(best_trajectory_costs[::-1])[::-1]
        # Discounted cumsum: G_t = c_t + gamma*c_{t+1} + gamma^2*c_{t+2} + ...
        # Bounded by c_max/(1-gamma) regardless of horizon length
        # def _discounted_cumsum_fn(carry, cost):
        #     return cost + self.heuristic_discount_factor * carry, cost + self.heuristic_discount_factor * carry
        # _, cumsum_costs = jax.lax.scan(_discounted_cumsum_fn, 0.0, best_trajectory_costs[::-1])
        # cumsum_costs = cumsum_costs[::-1]
        # global_memory, _ , _ = jax.lax.fori_loop(0,cumsum_costs.shape[0], _fori_fn, (global_memory,best_trajectory_states[::-1], cumsum_costs[::-1]))

        best_trajectory_states = jnp_states[min_idx]

        return (best_trajectory_states, jnp_states), Trajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
        ), global_memory
