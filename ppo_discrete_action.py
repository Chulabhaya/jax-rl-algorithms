"""An end-to-end JAX implementation of PPO designed for discrete actions, based on a combination of CleanRL's and PureJaxRL's styles of algorithm implementations."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, NamedTuple, Tuple

import chex
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tyro
from flax.linen.initializers import constant, orthogonal
from flax.training import orbax_utils
from flax.training.train_state import TrainState
from gymnax.environments import environment
from jax import config  # noqa F401

import wandb
from wrappers import FlattenObservationWrapper, LogWrapper

# config.update("jax_disable_jit", True)  # Uncomment for debugging


@dataclass
class Args:
    """Arguments for running discrete-action PPO."""

    # General arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """Name of this experiment."""
    seed: int = 1
    """Seed of the experiment."""

    # Weights & Biases specific arguments
    track: bool = True
    """If toggled, this experiment will be tracked with Weights and Biases."""
    wandb_project: str = "test3"
    """"W&B project name."""
    wandb_dir: str = "./"
    """"W&B project directory."""
    wandb_entity: str = None
    """W&B entity (team) name."""
    wandb_group: str = None
    """W&B group name."""
    wandb_job_type: str = None
    """W&B job type name, for categorizing runs in a group."""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """ID of the environment."""
    total_timesteps: int = 100000
    """Total timesteps of the experiment."""
    learning_rate: float = 2.5e-4
    """Learning rate of the optimizer."""
    num_envs: int = 4
    """Number of parallel game environments."""
    num_steps: int = 128
    """Number of steps to run in each environment per policy rollout."""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks."""
    gamma: float = 0.99
    """Discount factor gamma."""
    gae_lambda: float = 0.95
    """Lambda for the general advantage estimation."""
    num_minibatches: int = 4
    """Number of minibatches."""
    update_epochs: int = 4
    """K epochs to update the policy."""
    norm_adv: bool = True
    """Toggle advantage normalization."""
    clip_coef: float = 0.2
    """Surrogate clipping coefficient."""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function."""
    ent_coef: float = 0.01
    """Coefficient of the entropy."""
    vf_coef: float = 0.5
    """Coefficient of the value function."""
    max_grad_norm: float = 0.5
    """Maximum norm for the gradient clipping."""
    target_kl: float = None
    """Target KL divergence threshold."""

    # Checkpointing specific arguments
    save: bool = True
    """Automatic tuning of the entropy coefficient."""
    save_checkpoint_dir: str = (
        "/home/chulabhaya/phd/research/jax-rl-algorithms/checkpoints"
    )
    """Path to directory to save checkpoints in."""
    checkpoint_interval: int = 25
    """How often to save checkpoints during training (in training iterations)."""
    resume: bool = False
    """Whether to resume training from a checkpoint."""
    resume_checkpoint_path: str = (
        "/home/chulabhaya/phd/research/jax-rl-algorithms/checkpoints/ppo_discrete_action_ozrrr13s/51200"
    )
    """Path to checkpoint to resume training from."""
    run_id: str = None
    """W&B unique run id for resuming."""

    # To be filled in runtime
    batch_size: int = 0
    """Batch size (computed in runtime)."""
    minibatch_size: int = 0
    """Mini-batch size (computed in runtime)."""
    num_iterations: int = 0
    """Number of iterations (computed in runtime)."""


def save(
    run_id: str,
    checkpoint_dir: str,
    timestep: int,
    runner_state: Tuple[
        TrainState,
        TrainState,
        dict,
        environment.EnvState,
        chex.Array,
        chex.PRNGKey,
    ],
) -> None:
    """Saves checkpoints.

    Parameters
    ----------
    run_id : str
        Run ID.
    checkpoint_dir : str
        Directory to store checkpoints in.
    timestep : int
        Timestep of checkpoint.
    runner_state : Tuple[ TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey, ]
        Training state.
    """
    # Set up directory for saving data
    save_dir = Path(checkpoint_dir) / run_id
    save_dir.mkdir(parents=True, exist_ok=True)
    os.chmod(
        save_dir, 0o777
    )  # Prevent permission issues when writing to this directory after resuming a training job

    # Save data
    checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(runner_state)
    checkpoint_path = save_dir / Path(str(timestep))
    print(f"Saving checkpoint: {checkpoint_path}", flush=True)
    checkpointer.save(
        checkpoint_path,
        runner_state,
        save_args=save_args,
        force=True,
    )


def make_env(env_id: str) -> environment.Environment:
    """Initializes Gymnax environment.

    Parameters
    ----------
    env_id : str
        ID for the Gymnax environment.
    seed : int
        Seed.

    Returns
    -------
    Environment
        Gymnax environment.
    """
    env, env_params = gymnax.make(env_id)
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    return env, env_params


class Critic(nn.Module):
    """Model for representing the PPO critic."""

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        Computes state-values for input observations.

        Returns
        -------
        state_values : chex.Array
            Computed state-values.
        """
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            obs
        )
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        state_values = nn.Dense(1, kernel_init=orthogonal(1), bias_init=constant(0.0))(
            x
        )
        return state_values


class Actor(nn.Module):
    """Model for representing the actor.

    Attributes
    ----------
    action_dim : int
        Action dimensionality for output of network.
    """

    action_dim: int

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        Computes action logits for input observations.

        Returns
        -------
        action_logits : chex.Array
            Computed action logits.
        """
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            obs
        )
        x = nn.tanh(x)
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        action_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(x)
        return action_logits


def sample_actions(
    logits: chex.Array,
    sampling_key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Calculates probabilities based on sampled actions.

    Parameters
    ----------
    logits : chex.Array
        Action logits output from actor network.
    sampling_key : chex.PRNGKey
        RNG key.

    Returns
    -------
    actions : chex.Array
        Sampled actions from action distributions.
    log_action_probs : chex.Array
        Logs of action probabilities for sampled actions.
    entropy : chex.Array
        Entropies of action distributions.
    """
    # Directly use logits to sample actions
    actions = jax.random.categorical(sampling_key, logits)

    # Calculate action probabilities for return or other uses
    action_probs = jax.nn.softmax(logits)

    # Compute log probabilities using log_softmax for numerical stability
    log_action_probs = jax.nn.log_softmax(logits)
    selected_log_action_probs = log_action_probs[
        jnp.arange(log_action_probs.shape[0]), actions
    ]

    # Calculate entropy
    entropy = (-(action_probs * log_action_probs)).sum(1)

    return actions, selected_log_action_probs, entropy


def give_actions(
    logits: chex.Array,
    actions: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Calculates probabilities based on given actions.

    Parameters
    ----------
    logits : chex.Array
        Action logits output from actor network.
    actions : chex.Array
        Given input actions to use instead of sampling

    Returns
    -------
    actions : chex.Array
        Return input given actions.
    log_action_probs : chex.Array
        Logs of action probabilities for given actions.
    entropy : chex.Array
        Entropies of action distributions.
    """
    # Calculate action probabilities for return or other uses
    action_probs = jax.nn.softmax(logits)

    # Compute log probabilities using log_softmax for numerical stability
    log_action_probs = jax.nn.log_softmax(logits)
    selected_log_action_probs = log_action_probs[
        jnp.arange(log_action_probs.shape[0]), actions
    ]

    # Calculate entropy
    entropy = (-(action_probs * log_action_probs)).sum(1)

    return actions, selected_log_action_probs, entropy


class Transition(NamedTuple):
    """Stores a transition of data."""

    obs: chex.Array
    action: chex.Array
    reward: chex.Array
    done: chex.Array
    log_prob: chex.Array
    value: chex.Array
    info: chex.Array


def make_train(args: Args) -> Callable[[chex.PRNGKey], dict]:
    """Initializes training.

    Parameters
    ----------
    args : Args
        Parameters used for initializing training function.

    Returns
    -------
    Callable[[chex.PRNGKey], dict]
        Train function.
    """
    # Initialize batch size, minibatch size, and number of training iterations
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

    # Set up environment
    env, env_params = make_env(args.env_id)

    # Set up learning rate scheduler
    def linear_schedule(iteration: chex.Array) -> chex.Array:
        """Linear scheduling function for learning rate.

        Parameters
        ----------
        iteration : chex.Array
            Training iteration.

        Returns
        -------
        chex.Array
            Annealed learning rate.
        """
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = (
            1.0
            - (iteration // (args.num_minibatches * args.update_epochs))
            / args.num_iterations
        )
        return args.learning_rate * frac

    # Set up training function
    def train(key: chex.PRNGKey) -> dict:
        """Performs training.

        Parameters
        ----------
        key : chex.PRNGKey
            RNG key.

        Returns
        -------
        dict
            Metrics from the end of training.
        """
        # Initialize critic
        init_obs = jnp.zeros(env.observation_space(env_params).shape)
        vf = Critic()
        key, critic_key = jax.random.split(key)
        vf_state = TrainState.create(
            apply_fn=vf.apply,
            params=vf.init(critic_key, init_obs),
            tx=optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=(
                        linear_schedule if args.anneal_lr else args.learning_rate
                    ),
                    eps=1e-5,
                ),
            ),
        )
        # Initialize actor
        actor = Actor(action_dim=env.action_space(env_params).n)
        key, actor_key = jax.random.split(key)
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, init_obs),
            tx=optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.inject_hyperparams(optax.adam)(
                    learning_rate=(
                        linear_schedule if args.anneal_lr else args.learning_rate
                    ),
                    eps=1e-5,
                ),
            ),
        )

        # Initialize environment
        key, env_key = jax.random.split(key)
        env_reset_key = jax.random.split(env_key, args.num_envs)
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(
            env_reset_key, env_params
        )  # Vectorizes reset function

        # Step function representing a training iteration
        @jax.jit
        def _update_step(
            runner_state: Tuple[
                TrainState,
                TrainState,
                dict,
                environment.EnvState,
                chex.Array,
                chex.PRNGKey,
            ],
            unused: Any,
        ) -> Tuple[
            Tuple[
                TrainState,
                TrainState,
                dict,
                environment.EnvState,
                chex.Array,
                chex.PRNGKey,
            ],
            dict,
        ]:
            """Training iteration of PPO that processes a single batch of data.

            Parameters
            ----------
            runner_state : Tuple[TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey]
                Input training state carried over from previous iteration of training.
            unused : Any
                Unused input value.

            Returns
            -------
            Tuple[Tuple[TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey], dict]
                Updated training state to pass to next iteration of training.
            """
            actor_state, vf_state, time_state, env_state, last_obs, key = runner_state

            # Step function used to collect data from the environment
            def _env_step(
                step_state: Tuple[
                    TrainState,
                    TrainState,
                    chex.Array,
                    environment.EnvState,
                    chex.Array,
                    chex.PRNGKey,
                ],
                unused: Any,
            ) -> Tuple[
                Tuple[
                    TrainState,
                    TrainState,
                    chex.Array,
                    environment.EnvState,
                    chex.Array,
                    chex.PRNGKey,
                ],
                Transition,
            ]:
                """Collect a single transition from the environment, used to generate batches of environment data.

                Parameters
                ----------
                step_state : Tuple[TrainState, TrainState, chex.Array, environment.EnvState, chex.Array, chex.PRNGKey]
                    Step state passed from one environment step to the next.
                unused : Any
                    Unused input value.

                Returns
                -------
                Tuple[Tuple[TrainState, TrainState, chex.Array, environment.EnvState, chex.Array, chex.PRNGKey], Transition]
                    Updated step state to pass to the next step of the environment.
                """
                actor_state, vf_state, timestep, env_state, last_obs, key = step_state

                # Sample actions from actor
                key, sampling_key = jax.random.split(key)
                logits = actor.apply(actor_state.params, last_obs)
                action, log_prob, _ = sample_actions(logits, sampling_key)

                # Get state-values from critic
                value = vf.apply(vf_state.params, last_obs).squeeze()

                # Step environments
                key, env_key = jax.random.split(key)
                env_step_key = jax.random.split(env_key, args.num_envs)
                obsv, env_state, reward, done, info = (
                    jax.vmap(  # Vectorize step function
                        env.step, in_axes=(0, 0, 0, None)
                    )(env_step_key, env_state, action, env_params)
                )

                # Store transition
                transition = Transition(
                    last_obs, action, reward, done, log_prob, value, info
                )

                # Update carry needed for next step
                step_state = (
                    actor_state,
                    vf_state,
                    timestep + 1,
                    env_state,
                    obsv,
                    key,
                )
                return step_state, transition

            # Collect a batch of trajectories
            step_state = (
                actor_state,
                vf_state,
                time_state["timesteps"],
                env_state,
                last_obs,
                key,
            )
            step_state, traj_batch = jax.lax.scan(
                _env_step, step_state, None, args.num_steps
            )
            actor_state, vf_state, timestep, env_state, last_obs, key = step_state

            # Calculate advantages with Generalized Advantage Estimation (GAE)
            last_value = vf.apply(vf_state.params, last_obs).squeeze()

            def _calculate_gae(
                traj_batch: Transition, last_value: chex.Array
            ) -> Tuple[chex.Array, chex.Array]:
                """Calculates GAE with a batch of trajectory data.

                Parameters
                ----------
                traj_batch : Transition
                    Batch of trajectories.
                last_value : chex.Array
                    State-value for last observation from batch collection rollouts.

                Returns
                -------
                Tuple[chex.Array, chex.Array]
                    Advantages and target values.
                """

                def _get_advantages(
                    gae_and_next_value: Tuple[chex.Array, chex.Array],
                    transition: Transition,
                ) -> Tuple[Tuple[chex.Array, chex.Array], chex.Array]:
                    """Calculates a single backward step of GAE.

                    Parameters
                    ----------
                    gae_and_next_value : Tuple[chex.Array, chex.Array]
                        Advantage and value from previous step.
                    transition : Transition
                        Transition data corresponding to this step.

                    Returns
                    -------
                    Tuple[Tuple[chex.Array, chex.Array], chex.Array]
                        Updated advantage and value to pass to next step.
                    """
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + args.gamma * next_value * (1 - done) - value
                    gae = delta + args.gamma * args.gae_lambda * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_value), last_value),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_value)

            # Update actor and critic networks over multiple epochs
            def _update_epoch(
                update_state: Tuple[
                    TrainState,
                    TrainState,
                    Transition,
                    chex.Array,
                    chex.Array,
                    chex.PRNGKey,
                ],
                unused: Any,
            ) -> Tuple[
                Tuple[
                    TrainState,
                    TrainState,
                    Transition,
                    chex.Array,
                    chex.Array,
                    chex.PRNGKey,
                ],
                Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
            ]:
                """Training epoch update.

                Parameters
                ----------
                update_state : Tuple[TrainState, TrainState, Transition, chex.Array, chex.Array, chex.PRNGKey]
                    Training state for performing update.
                unused : Any
                    Unused input parameter for lax.scan.

                Returns
                -------
                Tuple[Tuple[TrainState, TrainState, Transition, chex.Array, chex.Array, chex.PRNGKey], Tuple[chex.Array, chex.Array, chex.Array, chex.Array]]
                    Updated training state and losses.
                """

                # Update actor and critic networks from a single minibatch
                def _update_minibatch(
                    train_state: Tuple[TrainState, TrainState],
                    batch_info: Tuple[Transition, chex.Array, chex.Array],
                ) -> Tuple[
                    Tuple[TrainState, TrainState],
                    Tuple[chex.Array, chex.Array, chex.Array, chex.Array],
                ]:
                    """Performs a minibatch update.

                    Parameters
                    ----------
                    train_state : Tuple[TrainState, TrainState]
                        Training state.
                    batch_info : Tuple[Transition, chex.Array, chex.Array]
                        Batch data used for updates.

                    Returns
                    -------
                    Tuple[Tuple[TrainState, TrainState], Tuple[chex.Array, chex.Array, chex.Array, chex.Array]]
                        Updated training state.
                    """
                    actor_state, vf_state = train_state
                    traj_batch, advantages, targets = batch_info

                    # Calculate actor loss
                    def _actor_loss(
                        actor_params: dict, traj_batch: Transition, gae: chex.Array
                    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
                        """Loss function for the actor network.

                        Parameters
                        ----------
                        actor_params : dict
                            Params for the actor network.
                        traj_batch : Transition
                            Batch of trajectory data.
                        gae : chex.Array
                            GAE advantages.

                        Returns
                        -------
                        Tuple[chex.Array, chex.Array, chex.Array]
                            Losses associated with actor.
                        """
                        # Calculate new log probabilities and entropies
                        logits = actor.apply(actor_params, traj_batch.obs)
                        _, log_prob, entropy = give_actions(logits, traj_batch.action)

                        # Normalize advantages, if specified
                        norm_gae = (
                            (gae - gae.mean()) / (gae.std() + 1e-8)
                            if args.norm_adv
                            else gae
                        )

                        # Calculate policy loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        pg_loss1 = -norm_gae * ratio
                        pg_loss2 = -norm_gae * jnp.clip(
                            ratio, 1 - args.clip_coef, 1 + args.clip_coef
                        )
                        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                        # Calculate entropy loss
                        entropy_loss = entropy.mean()

                        # Calculate total actor loss
                        actor_loss = pg_loss - args.ent_coef * entropy_loss

                        return actor_loss, (pg_loss, entropy_loss)

                    # Calculate critic loss
                    def _critic_loss(
                        critic_params: dict, traj_batch: Transition, targets: chex.Array
                    ) -> Tuple[chex.Array, Tuple[chex.Array]]:
                        """Loss function for critic.

                        Parameters
                        ----------
                        critic_params : dict
                            Parameters of the critic network.
                        traj_batch : Transition
                            Batch of trajectory data.
                        targets : chex.Array
                            Target values calculated from GAE.

                        Returns
                        -------
                        Tuple[chex.Array, Tuple[chex.Array]]
                            Loss values associated with critic.
                        """
                        # Get state-values from critic
                        value = vf.apply(critic_params, traj_batch.obs).squeeze()

                        # Calculate value loss
                        if args.clip_vloss:
                            value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                            ).clip(-args.clip_coef, args.clip_coef)
                            value_losses_clipped = jnp.square(
                                value_pred_clipped - targets
                            )
                            value_losses_unclipped = jnp.square(value - targets)
                            value_loss = (
                                0.5
                                * jnp.maximum(
                                    value_losses_unclipped, value_losses_clipped
                                ).mean()
                            )
                        else:
                            value_loss = 0.5 * jnp.square(value - targets).mean()

                        # Calculate total critic loss
                        critic_loss = value_loss * args.vf_coef

                        return critic_loss, (value_loss)

                    # Calculate actor network gradients with respect to loss and update it
                    actor_grad_fn = jax.value_and_grad(_actor_loss, has_aux=True)
                    (actor_loss, (pg_loss, entropy_loss)), grads = actor_grad_fn(
                        actor_state.params, traj_batch, advantages
                    )
                    actor_state = actor_state.apply_gradients(grads=grads)

                    # Calculate critic network gradients with respect to loss and update it
                    critic_grad_fn = jax.value_and_grad(_critic_loss, has_aux=True)
                    (critic_loss, (value_loss)), grads = critic_grad_fn(
                        vf_state.params, traj_batch, targets
                    )
                    vf_state = vf_state.apply_gradients(grads=grads)

                    return (actor_state, vf_state), (
                        actor_loss + critic_loss,
                        pg_loss,
                        entropy_loss,
                        value_loss,
                    )

                # Carry data from previous epoch of training
                actor_state, vf_state, traj_batch, advantages, targets, key = (
                    update_state
                )

                # Create minibatches from batch for current epoch of training
                key, batch_key = jax.random.split(key)
                permutation = jax.random.permutation(batch_key, args.batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree.map(
                    lambda x: x.reshape((args.batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x, [args.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                # Train on all the minibatches
                train_state = (actor_state, vf_state)
                train_state, (total_loss, pg_loss, entropy_loss, value_loss) = (
                    jax.lax.scan(_update_minibatch, train_state, minibatches)
                )
                update_state = (
                    train_state[0],
                    train_state[1],
                    traj_batch,
                    advantages,
                    targets,
                    key,
                )
                return update_state, (total_loss, pg_loss, entropy_loss, value_loss)

            # Update time state
            time_state["timesteps"] = timestep
            time_state["updates"] = time_state["updates"] + 1

            # Update training state
            update_state = (actor_state, vf_state, traj_batch, advantages, targets, key)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, args.update_epochs
            )
            actor_state = update_state[0]
            vf_state = update_state[1]
            key = update_state[-1]

            # Update overall runner state
            runner_state = (actor_state, vf_state, time_state, env_state, last_obs, key)

            # Update metrics for logging
            metric = {
                "batch_info": traj_batch.info,
                "learning_rate": actor_state.opt_state[1].hyperparams["learning_rate"],
                "value_loss": loss_info[3].mean(),
                "policy_loss": loss_info[1].mean(),
                "entropy": loss_info[2].mean(),
                "total_loss": loss_info[0].mean(),
                "timesteps": time_state["timesteps"] * args.num_envs,
                "updates": time_state["updates"],
            }

            def logging_callback(
                metric: dict,
            ) -> None:
                """Callback for logging data during training.

                Parameters
                ----------
                metric : dict
                    Training metrics.
                """
                return_values = metric["batch_info"]["returned_episode_returns"][
                    metric["batch_info"]["returned_episode"]
                ]
                length_values = metric["batch_info"]["returned_episode_lengths"][
                    metric["batch_info"]["returned_episode"]
                ]
                timesteps = (
                    metric["batch_info"]["timestep"][
                        metric["batch_info"]["returned_episode"]
                    ]
                    * args.num_envs
                )
                for t in range(len(timesteps)):
                    print(
                        f"global step={timesteps[t]}, episodic return={return_values[t]}, episodic length={length_values[t]}"
                    )

                # If logging to wandb
                if args.track:
                    data_log = {
                        "misc/learning_rate": metric["learning_rate"].item(),
                        "losses/value_loss": metric["value_loss"].item(),
                        "losses/policy_loss": metric["policy_loss"].item(),
                        "losses/entropy": metric["entropy"].item(),
                        "losses/total_loss": metric["total_loss"].item(),
                        "misc/global_step": metric["timesteps"],
                        "misc/updates": metric["updates"],
                    }
                    if return_values.size > 0:
                        data_log["misc/episodic_return"] = return_values.mean().item()
                        data_log["misc/episodic_length"] = length_values.mean().item()
                    wandb.log(data_log, step=metric["timesteps"])

                # # If saving checkpoints
                # if args.save:
                #     if metric["updates"] % args.checkpoint_interval == 0:
                #         save(
                #             args.run_id,
                #             args.save_checkpoint_dir,
                #             metric["timesteps"],
                #             runner_state,
                #         )

            # Perform callback for logging
            jax.debug.callback(logging_callback, metric)

            # Perform checkpointing, if necessary
            if args.save:

                def _checkpointing_callback(
                    timestep: int,
                    runner_state: Tuple[
                        TrainState,
                        TrainState,
                        dict,
                        environment.EnvState,
                        chex.Array,
                        chex.PRNGKey,
                    ],
                ) -> None:
                    """Callback for saving checkpoints during training.

                    Parameters
                    ----------
                    timestep : int
                        Timestep.
                    runner_state : Tuple[ TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey, ]
                        Training state.
                    """
                    save(args.run_id, args.save_checkpoint_dir, timestep, runner_state)

                def _checkpoint(
                    timestep: int,
                    runner_state: Tuple[
                        TrainState,
                        TrainState,
                        dict,
                        environment.EnvState,
                        chex.Array,
                        chex.PRNGKey,
                    ],
                ) -> Tuple[
                    TrainState,
                    TrainState,
                    dict,
                    environment.EnvState,
                    chex.Array,
                    chex.PRNGKey,
                ]:
                    """Generates training checkpoints.

                    Parameters
                    ----------
                    timestep : int
                        Timestep.
                    runner_state : Tuple[ TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey, ]
                        Training state.

                    Returns
                    -------
                    Tuple[ TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey, ]
                        Un-modified training state.
                    """
                    jax.debug.callback(_checkpointing_callback, timestep, runner_state)
                    return runner_state

                def _no_op_checkpoint(
                    timestep: int,
                    runner_state: Tuple[
                        TrainState,
                        TrainState,
                        dict,
                        environment.EnvState,
                        chex.Array,
                        chex.PRNGKey,
                    ],
                ) -> Tuple[
                    TrainState,
                    TrainState,
                    dict,
                    environment.EnvState,
                    chex.Array,
                    chex.PRNGKey,
                ]:
                    """No-op that returns unchanged training state.

                    Parameters
                    ----------
                    timestep : int
                        Timestep.
                    runner_state : Tuple[ TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey, ]
                        Training state.

                    Returns
                    -------
                    Tuple[ TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.PRNGKey, ]
                        Un-modified training state.
                    """
                    return runner_state

                # If checkpointing interval condition is met, then save a checkpoint
                runner_state = jax.lax.cond(
                    time_state["updates"] % args.checkpoint_interval == 0,
                    _checkpoint,
                    _no_op_checkpoint,
                    *(time_state["timesteps"] * args.num_envs, runner_state),
                )

            return runner_state, metric

        # Run training
        key, train_key = jax.random.split(key)
        time_state = {"timesteps": jnp.array(0), "updates": jnp.array(0)}
        runner_state = (actor_state, vf_state, time_state, env_state, obsv, train_key)

        # Restore training state from checkpoint, if specified
        if args.resume:
            checkpointer = ocp.PyTreeCheckpointer()
            restored_runner_state = checkpointer.restore(
                args.resume_checkpoint_path, item=runner_state
            )
            runner_state = restored_runner_state

        train_iterations = args.num_iterations - runner_state[2]["updates"]
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, train_iterations
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    args = tyro.cli(Args)

    # If experiment identification isn't already given, create it
    if args.resume and args.run_id is not None:
        run_id = args.run_id
    else:
        run_name = f"{args.exp_name}"
        wandb_id = wandb.util.generate_id()
        run_id = f"{run_name}_{wandb_id}"

    # If tracking, set up wandb experiment
    if args.track:
        # If a unique wandb run id is given, then resume from that, otherwise
        # generate new run for resuming
        if args.resume and args.run_id is not None:
            wandb.init(
                id=run_id,
                dir=args.wandb_dir,
                project=args.wandb_project,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                entity=args.wandb_entity,
                resume="must",
                mode="online",
            )
        else:
            args.run_id = run_id
            wandb.init(
                id=run_id,
                dir=args.wandb_dir,
                project=args.wandb_project,
                group=args.wandb_group,
                job_type=args.wandb_job_type,
                entity=args.wandb_entity,
                config=vars(args),
                name=run_name,
                save_code=True,
                settings=wandb.Settings(code_dir="."),
                mode="online",
            )

    # Seeding
    key = jax.random.PRNGKey(args.seed)

    # Initialize training function
    train = make_train(args)

    # Run training
    train_output = jax.block_until_ready(train(key))
