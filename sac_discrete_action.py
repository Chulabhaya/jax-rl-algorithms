"""An end-to-end JAX implementation of SAC designed for discrete actions with a learned temperature, based on a combination of CleanRL's and PureJaxRL's styles of algorithm implementations."""

import os
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import chex
import flashbax as fbx
import flax
import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from gymnax.environments import environment

import wandb
from wrappers import FlattenObservationWrapper, LogWrapper

# config.update("jax_disable_jit", True)


@dataclass
class Args:
    """Arguments for running discrete-action SAC."""

    # General arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Weights & Biases specific arguments
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "sac_discrete_action"
    """"wandb project name"""
    wandb_dir: str = "./"
    """"wandb project directory"""
    wandb_entity: str = None
    """wandb entity (team) name"""
    wandb_group: str = None
    """wandb group name"""
    wandb_job_type: str = None
    """wandb job_type name, for categorizing runs in a group"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """id of the environment"""
    total_timesteps: int = 100000
    """total timesteps of the experiments"""
    buffer_size: int = 100000
    """replay memory buffer size"""
    gamma: float = 0.99
    """discount factor gamma"""
    tau: float = 0.005
    """target network update rate"""
    batch_size: int = 256
    """batch size of sample from the reply memory"""
    learning_starts: int = 1
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """frequency of updates for the target networks"""

    # Checkpointing specific arguments
    save: bool = False
    """automatic tuning of the entropy coefficient"""
    save_checkpoint_dir: str = None
    """path to directory to save checkpoints in"""
    checkpoint_interval: int = 25000
    """how often to save checkpoints during training (in timesteps)"""
    resume: bool = False
    """whether to resume training from a checkpoint"""
    resume_checkpoint_path: str = None
    """path to checkpoint to resume training from"""
    run_id: str = None
    """wandb unique run id for resuming"""


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


class CriticTrainState(TrainState):
    """Train state for SAC critic.

    Attributes
    ----------
    target_params : flax.core.FrozenDict
        Stores target network parameters.
    """

    target_params: flax.core.FrozenDict


class Critic(nn.Module):
    """Model for representing the SAC critic.

    Attributes
    ----------
    action_dim : int
        Action dimensionality for output of network.
    """

    action_dim: int

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        Computes q-values for input observations.

        Returns
        -------
        q_values : chex.Array
            Computed q-values.
        """
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        q_values = nn.Dense(self.action_dim)(x)
        return q_values


class VectorCritic(nn.Module):
    """Vectorized model for representing the SAC critic.

    Attributes
    ----------
    action_dim : int
        Action dimensionality for output of network.
    n_critics : int
        How many critics to create.
    """

    action_dim: int
    n_critics: int = 2

    @nn.compact
    def __call__(self, obs: chex.Array) -> chex.Array:
        """
        Computes q-values for input observations using multiple critics.

        Returns
        -------
        q_values : chex.Array
            Computed q-values.
        """
        # Idea taken from https://github.com/perrin-isir/xpag
        # Similar to https://github.com/tinkoff-ai/CORL for PyTorch
        vmap_critic = nn.vmap(
            Critic,
            variable_axes={"params": 0},  # parameters not shared between the critics
            split_rngs={"params": True},  # different initializations
            in_axes=None,  # doesn't batch inputs to network automatically
            out_axes=0,
            axis_size=self.n_critics,
        )
        q_values = vmap_critic(
            action_dim=self.action_dim,
        )(obs)
        return q_values


class Actor(nn.Module):
    """Model for representing the SAC actor.

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
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        action_logits = nn.Dense(self.action_dim)(x)
        return action_logits


class Temperature(nn.Module):
    """Model for representing the SAC temperature (entropy coefficient).

    Attributes
    ----------
    temperature_init : float
        Initial value of the temperature.
    """

    temperature_init: float = 1.0

    @nn.compact
    def __call__(self) -> chex.Array:
        """
        Computes and returns the current value of the temperature.

        The temperature is represented as the exponential of a learnable
        parameter (log_temperature), ensuring that the coefficient is always positive.

        Returns
        -------
        temperature : chex.Array
            The current value of the temperature.
        """
        log_temperature = self.param(
            "log_temperature",
            init_fn=lambda key: jnp.full((), jnp.log(self.temperature_init)),
        )
        temperature = jnp.exp(log_temperature)
        return temperature


def sample_actions(
    logits: chex.Array,
    sampling_key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """
    Calculates actions and probabilities.

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
    action_probs : chex.Array
        Probabilities for all actions possible.
    log_action_probs : chex.Array
        Logs of action probabilities, used for entropy.
    """
    # Directly use logits to sample actions
    actions = jax.random.categorical(sampling_key, logits)

    # Calculate action probabilities for return or other uses
    action_probs = jax.nn.softmax(logits)

    # Compute log probabilities using log_softmax for numerical stability
    log_action_probs = jax.nn.log_softmax(logits)

    return actions, action_probs, log_action_probs


def make_train(args: Args) -> Callable[[chex.PRNGKey], dict]:
    """Create training function.

    Parameters
    ----------
    args : Args
        Parameters used for initializing training function.

    Returns
    -------
    Callable[[chex.PRNGKey], dict]
        Train function.
    """
    # Initialize number of total training timesteps
    args.training_timesteps = args.total_timesteps - args.learning_starts

    # Set up environment
    env, env_params = make_env(args.env_id)

    # Training function
    def train(key: chex.PRNGKey) -> dict:
        """SAC training function.

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
        qf = VectorCritic(action_dim=env.action_space(env_params).n)
        key, critic_key = jax.random.split(key)
        qf_state = CriticTrainState.create(
            apply_fn=qf.apply,
            params=qf.init({"params": critic_key}, init_obs),
            target_params=qf.init({"params": critic_key}, init_obs),
            tx=optax.adam(learning_rate=args.q_lr),
        )
        # Initialize actor
        actor = Actor(action_dim=env.action_space(env_params).n)
        key, actor_key = jax.random.split(key)
        actor_state = TrainState.create(
            apply_fn=actor.apply,
            params=actor.init(actor_key, init_obs),
            tx=optax.adam(learning_rate=args.policy_lr),
        )
        # Initialize temperature model
        temperature_model = Temperature(temperature_init=1.0)
        target_entropy = -0.3 * np.log(1 / env.action_space(env_params).n)
        key, temperature_model_key = jax.random.split(key)
        temperature_state = TrainState.create(
            apply_fn=temperature_model.apply,
            params=temperature_model.init(temperature_model_key)["params"],
            tx=optax.adam(learning_rate=args.q_lr),
        )

        # Calculate initial temperature value
        temperature = temperature_model.apply({"params": temperature_state.params})

        # Initialize replay buffer and buffer state
        replay_buffer = fbx.make_flat_buffer(
            args.buffer_size, args.learning_starts, args.batch_size, False, None
        )
        # TODO: Generalize this
        buffer_state = replay_buffer.init(
            {
                "observations": jnp.array([5.0, 4.0, 2.0, 1.0]),
                "actions": jnp.array(0),
                "rewards": jnp.array(1.0),
                "next_observations": jnp.array([5.0, 4.0, 2.0, 1.0]),
                "dones": jnp.array(True),
            }
        )

        # Initialize environment
        key, env_reset_key = jax.random.split(key)
        obs, env_state = env.reset(env_reset_key, env_params)

        # Warm up replay buffer with random data
        def _warmup_step(
            warmup_state: Tuple[
                dict,
                environment.EnvState,
                chex.Array,
                chex.PRNGKey,
            ],
            unused: Any,
        ) -> Tuple[
            Tuple[
                dict,
                environment.EnvState,
                chex.Array,
                chex.PRNGKey,
            ],
        ]:
            """Warm up step used to initialize replay buffer with random data.

            Parameters
            ----------
            warmup_state : Tuple[dict, environment.EnvState, chex.Array, chex.PRNGKey]
                Carry warmup state from the previous iteration.
            unused : Any
                Unused input value.

            Returns
            -------
            Tuple[Tuple[dict, environment.EnvState, chex.Array, chex.PRNGKey], None]
                Updated warmup state to pass to the next iteration.
            """
            buffer_state, env_state, last_obs, key = warmup_state

            # Sample random actions from actor
            key, sampling_key = jax.random.split(key)
            action = env.action_space(env_params).sample(sampling_key)

            # Step environment
            key, env_step_key = jax.random.split(key)
            obs, env_state, reward, done, info = env.step(
                env_step_key, env_state, action, env_params
            )

            # Update replay buffer
            buffer_state = replay_buffer.add(
                buffer_state,
                {
                    "observations": last_obs,
                    "actions": action,
                    "rewards": reward,
                    "next_observations": obs,
                    "dones": done,
                },
            )

            # Update carry needed for next step
            warmup_state = (
                buffer_state,
                env_state,
                obs,
                key,
            )
            return warmup_state, None

        # Warm up replay buffer
        warmup_state = (
            buffer_state,
            env_state,
            obs,
            key,
        )
        warmup_state, _ = jax.lax.scan(
            _warmup_step, warmup_state, None, args.learning_starts
        )
        buffer_state, env_state, obs, key = warmup_state

        # Step function representing a training iteration
        def _update_step(
            runner_state: Tuple[
                TrainState,
                TrainState,
                TrainState,
                dict,
                environment.EnvState,
                chex.Array,
                chex.Array,
                chex.PRNGKey,
            ],
            unused: Any,
        ) -> Tuple[
            Tuple[
                TrainState,
                TrainState,
                TrainState,
                dict,
                environment.EnvState,
                chex.Array,
                chex.Array,
                chex.PRNGKey,
            ],
            dict,
        ]:
            """Training iteration of SAC.

            Parameters
            ----------
            runner_state : Tuple[TrainState, TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.Array, chex.PRNGKey]
                Carry train state passed from one training iteration to the next.
            unused : Any
                Unused input value.

            Returns
            -------
            Tuple[Tuple[TrainState, TrainState, TrainState, dict, environment.EnvState, chex.Array, chex.Array, chex.PRNGKey], dict]
                Updated training state to pass to the next iteration of training.
            """
            (
                actor_state,
                qf_state,
                temperature_state,
                buffer_state,
                env_state,
                temperature,
                last_obs,
                key,
            ) = runner_state
            # Sample action from actor
            key, sampling_key = jax.random.split(key)
            logits = actor.apply(actor_state.params, last_obs)
            action, _, _ = sample_actions(logits, sampling_key)

            # Step environment
            key, env_step_key = jax.random.split(key)
            obs, env_state, reward, done, info = env.step(
                env_step_key, env_state, action, env_params
            )

            # Update replay buffer
            buffer_state = replay_buffer.add(
                buffer_state,
                {
                    "observations": last_obs,
                    "actions": action,
                    "rewards": reward,
                    "next_observations": obs,
                    "dones": done,
                },
            )

            # Sample batch of data from buffer
            key, buffer_key = jax.random.split(key)
            traj_batch = replay_buffer.sample(buffer_state, buffer_key)

            # Update critic
            def _update_critic(
                actor_state: TrainState,
                qf_state: CriticTrainState,
                temperature: chex.Array,
                observations: chex.Array,
                actions: chex.Array,
                next_observations: chex.Array,
                rewards: chex.Array,
                terminateds: chex.Array,
                key: chex.PRNGKey,
            ) -> Tuple[CriticTrainState, Tuple[chex.Array, chex.Array], chex.PRNGKey]:
                """Updates the critic.

                Parameters
                ----------
                actor_state : TrainState
                    State of the actor network.
                qf_state : CriticTrainState
                    State of the critic network.
                temperature : chex.Array
                    Temperature value.
                observations : chex.Array
                    Batch of observations.
                actions : chex.Array
                    Batch of actions.
                next_observations : chex.Array
                    Batch of next observations.
                rewards : chex.Array
                    Batch of rewards.
                terminateds : chex.Array
                    Batch of terminateds.
                key : chex.PRNGKey
                    Key for RNG.

                Returns
                -------
                Tuple[CriticTrainState, Tuple[chex.Array, chex.Array], chex.PRNGKey]
                    Results from updating the critic.
                """
                # Get next state action probs and log probs
                logits = actor.apply(actor_state.params, next_observations)
                _, next_state_action_probs, next_state_log_action_probs = (
                    sample_actions(logits, key)
                )

                # Calculate Q-value distribution estimates (two for reducing overestimation bias)
                qf_next_values = qf.apply(
                    qf_state.target_params, next_observations
                )  # Shape is (n_critics, batch_size, feature_size)
                min_qf_next_values = jnp.min(
                    qf_next_values, axis=0
                )  # Shape is (batch_size, feature_size)

                # Calculate entropy-regularized next Q-values
                next_q_values = next_state_action_probs * (
                    min_qf_next_values - temperature * next_state_log_action_probs
                )

                # Calculate target Q-values
                target_q_values = rewards + (
                    (1 - terminateds) * args.gamma * jnp.sum(next_q_values, axis=1)
                )

                def _mse_loss(qf_params: dict) -> Tuple[chex.Array, chex.Array]:
                    """
                    Computes the MSE loss for updating the critic.

                    Parameters:
                    -----------
                    qf_params : dict
                        The parameters of the Q-function model.

                    Returns:
                    --------
                    Tuple[chex.Array, chex.Array]
                        A tuple containing the MSE loss and the mean of the current Q-values.
                    """
                    # Calculate current Q-values distribution
                    current_q_values = qf.apply(qf_params, observations)

                    # Get current Q-values specifically corresponding to actions taken
                    selected_current_q_values = current_q_values[
                        jnp.arange(current_q_values.shape[0])[:, None],
                        jnp.arange(current_q_values.shape[1]),
                        actions,
                    ]  # Shape is (n_critics, batch_size)

                    # Calculate mean over the batch and then sum for each critic
                    critic_loss = (
                        0.5
                        * ((selected_current_q_values - target_q_values) ** 2)
                        .mean(axis=1)
                        .sum()
                    )
                    return critic_loss, current_q_values.mean()

                # Calculate values and gradients of loss with respect to Q-function network params
                (qf_loss_value, qf_values), grads = jax.value_and_grad(
                    _mse_loss, has_aux=True
                )(qf_state.params)

                # Apply gradients to Q-function network
                qf_state = qf_state.apply_gradients(grads=grads)

                return (qf_state, (qf_loss_value, qf_values))

            key, critic_key = jax.random.split(key)
            qf_state, (qf_loss_value, qf_values) = _update_critic(
                actor_state,
                qf_state,
                temperature,
                traj_batch.experience.first["observations"],
                traj_batch.experience.first["actions"],
                traj_batch.experience.first["next_observations"],
                traj_batch.experience.first["rewards"],
                traj_batch.experience.first["dones"],
                critic_key,
            )

            # Updates the actor and temperature multiple times to
            # account for delayed frequency
            def _delayed_update_actor_and_temperature(
                actor_state: TrainState,
                qf_state: CriticTrainState,
                temperature_state: TrainState,
                temperature: chex.Array,
                observations: chex.Array,
                key: chex.PRNGKey,
            ) -> Tuple[
                Tuple[
                    TrainState,
                    CriticTrainState,
                    TrainState,
                    chex.Array,
                    chex.Array,
                    chex.PRNGKey,
                ],
                Tuple[chex.Array, chex.Array, chex.Array],
            ]:
                """Performs delayed update of the actor and temperature models.

                Parameters
                ----------
                actor_state : TrainState
                    State of the actor network.
                qf_state : CriticTrainState
                    State of the critic network.
                temperature_state : TrainState
                    State of the temperature.
                temperature : chex.Array
                    Temperature value.
                observations : chex.Array
                    Observations.
                key : chex.PRNGKey
                    RNG key.

                Returns
                -------
                Tuple[ Tuple[ TrainState, CriticTrainState, TrainState, chex.Array, chex.Array, chex.PRNGKey, ], Tuple[chex.Array, chex.Array, chex.Array], ]
                    Updated actor and temperature and corresponding losses.
                """

                # Update function for updating actor and temperature
                def _update_actor_and_temperature(
                    actor_and_temperature_update_state: Tuple[
                        TrainState,
                        CriticTrainState,
                        TrainState,
                        chex.Array,
                        chex.Array,
                        chex.PRNGKey,
                    ],
                    unused: Any,
                ) -> Tuple[
                    Tuple[
                        TrainState,
                        CriticTrainState,
                        TrainState,
                        chex.Array,
                        chex.Array,
                        chex.PRNGKey,
                    ],
                    Tuple[chex.Array, chex.Array, chex.Array],
                ]:
                    """A single step of updating the actor and temperature models.

                    Parameters
                    ----------
                    actor_and_temperature_update_state : Tuple[ TrainState, CriticTrainState, TrainState, chex.Array, chex.Array, chex.PRNGKey ]
                        State of the actor network to pass from one iteration to the next.
                    unused : Any
                        Unused input value.

                    Returns
                    -------
                    Tuple[ Tuple[ TrainState, CriticTrainState, TrainState, chex.Array, chex.Array, chex.PRNGKey, ], Tuple[chex.Array, chex.Array, chex.Array], ]
                        Updated actor and temperature and corresponding losses to pass to the next iteration.
                    """
                    (
                        actor_state,
                        qf_state,
                        temperature_state,
                        temperature,
                        observations,
                        key,
                    ) = actor_and_temperature_update_state
                    key, actor_key, temperature_key = jax.random.split(key, num=3)

                    # Update actor
                    def _actor_loss(
                        actor_params: dict,
                    ) -> Tuple[chex.Array, chex.Array]:
                        """
                        Computes the actor loss for updating the actor.

                        Parameters:
                        -----------
                        actor_params : dict
                            The parameters of the actor model.

                        Returns:
                        --------
                        Tuple[chex.Array, chex.Array]
                            A tuple containing the actor loss and the entropy.
                        """
                        # Get current state action probs and log probs
                        logits = actor.apply(actor_params, observations)
                        _, state_action_probs, state_log_action_probs = sample_actions(
                            logits, actor_key
                        )

                        # Calculate Q-values of current state.
                        qf_pi = qf.apply(
                            qf_state.params, observations
                        )  # Shape is (n_critics, batch_size, feature_size)
                        min_qf_pi = jnp.min(
                            qf_pi, axis=0
                        )  # Shape is (batch_size, feature_size)

                        # Calculate actor loss
                        actor_loss = (
                            (
                                state_action_probs
                                * ((temperature * state_log_action_probs) - min_qf_pi)
                            )
                            .sum(1)
                            .mean()
                        )

                        # Calculate average entropy of actor
                        entropy = (
                            (-(state_action_probs * state_log_action_probs))
                            .sum(1)
                            .mean()
                        )
                        return actor_loss, entropy

                    # Calculate values and gradients of loss with respect to actor network params
                    (actor_loss_value, entropy), grads = jax.value_and_grad(
                        _actor_loss, has_aux=True
                    )(actor_state.params)

                    # Apply gradients to actor network
                    actor_state = actor_state.apply_gradients(grads=grads)

                    # Update temperature
                    def temperature_loss(
                        temperature_params: dict,
                    ) -> chex.Array:
                        """Computes the temperature loss.

                        Parameters
                        ----------
                        temperature_params : dict
                            Params for the temperature model.

                        Returns
                        -------
                        chex.Array
                            Temperature loss.
                        """
                        # Get current state action probs and log probs
                        logits = actor.apply(actor_state.params, observations)
                        _, state_action_probs, state_log_action_probs = sample_actions(
                            logits, temperature_key
                        )

                        # Calculate entropy coefficient value
                        temperature = temperature_model.apply(
                            {"params": temperature_params}
                        )

                        # Calculate entropy coefficient loss
                        temperature_loss = (
                            (
                                state_action_probs
                                * (
                                    -jnp.log(temperature)
                                    * (state_log_action_probs + target_entropy)
                                )
                            )
                            .sum(1)
                            .mean()
                        )
                        return temperature_loss

                    temperature_loss_value, grads = jax.value_and_grad(
                        temperature_loss
                    )(temperature_state.params)
                    temperature_state = temperature_state.apply_gradients(grads=grads)

                    # Update temperature
                    temperature = temperature_model.apply(
                        {"params": temperature_state.params}
                    )

                    return (
                        actor_state,
                        qf_state,
                        temperature_state,
                        temperature,
                        observations,
                        key,
                    ), (actor_loss_value, entropy, temperature_loss_value)

                # Compensate for delay by doing multiple updates
                actor_and_temperature_update_state = (
                    actor_state,
                    qf_state,
                    temperature_state,
                    temperature,
                    observations,
                    key,
                )
                actor_and_temperature_update_state, actor_and_temperature_loss_info = (
                    jax.lax.scan(
                        _update_actor_and_temperature,
                        actor_and_temperature_update_state,
                        None,
                        args.policy_frequency,
                    )
                )
                return (
                    actor_and_temperature_update_state,
                    actor_and_temperature_loss_info,
                )

            def _no_op_delayed_update_actor_and_temperature(
                actor_state: TrainState,
                qf_state: CriticTrainState,
                temperature_state: TrainState,
                temperature: chex.Array,
                observations: chex.Array,
                key: chex.PRNGKey,
            ) -> Tuple[
                Tuple[
                    TrainState,
                    CriticTrainState,
                    TrainState,
                    chex.Array,
                    chex.Array,
                    chex.PRNGKey,
                ],
                Tuple[chex.Array, chex.Array, chex.Array],
            ]:
                """No-op that leaves actor and temperature models unchanged and returns placeholder loss values.

                Parameters
                ----------
                actor_state : TrainState
                    State of the actor network.
                qf_state : CriticTrainState
                    State of the critic network.
                temperature_state : TrainState
                    State of the temperature.
                temperature : chex.Array
                    Temperature value.
                observations : chex.Array
                    Observations.
                key : chex.PRNGKey
                    RNG key.

                Returns
                -------
                Tuple[ Tuple[ TrainState, CriticTrainState, TrainState, chex.Array, chex.Array, chex.PRNGKey, ], Tuple[chex.Array, chex.Array, chex.Array], ]
                    Unchanged actor and temperature models and placeholder losses.
                """
                return (
                    actor_state,
                    qf_state,
                    temperature_state,
                    temperature,
                    observations,
                    key,
                ), (
                    jnp.zeros(args.policy_frequency),
                    jnp.zeros(args.policy_frequency),
                    jnp.zeros(args.policy_frequency),
                )

            (actor_and_temperature_update_state, actor_and_temperature_loss_info) = (
                jax.lax.cond(
                    info["timestep"] % args.policy_frequency == 0,  # TD3 delayed update
                    _delayed_update_actor_and_temperature,
                    _no_op_delayed_update_actor_and_temperature,
                    *(
                        actor_state,
                        qf_state,
                        temperature_state,
                        temperature,
                        traj_batch.experience.first["observations"],
                        key,
                    ),
                )
            )
            actor_state = actor_and_temperature_update_state[0]
            temperature_state = actor_and_temperature_update_state[2]
            temperature = actor_and_temperature_update_state[3]
            key = actor_and_temperature_update_state[5]

            # Update target networks
            def _target_soft_update(qf_state: Tuple[TrainState]) -> Tuple[TrainState]:
                """Updates target Q-function networks.

                Parameters
                ----------
                qf_state : Tuple[TrainState]
                    Q-function train state.

                Returns
                -------
                Tuple[TrainState]
                    Updated Q-function train state with updated target networks.
                """
                qf_state = qf_state.replace(
                    target_params=optax.incremental_update(
                        qf_state.params, qf_state.target_params, args.tau
                    )
                )
                return qf_state

            def _no_op_target_soft_update(
                qf_state: Tuple[TrainState],
            ) -> Tuple[TrainState]:
                """No-op that returns an unchanged Q-function state.

                Parameters
                ----------
                qf_state : Tuple[TrainState]
                    Q-function train state.

                Returns
                -------
                Tuple[TrainState]
                    Unchanged Q-function train state.
                """
                return qf_state

            qf_state = jax.lax.cond(
                info["timestep"] % args.target_network_frequency
                == 0,  # TD3 delayed update
                _target_soft_update,
                _no_op_target_soft_update,
                qf_state,
            )

            # Update the returning metrics
            metric = info
            metric["qf_values"] = qf_values
            metric["qf_loss"] = qf_loss_value
            metric["actor_loss"] = actor_and_temperature_loss_info[0].mean()
            metric["actor_entropy"] = actor_and_temperature_loss_info[1].mean()
            metric["temperature"] = temperature
            metric["temperature_loss"] = actor_and_temperature_loss_info[2].mean()

            def callback(info: dict) -> None:
                """Callback for logging data during training.

                Parameters
                ----------
                info : dict
                    Logging data.
                """
                return_value = info["returned_episode_returns"]
                length_value = info["returned_episode_lengths"]
                done = info["returned_episode"]
                timestep = info["timestep"]
                if done:
                    print(
                        f"global step={timestep}, episodic return={return_value}, episodic length={length_value}"
                    )

                if args.track:
                    data_log = {"misc/global_step": timestep}
                    if done:
                        data_log["misc/episodic_return"] = return_value.item()
                        data_log["misc/episodic_length"] = length_value.item()
                    if timestep % 100 == 0:
                        data_log = {
                            "losses/qf_values": info["qf_values"].item(),
                            "losses/qf_loss": info["qf_loss"].item(),
                        }
                        if timestep % args.policy_frequency == 0:
                            data_log["losses/actor_loss"] = info["actor_loss"].item()
                            data_log["losses/actor_entropy"] = info[
                                "actor_entropy"
                            ].item()
                            data_log["losses/temperature"] = info["temperature"].item()
                            data_log["losses/temperature_loss"] = info[
                                "temperature_loss"
                            ].item()
                    wandb.log(data_log, step=timestep)

            jax.debug.callback(callback, metric)
            runner_state = (
                actor_state,
                qf_state,
                temperature_state,
                buffer_state,
                env_state,
                temperature,
                obs,
                key,
            )
            return runner_state, metric

        # Run training
        key, train_key = jax.random.split(key)
        runner_state = (
            actor_state,
            qf_state,
            temperature_state,
            buffer_state,
            env_state,
            temperature,
            obs,
            train_key,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, args.training_timesteps
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Set up experiment identification
    run_name = f"{args.exp_name}"
    wandb_id = wandb.util.generate_id()
    run_id = f"{run_name}_{wandb_id}"

    # If tracking, set up wandb experiment
    if args.track:
        # If a unique wandb run id is given, then resume from that, otherwise
        # generate new run for resuming
        if args.resume and args.run_id is not None:
            run_id = args.run_id
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

    # Compile training function
    train_vjit = jax.jit(make_train(args))

    # Run training
    train_output = jax.block_until_ready(train_vjit(key))
