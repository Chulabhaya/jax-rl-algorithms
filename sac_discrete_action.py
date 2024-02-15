import os
import random
import time
from dataclasses import dataclass
from typing import Tuple

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
import wandb

from replay_buffers import ReplayBuffer


@dataclass
class Args:
    """Arguments for running discrete-action SAC."""

    # General arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Weights & Biases specific arguments
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "jax-sac-discrete-action"
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
    env_id: str = "CartPole-v0"
    """id of the environment"""
    total_timesteps: int = 100500
    """total timesteps of the experiments"""
    buffer_size: int = 100000
    """replay memory buffer size"""
    gamma: float = 0.99
    """discount factor gamma"""
    tau: float = 0.005
    """target network update rate"""
    batch_size: int = 256
    """batch size of sample from the reply memory"""
    learning_starts: int = 5000
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """learning rate of the policy network optimizer"""
    q_lr: float = 3e-4
    """learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """frequency of training policy (delayed)"""
    target_network_frequency: int = 1
    """frequency of updates for the target networks"""
    alpha: float = 0.2
    """entropy regularization coefficient"""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

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


def make_env(env_id: str, seed: int) -> gym.Env:
    """Initializes Gymnasium environment.

    Parameters
    ----------
    env_id : str
        ID for the Gymnasium environment.
    seed : int
        Seed.

    Returns
    -------
    gym.Env
        Gymnasium environment.
    """
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env.action_space.seed(seed)

    return env


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
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Computes q-values for input observations.

        Returns
        -------
        q_values : jnp.ndarray
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
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Computes q-values for input observations using multiple critics.

        Returns
        -------
        q_values : jnp.ndarray
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
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Computes action logits for input observations.

        Returns
        -------
        action_logits : jnp.ndarray
            Computed action logits.
        """
        x = nn.Dense(128)(obs)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        action_logits = nn.Dense(self.action_dim)(x)
        return action_logits


class EntropyCoefficient(nn.Module):
    """Model for representing the SAC entropy coefficient.

    Attributes
    ----------
    alpha_init : float
        Initial value of the entropy coefficient alpha.
    """

    alpha_init: float = 1.0

    @nn.compact
    def __call__(self) -> jnp.ndarray:
        """
        Computes and returns the current value of the entropy coefficient.

        The entropy coefficient is represented as the exponential of a learnable
        parameter (log_alpha), ensuring that the coefficient is always positive.

        Returns
        -------
        entropy_coefficient_value : jnp.ndarray
            The current value of the entropy coefficient.
        """
        log_alpha = self.param(
            "log_alpha", init_fn=lambda key: jnp.full((), jnp.log(self.alpha_init))
        )
        entropy_coefficient_value = jnp.exp(log_alpha)
        return entropy_coefficient_value

@jax.jit
def sample_actions(
    logits: jnp.ndarray,
    sampling_key: jax.random.KeyArray,
):
    """
    Calculates actions and probabilities.

    Parameters
    ----------
    logits : jnp.ndarray
        Action logits output from actor network.
    sampling_key : jax.random.KeyArray
        RNG key.

    Returns
    -------
    actions : jnp.ndarray
        Sampled actions from action distributions.
    action_probs : jnp.ndarray
        Probabilities for all actions possible.
    log_action_probs : jnp.ndarray
        Logs of action probabilities, used for entropy.
    """
    # Directly use logits to sample actions
    actions = jax.random.categorical(sampling_key, logits)

    # Calculate action probabilities for return or other uses
    action_probs = jax.nn.softmax(logits)

    # Compute log probabilities using log_softmax for numerical stability
    log_action_probs = jax.nn.log_softmax(logits)

    return actions, action_probs, log_action_probs

@jax.jit
def soft_update(tau: float, qf_state: CriticTrainState) -> CriticTrainState:
    """Performs a soft update of the target network for the critic.

    Parameters
    ----------
    tau : float
        Target smoothing coefficient.
    qf_state : CriticTrainState
        Training state of the critic.

    Returns
    -------
    CriticTrainState
        Updated training state of the critic.
    """
    qf_state = qf_state.replace(
        target_params=optax.incremental_update(
            qf_state.params, qf_state.target_params, tau
        )
    )
    return qf_state


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
                mode="offline",
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
                mode="offline",
            )

    # Seeding
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Initialize environment
    env = make_env(args.env_id, args.seed)

    # Initialize models
    obs, _ = env.reset(seed=args.seed)
    qf = VectorCritic(action_dim=env.action_space.n)
    key, critic_key = jax.random.split(key, 2)
    qf_state = CriticTrainState.create(
        apply_fn=qf.apply,
        params=qf.init({"params": critic_key}, obs),
        target_params=qf.init({"params": critic_key}, obs),
        tx=optax.adam(learning_rate=args.q_lr),
    )
    qf.apply = jax.jit(qf.apply)

    actor = Actor(action_dim=env.action_space.n)
    key, actor_key = jax.random.split(key, 2)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.adam(learning_rate=args.policy_lr),
    )
    actor.apply = jax.jit(actor.apply)

    # Set up automatic entropy tuning
    if args.autotune:
        entropy_coefficient = EntropyCoefficient(alpha_init=1.0)
        target_entropy = -0.3 * np.log(1 / env.action_space.n)
        key, entropy_coefficient_key = jax.random.split(key, 2)
        entropy_coefficient_state = TrainState.create(
            apply_fn=entropy_coefficient.apply,
            params=entropy_coefficient.init(entropy_coefficient_key)["params"],
            tx=optax.adam(learning_rate=args.q_lr),
        )
        entropy_coefficient.apply = jax.jit(entropy_coefficient.apply)
    else:
        entropy_coefficient_value = jnp.array(args.alpha)

    # Initialize replay buffer
    rb = ReplayBuffer(
        args.buffer_size, env.observation_space.shape[0], 0  # Discrete action
    )

    # Define update functions here to limit the need for static argname
    # Define critic update function
    @jax.jit
    def update_critic(
        actor_state: TrainState,
        qf_state: CriticTrainState,
        entropy_coefficient_value: jnp.ndarray,
        observations: np.ndarray,
        actions: np.ndarray,
        next_observations: np.ndarray,
        rewards: np.ndarray,
        terminateds: np.ndarray,
        key: jax.random.KeyArray,
    ) -> Tuple[CriticTrainState, Tuple[jnp.ndarray, jnp.ndarray], jax.random.KeyArray]:
        """Updates the critic.

        Parameters
        ----------
        actor_state : TrainState
            State of the actor network.
        qf_state : CriticTrainState
            State of the critic network.
        entropy_coefficient_value : jnp.ndarray
            Entropy coefficient value alpha.
        observations : np.ndarray
            Batch of observations.
        actions : np.ndarray
            Batch of actions.
        next_observations : np.ndarray
            Batch of next observations.
        rewards : np.ndarray
            Batch of rewards.
        terminateds : np.ndarray
            Batch of terminateds.
        key : jax.random.KeyArray
            Key for RNG.

        Returns
        -------
        Tuple[CriticTrainState, Tuple[jnp.ndarray, jnp.ndarray], jax.random.KeyArray]
            Results from updating the critic.
        """
        # Split key for RNG
        key, sampling_key = jax.random.split(key, 2)

        # Get next state action probs and log probs
        logits = actor.apply(actor_state.params, next_observations)
        _, next_state_action_probs, next_state_log_action_probs = sample_actions(
            logits, sampling_key
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
            min_qf_next_values - entropy_coefficient_value * next_state_log_action_probs
        )

        # Calculate target Q-values
        target_q_values = rewards + (
            (1 - terminateds) * args.gamma * jnp.sum(next_q_values, axis=1)
        )

        def mse_loss(qf_params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Computes the MSE loss for updating the critic.

            Parameters:
            -----------
            qf_params : dict
                The parameters of the Q-function model.

            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray]
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
        (qf_loss_value, qf_values), grads = jax.value_and_grad(mse_loss, has_aux=True)(
            qf_state.params
        )

        # Apply gradients to Q-function network
        qf_state = qf_state.apply_gradients(grads=grads)

        return (
            qf_state,
            (qf_loss_value, qf_values),
            key,
        )

    # Define actor update function
    @jax.jit
    def update_actor(
        actor_state: TrainState,
        qf_state: CriticTrainState,
        entropy_coefficient_value: jnp.ndarray,
        observations: np.ndarray,
        key: jax.random.KeyArray,
    ) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray, jax.random.KeyArray]:
        """Updates the actor.

        Parameters
        ----------
        actor_state : TrainState
            State of the actor network.
        qf_state : CriticTrainState
            State of the critic network.
        entropy_coefficient_value : jnp.ndarray
            Entropy coefficient value alpha.
        observations : np.ndarray
            Observations.
        key : jax.random.KeyArray
            RNG key.

        Returns
        -------
        Tuple[TrainState, jnp.ndarray, jnp.ndarray, jax.random.KeyArray]
            Results from updating the actor.
        """
        # Split key for RNG
        key, sampling_key = jax.random.split(key, 2)

        def actor_loss(actor_params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Computes the actor loss for updating the actor.

            Parameters:
            -----------
            actor_params : dict
                The parameters of the actor model.

            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray]
                A tuple containing the actor loss and the entropy.
            """
            # Get current state action probs and log probs
            logits = actor.apply(actor_params, observations)
            _, state_action_probs, state_log_action_probs = sample_actions(
                logits, sampling_key
            )

            # Calculate Q-values of current state.
            qf_pi = qf.apply(
                qf_state.params, observations
            )  # Shape is (n_critics, batch_size, feature_size)
            min_qf_pi = jnp.min(qf_pi, axis=0)  # Shape is (batch_size, feature_size)

            # Calculate actor loss
            actor_loss = (
                (
                    state_action_probs
                    * ((entropy_coefficient_value * state_log_action_probs) - min_qf_pi)
                )
                .sum(1)
                .mean()
            )

            # Calculate average entropy of actor
            entropy = (-(state_action_probs * state_log_action_probs)).sum(1).mean()
            return actor_loss, entropy

        # Calculate values and gradients of loss with respect to actor network params
        (actor_loss_value, entropy), grads = jax.value_and_grad(
            actor_loss, has_aux=True
        )(actor_state.params)

        # Apply gradients to actor network
        actor_state = actor_state.apply_gradients(grads=grads)

        return actor_state, actor_loss_value, entropy, key

    # Define entropy coefficient update function
    @jax.jit
    def update_temperature(
        entropy_coefficient_state: TrainState,
        actor_state: TrainState,
        observations: np.ndarray,
        key: jax.random.KeyArray,
    ) -> Tuple[TrainState, jnp.ndarray, jax.random.KeyArray]:
        """Updates the entropy coefficient.

        Parameters
        ----------
        entropy_coefficient_state : TrainState
            Entropy coefficient state.
        actor_state : TrainState
            Actor state.
        observations : np.ndarray
            Observations.
        key : jax.random.KeyArray
            RNG key.

        Returns
        -------
        Tuple[TrainState, jnp.ndarray, jax.random.KeyArray]
            Result from updating entropy coefficient.
        """
        # Split key for RNG
        key, sampling_key = jax.random.split(key, 2)

        def temperature_loss(entropy_coefficient_params: dict) -> jnp.ndarray:
            """Computes the entropy coefficient loss.

            Parameters
            ----------
            entropy_coefficient_params : dict
                Params for the entropy coefficient.

            Returns
            -------
            jnp.ndarray
                Entropy coefficient loss.
            """
            # Get current state action probs and log probs
            logits = actor.apply(actor_state.params, observations)
            _, state_action_probs, state_log_action_probs = sample_actions(
                logits, sampling_key
            )

            # Calculate entropy coefficient value
            entropy_coefficient_value = entropy_coefficient.apply(
                {"params": entropy_coefficient_params}
            )

            # Calculate entropy coefficient loss
            entropy_coefficient_loss = (
                (
                    state_action_probs
                    * (
                        -jnp.log(entropy_coefficient_value)
                        * (state_log_action_probs + target_entropy)
                    )
                )
                .sum(1)
                .mean()
            )
            return entropy_coefficient_loss

        entropy_coefficient_loss, grads = jax.value_and_grad(temperature_loss)(
            entropy_coefficient_state.params
        )
        entropy_coefficient_state = entropy_coefficient_state.apply_gradients(
            grads=grads
        )

        return entropy_coefficient_state, entropy_coefficient_loss, key

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    obs, _ = env.reset(seed=args.seed)
    for global_step in range(args.total_timesteps):
        # Store values for data logging for each global step
        data_log = {}

        # Action logic
        if global_step < args.learning_starts:
            action = env.action_space.sample()
        else:
            # Split key for RNG
            key, sampling_key = jax.random.split(key, 2)

            # Sample action from actor
            logits = actor.apply(actor_state.params, obs)
            action, _, _ = sample_actions(logits, sampling_key)
            
            # Convert action to Numpy format for environment
            action = np.array(action)

        # Take step in environment
        next_obs, reward, terminated, truncated, info = env.step(action)

        # Save data to replay buffer
        rb.add(obs, action, reward, next_obs, terminated, truncated)

        # Update next obs
        obs = next_obs

        # Handle episode end, record rewards for plotting purposes
        if terminated or truncated:
            print(
                f"global_step={global_step}, episodic_return={info['episode']['r'][0]}, episodic_length={info['episode']['l'][0]}",
                flush=True,
            )
            data_log["misc/episodic_return"] = info["episode"]["r"][0]
            data_log["misc/episodic_length"] = info["episode"]["l"][0]

            obs, info = env.reset()

        # Training logic
        if global_step > args.learning_starts:
            # Sample data from replay buffer
            (
                observations,
                actions,
                rewards,
                next_observations,
                terminateds,
                truncateds,
            ) = rb.sample(args.batch_size)

            # Calculate alpha value
            if args.autotune:
                entropy_coefficient_value = entropy_coefficient.apply(
                    {"params": entropy_coefficient_state.params}
                )

            # Update critic
            qf_state, (qf_loss_value, qf_values), key = update_critic(
                actor_state,
                qf_state,
                entropy_coefficient_value,
                observations,
                actions,
                next_observations,
                rewards,
                terminateds,
                key,
            )

            # Update actor and entropy regularization
            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                (actor_state, actor_loss_value, entropy, key) = update_actor(
                    actor_state,
                    qf_state,
                    entropy_coefficient_value,
                    observations,
                    key,
                )

                if args.autotune:
                    entropy_coefficient_state, entropy_coefficient_loss, key = (
                        update_temperature(
                            entropy_coefficient_state, actor_state, observations, key
                        )
                    )

            # Update the target networks
            if global_step % args.target_network_frequency == 0:
                qf_state = soft_update(args.tau, qf_state)
            
            if args.track:
                # Update logging data
                if global_step % 100 == 0:
                    data_log["losses/qf_values"] = qf_values.mean().item()
                    data_log["losses/qf_loss"] = qf_loss_value.item()
                    data_log["losses/actor_loss"] = actor_loss_value.item()
                    data_log["losses/actor_entropy"] = entropy.item()
                    data_log["losses/alpha"] = entropy_coefficient_value.item()
                    data_log["misc/steps_per_second"] = int(
                        global_step / (time.time() - start_time)
                    )
                    print(
                        "SPS:",
                        int(global_step / (time.time() - start_time)),
                        flush=True,
                    )
                    if args.autotune:
                        data_log["losses/alpha_loss"] = entropy_coefficient_loss.item()

        if args.track:
            # Perform logging
            data_log["misc/global_step"] = global_step
            wandb.log(data_log, step=global_step)

    # Close environment after training
    env.close()
