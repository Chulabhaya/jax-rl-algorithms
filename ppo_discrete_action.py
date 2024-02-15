import os
import random
import time
from dataclasses import dataclass
from typing import Tuple
from functools import partial

import flax
import flax.linen as nn
import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from flax.training.train_state import TrainState
from flax.linen.initializers import constant, orthogonal
import wandb


@dataclass
class Args:
    """Arguments for running discrete-action PPO."""

    # General arguments
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """name of this experiment"""
    seed: int = 1
    """seed of the experiment"""

    # Weights & Biases specific arguments
    track: bool = True
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project: str = "jaxtest1"
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
    learning_rate: float = 2.5e-4
    """learning rate of the optimizer"""
    num_envs: int = 4
    """number of parallel game environments"""
    num_steps: int = 128
    """number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """discount factor gamma"""
    gae_lambda: float = 0.95
    """lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """number of minibatches"""
    update_epochs: int = 4
    """K epochs to update the policy"""
    norm_adv: bool = True
    """toggle advantage normalization"""
    clip_coef: float = 0.2
    """surrogate clipping coefficient"""
    clip_vloss: bool = True
    """toggles whether or not to use a clipped loss for the value function"""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """maximum norm for the gradient clipping"""
    target_kl: float = None
    """target KL divergence threshold"""

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

    # To be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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

    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        return env

    return thunk


class Critic(nn.Module):
    """Model for representing the PPO critic."""

    @nn.compact
    def __call__(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Computes state-values for input observations.

        Returns
        -------
        state_values : jnp.ndarray
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
    """Model for representing the PPO actor.

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


@jax.jit
def sample_actions(
    logits: jnp.ndarray,
    sampling_key: jax.Array,
):
    """
    Calculates actions and probabilities.

    Parameters
    ----------
    logits : jnp.ndarray
        Action logits output from actor network.
    sampling_key : jax.Array
        RNG key.

    Returns
    -------
    actions : jnp.ndarray
        Sampled actions from action distributions.
    log_action_probs : jnp.ndarray
        Logs of action probabilities for sampled actions.
    entropy : jnp.ndarray
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


@jax.jit
def give_actions(
    logits: jnp.ndarray,
    actions: np.ndarray,
):
    """
    Calculates probabilities given actions.

    Parameters
    ----------
    logits : jnp.ndarray
        Action logits output from actor network.
    actions : np.ndarray
        Given input actions to use instead of sampling

    Returns
    -------
    actions : np.ndarray
        Return input given actions.
    log_action_probs : jnp.ndarray
        Logs of action probabilities for given actions.
    entropy : jnp.ndarray
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


@flax.struct.dataclass
class Storage:
    """Stores a batch of data."""

    obs: jnp.array
    actions: jnp.array
    logprobs: jnp.array
    rewards: jnp.array
    terminateds: jnp.array
    values: jnp.array
    advantages: jnp.array
    returns: jnp.array


if __name__ == "__main__":
    args = tyro.cli(Args)

    # Initialize batch size, minibatch size, and number of training iterations
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size

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
                mode="online",
            )

    # Seeding
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)

    # Initialize environment
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i) for i in range(args.num_envs)]
    )

    # Set up learning rate scheduler
    def linear_schedule(iteration):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches * args.update_epochs) gradient updates
        frac = (
            1.0
            - (iteration // (args.num_minibatches * args.update_epochs))
            / args.num_iterations
        )
        return args.learning_rate * frac

    # Initialize models
    obs, _ = envs.reset(seed=args.seed)
    vf = Critic()
    key, critic_key = jax.random.split(key, 2)
    vf_state = TrainState.create(
        apply_fn=vf.apply,
        params=vf.init(critic_key, obs),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                eps=1e-5,
            ),
        ),
    )
    vf.apply = jax.jit(vf.apply)

    actor = Actor(action_dim=envs.single_action_space.n)
    key, actor_key = jax.random.split(key, 2)
    actor_state = TrainState.create(
        apply_fn=actor.apply,
        params=actor.init(actor_key, obs),
        tx=optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=linear_schedule if args.anneal_lr else args.learning_rate,
                eps=1e-5,
            ),
        ),
    )
    actor.apply = jax.jit(actor.apply)

    def compute_gae_once(
        carry: jnp.ndarray,
        input: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Calculates a single iteration of GAE.

        Parameters
        ----------
        carry : jnp.ndarray
            Carry from the previous iteration.
        input : Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Current input of this iteration.
        gamma : float
            Discount factor gamma.
        gae_lambda : float
            GAE bias/variance tradeoff value lambda.

        Returns
        -------
        Tuple[jnp.ndarray, jnp.ndarray]
            Updated carry and current result.
        """
        advantages = carry
        nextdone, nextvalues, curvalues, reward = input
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    # Initialize function for computing single iteration of GAE
    compute_gae_once = partial(
        compute_gae_once, gamma=args.gamma, gae_lambda=args.gae_lambda
    )

    @jax.jit
    def compute_gae(
        vf_state: TrainState,
        next_obs: np.ndarray,
        next_terminated: np.ndarray,
        storage: Storage,
    ) -> Storage:
        """Calculated Generalized Advantage Estimation (GAE).

        Parameters
        ----------
        vf_state : TrainState
            State of the critic network.
        next_obs : np.ndarray
            Next observations.
        next_terminated : np.ndarray
            Next terminateds.
        storage : Storage
            Storage containing batch of data.

        Returns
        -------
        Storage
            Updated batch with added advantage and return data.
        """
        # Calculate next value
        next_value = vf.apply(vf_state.params, next_obs).squeeze()

        # Calculate advantages and returns using GAE
        advantages = jnp.zeros((args.num_envs,))
        terminateds = jnp.concatenate(
            [storage.terminateds, next_terminated[None, :]], axis=0
        )
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            compute_gae_once,
            advantages,
            (terminateds[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        storage = storage.replace(
            advantages=advantages,
            returns=advantages + storage.values,
        )
        return storage

    # Define update functions here to limit the need for static argname
    def update_critic(
        critic_state: TrainState,
        observations: jnp.ndarray,
        returns: jnp.ndarray,
    ) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray]:
        """Updates the critic.

        Parameters
        ----------
        critic_state : TrainState
            State of the critic network.
        observations : jnp.ndarray
            Observations.
        returns : jnp.ndarray
            Returns.

        Returns
        -------
        Tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Results from updating the critic.
        """

        def critic_loss(critic_params: dict) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """
            Computes the critic loss for updating the critic.

            Parameters:
            -----------
            critic_params : dict
                The parameters of the critic model.

            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray]
                A tuple containing the various loss values.
            """
            # Get state-values from critic
            new_values = vf.apply(critic_params, observations)
            new_values = new_values.squeeze()

            # Calculate value loss
            v_loss = 0.5 * ((new_values - returns) ** 2).mean()

            # Calculate total critic loss
            critic_loss = v_loss * args.vf_coef

            return critic_loss, v_loss

        # Calculate values and gradients of loss with respect to critic network params
        (critic_loss_value, v_loss_value), grads = jax.value_and_grad(
            critic_loss, has_aux=True
        )(critic_state.params)

        # Apply gradients to actor network
        critic_state = critic_state.apply_gradients(grads=grads)

        return critic_state, (critic_loss_value, v_loss_value)

    def update_actor(
        actor_state: TrainState,
        observations: jnp.ndarray,
        actions: jnp.ndarray,
        logprobs: jnp.ndarray,
        advantages: jnp.ndarray,
    ) -> Tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Updates the actor.

        Parameters
        ----------
        actor_state : TrainState
            State of the actor network.
        observations : jnp.ndarray
            Observations.
        actions : jnp.ndarray
            Actions.
        logprobs : jnp.ndarray
            Log probabilities of actions.
        advantages : jnp.ndarray
            Advantages.

        Returns
        -------
        Tuple[TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
            Results from updating the actor.
        """

        def actor_loss(
            actor_params: dict,
        ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            """
            Computes the actor loss for updating the actor.

            Parameters:
            -----------
            actor_params : dict
                The parameters of the actor model.

            Returns:
            --------
            Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
                A tuple containing the actor loss and the entropy.
            """

            # Calculate new log probabilities and entropies
            logits = actor.apply(actor_params, observations)
            _, newlogprobs, entropies = give_actions(logits, actions)

            # Calculate approximate KL divergence between old vs. new distribution
            logratio = newlogprobs - logprobs
            ratio = jnp.exp(logratio)
            approx_kl = (
                (ratio - 1) - logratio
            ).mean()  # From: http://joschu.net/blog/kl-approx.html

            # Normalize advantages, if specified
            norm_advantages = (
                (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                if args.norm_adv
                else advantages
            )

            # Calculate policy loss
            pg_loss1 = -norm_advantages * ratio
            pg_loss2 = -norm_advantages * jnp.clip(
                ratio, 1 - args.clip_coef, 1 + args.clip_coef
            )
            pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

            # Calculate entropy loss
            entropy_loss = entropies.mean()

            # Calculate total actor loss
            actor_loss = pg_loss - args.ent_coef * entropy_loss

            return actor_loss, (pg_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

        # Calculate values and gradients of loss with respect to actor network params
        (actor_loss_value, (pg_loss_value, entropy_loss_value, approx_kl)), grads = (
            jax.value_and_grad(actor_loss, has_aux=True)(actor_state.params)
        )

        # Apply gradients to actor network
        actor_state = actor_state.apply_gradients(grads=grads)

        return (
            actor_state,
            (actor_loss_value, pg_loss_value, entropy_loss_value, approx_kl),
        )

    @jax.jit
    def update_ppo(
        actor_state: TrainState,
        vf_state: TrainState,
        storage: Storage,
        key: jax.Array,
    ):
        def update_epoch(
            carry: Tuple[TrainState, TrainState, jax.Array], unused_input
        ) -> Tuple[
            TrainState,
            TrainState,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jnp.ndarray,
            jax.Array,
        ]:
            """Training epoch that updates the actor and critic networks.

            Parameters
            ----------
            carry : Tuple[TrainState, TrainState, jax.Array]
                Actor and critic train states as well as RNG key.
            unused_input : Any
                Unused input, can be of any type.

            Returns
            -------
            Tuple[ TrainState, TrainState, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jax.Array, ]
                Updated actor and critic train states as well as loss values.
            """
            actor_state, vf_state, key = carry
            key, minibatch_sampling_key = jax.random.split(key)

            def flatten(x: jnp.ndarray) -> jnp.ndarray:
                """Flattens batch of data, specifically converts shape
                (num_steps, num_envs) to (num_steps * num_envs, ).

                Parameters
                ----------
                x : jnp.ndarray
                    Batch of data to be flattened.

                Returns
                -------
                jnp.ndarray
                    Flattened batch of data.
                """
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray) -> jnp.ndarray:
                """Shuffles a flattened batch of data, then
                splits it into minibatches.

                Parameters
                ----------
                x : jnp.ndarray
                    Flattened batch of data

                Returns
                -------
                jnp.ndarray
                    Minibatches of data.
                """
                x = jax.random.permutation(minibatch_sampling_key, x)
                x = jnp.reshape(x, (args.num_minibatches, -1) + x.shape[1:])
                return x

            # Flattens batch
            flatten_storage = jax.tree_map(flatten, storage)

            # Splits flattened batch into minibatches
            shuffled_storage = jax.tree_map(convert_data, flatten_storage)

            def update_minibatch(
                carry: Tuple[TrainState, TrainState], input: Storage
            ) -> Tuple[
                Tuple[TrainState, TrainState],
                Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
            ]:
                """Updates the actor and critic using a minibatch of data.

                Parameters
                ----------
                carry : Tuple[TrainState, TrainState]
                    Current train states of actor and critic.
                input : Storage
                    Minibatch of data.

                Returns
                -------
                Tuple[ Tuple[TrainState, TrainState], Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray], ]
                    Updated actor and critic states, as well as loss values.
                """
                actor_state, vf_state = carry

                # Update actor
                actor_state, (actor_loss, pg_loss, entropy_loss, approx_kl) = (
                    update_actor(
                        actor_state,
                        input.obs,
                        input.actions,
                        input.logprobs,
                        input.advantages,
                    )
                )

                # Update critic
                vf_state, (critic_loss, v_loss) = update_critic(
                    vf_state,
                    input.obs,
                    input.returns,
                )

                # Calculate summed loss
                loss = actor_loss + critic_loss
                return (actor_state, vf_state), (
                    loss,
                    pg_loss,
                    v_loss,
                    entropy_loss,
                    approx_kl,
                )

            # Update actor and critic networks with minibatches of data
            (actor_state, vf_state), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
            ) = jax.lax.scan(
                update_minibatch, (actor_state, vf_state), shuffled_storage
            )
            return (actor_state, vf_state, key), (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
            )

        # Update actor and critic over multiple epochs of training
        (actor_state, vf_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = jax.lax.scan(
            update_epoch, (actor_state, vf_state, key), (), length=args.update_epochs
        )
        return (
            actor_state,
            vf_state,
            (loss, pg_loss, v_loss, entropy_loss, approx_kl),
            key,
        )

    # Initialize storage for batches
    obs = np.zeros(
        (args.num_steps, args.num_envs) + envs.single_observation_space.shape,
        dtype=np.float32,
    )
    actions = np.zeros(
        (args.num_steps, args.num_envs) + envs.single_action_space.shape, dtype=np.int64
    )
    logprobs = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    rewards = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)
    terminateds = np.zeros((args.num_steps, args.num_envs), dtype=bool)
    values = np.zeros((args.num_steps, args.num_envs), dtype=np.float32)

    # Start time tracking for run
    start_time = time.time()

    # Start the game
    global_step = 0
    next_obs, info = envs.reset(seed=args.seed)
    next_terminated = np.zeros(args.num_envs, dtype=np.bool_)
    for iteration in range(1, args.num_iterations + 1):
        # Store values for data logging for each global step
        data_log = {}

        # Collect batch of timesteps from multiple environments
        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            terminateds[step] = next_terminated

            # Sample actions from actor
            key, sampling_key = jax.random.split(key, 2)  # Split key for RNG
            logits = actor.apply(actor_state.params, next_obs)
            action, logprob, _ = sample_actions(logits, sampling_key)
            actions[step] = action
            logprobs[step] = logprob

            # Get state-values from critic
            value = vf.apply(vf_state.params, next_obs)
            values[step] = value.squeeze()

            # Take step in environments
            next_obs, reward, terminated, truncated, infos = envs.step(np.array(action))
            rewards[step] = reward
            next_terminated = terminated

            # Handle episode end, record rewards for plotting purposes
            if "final_info" not in infos:
                continue  # Only print when at least 1 env is done
            for info in infos["final_info"]:
                if info is None:
                    continue  # Skip the envs that are not done
                print(
                    f"global_step={global_step}, episodic_return={info['episode']['r']}"
                )
                data_log["misc/episodic_return"] = info["episode"]["r"]
                data_log["misc/episodic_length"] = info["episode"]["l"]

        # Do Generalized Advantaged Estimation (GAE) and calculate advantages and returns
        storage = Storage(
            obs=jnp.array(obs),
            actions=jnp.array(actions),
            logprobs=jnp.array(logprobs),
            rewards=jnp.array(rewards),
            terminateds=jnp.array(terminateds),
            values=jnp.array(values),
            returns=jnp.zeros_like(rewards),
            advantages=jnp.zeros_like(rewards),
        )
        storage = compute_gae(vf_state, next_obs, next_terminated, storage)

        # Optimize the actor and critic networks
        actor_state, vf_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl), key = (
            update_ppo(actor_state, vf_state, storage, key)
        )

        if args.track:
            # Perform logging
            data_log["misc/learning_rate"] = (
                actor_state.opt_state[1].hyperparams["learning_rate"].item()
            )
            data_log["losses/value_loss"] = v_loss[-1, -1].item()
            data_log["losses/policy_loss"] = pg_loss[-1, -1].item()
            data_log["losses/entropy"] = entropy_loss[-1, -1].item()
            data_log["losses/approx_kl"] = approx_kl[-1, -1].item()
            data_log["losses/loss"] = loss[-1, -1].item()
            data_log["misc/steps_per_second"] = int(
                global_step / (time.time() - start_time)
            )
            print("SPS:", int(global_step / (time.time() - start_time)))

            data_log["misc/global_step"] = global_step
            wandb.log(data_log, step=global_step)

    # Close environment after training
    envs.close()
