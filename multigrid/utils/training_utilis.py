""" Expected for restricted changes """

""" 
Utilities for Training

This module contains a set of utility functions to assist with training including 
model configuration, algorithm configuration, and custom callback classes for 
evaluation and weight restoration.

Note: This script is expected to have restricted changes.

"""

import os
from pathlib import Path
import numpy as np
import copy
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms import AlgorithmConfig
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel, TorchCentralizedCriticModel
from multigrid.envs import CONFIGURATIONS
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls, registry_get_input
from gymnasium.envs import registry as gym_envs_registry
from gymnasium import spaces
import ray.rllib.algorithms.callbacks as callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.callback import Callback
from ray.rllib import BaseEnv, Policy, RolloutWorker
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.examples.policy.random_policy import RandomPolicy

from typing import Dict, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID


# The new RLModule / Learner API
from ray.rllib.core.rl_module.rl_module import SingleAgentRLModuleSpec
from ray.rllib.core.rl_module.marl_module import MultiAgentRLModuleSpec
from ray.rllib.examples.rl_module.random_rl_module import RandomRLModule


def get_checkpoint_dir(search_dir: Path | str | None) -> Path | None:
    """
    Returns the most recently modified checkpoint directory.

    Parameters
    ----------
    search_dir : Union[Path, str, None]
        The directory to search for checkpoints within.

    Returns
    -------
    Union[Path, None]
        The most recently modified checkpoint directory or None if not found.
    """
    if search_dir:
        checkpoints = Path(search_dir).expanduser().glob("**/*.is_checkpoint")
        if checkpoints:
            return sorted(checkpoints, key=os.path.getmtime)[-1].parent

    return None


def can_use_gpu() -> bool:
    """
    Check if a GPU is available for training.

    Returns
    -------
    bool
        True if a GPU is available, False otherwise.
    """
    try:
        _, tf, _ = try_import_tf()
        return tf.test.is_gpu_available()
    except:
        pass

    try:
        torch, _ = try_import_torch()
        return torch.cuda.is_available()
    except:
        pass

    return False


def model_config(framework: str = "torch", lstm: bool = False, custom_model_config: dict = None):
    """
    Returns a model configuration dictionary for RLlib.

    Parameters
    ----------
    framework : str, optional
        The deep learning framework to use, default is "torch".
    lstm : bool, optional
        Whether to use LSTM model, default is False.
    custom_model_config : dict, optional
        Custom model configuration.

    Returns
    -------
    dict
        Model configuration dictionary for RLlib.
    """
    if framework == "torch":
        if lstm:
            model = TorchLSTMModel
        elif custom_model_config["training_scheme"] == "CTDE":
            model = TorchCentralizedCriticModel
        else:
            model = TorchModel
    else:
        if lstm:
            raise NotImplementedError
        else:
            model = TFModel

    return {
        "custom_model": model,
        "custom_model_config": custom_model_config,
        "conv_filters": [
            [16, [3, 3], 1],
            [32, [3, 3], 1],
            [64, [3, 3], 1],
        ],
        "fcnet_hiddens": [64, 64],
        "post_fcnet_hiddens": [],
        "lstm_cell_size": 256,
        "max_seq_len": 20,
    }


def self_play_policy_mapping_fn(agent_id, episode, worker, **kwargs):
    # agent_id = [0|1] -> policy depends on episode ID
    # This way, we make sure that both policies sometimes play agent0
    # (start player) and sometimes agent1 (player to move 2nd).

    if not episode:
        return agent_id
    else:
        return "red_0" if episode.episode_id % 2 == agent_id else "blue_0"


def algorithm_config(
    algo: str = "PPO",
    env: str = "MultiGrid-Empty-8x8-v0",
    env_config: dict = {},
    framework: str = "torch",
    lstm: bool = False,
    num_workers: int = 0,
    num_gpus: int = 0,
    lr: float | None = None,
    policies_to_train: list[int] | None = None,
    policies_map: dict = {},
    team_policies_mapping: dict = {},
    using_self_play: bool = False,
    **kwargs,
) -> AlgorithmConfig:
    """
    Returns the RL algorithm configuration dictionary.

    Parameters
    ----------
    algo : str, optional
        The name of the RLlib-registered algorithm to use. Default is "PPO".
    env : str, optional
        Environment to use. Default is "MultiGrid-Empty-8x8-v0".
    env_config : dict, optional
        Environment configuration dictionary. Default is empty dict.
    framework : str, optional
        Deep learning framework to use. Default is "torch".
    lstm : bool, optional
        Whether to use LSTM model. Default is False.
    num_workers : int, optional
        Number of rollout workers. Default is 0.
    num_gpus : int, optional
        Number of GPUs to use. Default is 0.
    lr : float or None, optional
        Learning rate. Default is None.
    policies_to_train : list[int] or None, optional
        List of policies to train. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    AlgorithmConfig
        The RL algorithm configuration.

    """

    # HW3 NOTE - Set up individual env_config and algorithm_training_config
    # from ray.tune.registry import _global_registry, ENV_CREATOR
    # env_creator = _global_registry.get(ENV_CREATOR, env)
    # env_config = registry_get_input(name=env)
    # env_config = gym_envs_registry[env].kwargs
    _, env_config = CONFIGURATIONS[env]

    env_config["policies_map"] = {}
    env_config["team_policies_mapping"] = team_policies_mapping
    algorithm_training_config = {}

    # HW3 NOTE - Update Reward Scheme

    if policies_map:
        for policy_id, this_policy in policies_map.items():
            env_config["reward_schemes"][policy_id] = this_policy.reward_schemes[policy_id]
            env_config["policies_map"][policy_id] = this_policy
            algorithm_training_config[policy_id] = this_policy.algorithm_training_config[policy_id]

    # # HW2 NOTE:
    # # ====== Extract PG and PPO specific configurations from kwargs ===== #
    # # PG-specific parameters used in RLlib:
    # # https://github.com/ray-project/ray/blob/master/rllib/algorithms/pg/pg.py#L66-L95
    # # e.g.
    # # gamma=0.9,
    # pg_config = kwargs.get("algorithm_training_config", {}).get("PG_params", {})

    # # PPO-specific parameters used in RLlib:
    # # https://github.com/ray-project/ray/blob/master/rllib/algorithms/ppo/ppo.py#L60-L126
    # # e.g.
    # # lambda_=1.0,
    # # kl_coeff=0.2,
    # # kl_target=0.01,
    # # clip_param=0.3,
    # # grad_clip=None,
    # # vf_clip_param = 10.0,
    # # vf_loss_coeff=0.5,
    # # entropy_coeff=0.001,
    # # sgd_minibatch_size=128,
    # # num_sgd_iter=30,
    # ppo_config = kwargs.get("algorithm_training_config", {}).get("PPO_params", {})

    # HW3 NOTE:
    policies = {}
    # Ensure to sort teams alphabetically since the rest of the operatation are impliclty ordered
    env_config["teams"] = dict(sorted(env_config["teams"].items()))
    for team_name, team_num in env_config["teams"].items():
        if env_config["training_scheme"] == "CTCE":
            if team_name in list(algorithm_training_config.keys()):
                policies[team_name] = PolicySpec(
                    # policy_class=get_trainable_cls(algorithm_training_config[team_name]["algo"]), # Future investigation - Do we need a different trainer for using different algo?
                    config=algorithm_training_config[team_name]["algo_config_class"].overrides(
                        **algorithm_training_config[team_name]["algo_config"]
                    ),
                    # observation_space=...,
                    # action_space=...,
                )
            else:
                policies[team_name] = PolicySpec()
        else:
            for i in range(team_num):
                if f"{team_name}_{i}" in list(algorithm_training_config.keys()):
                    policies[f"{team_name}_{i}"] = PolicySpec(
                        # policy_class=get_trainable_cls(algorithm_training_config[f"{team_name}_{i}"]["algo"]), # Future investigation - Do we need a different trainer for using different algo?
                        config=algorithm_training_config[f"{team_name}_{i}"]["algo_config_class"].overrides(
                            **algorithm_training_config[f"{team_name}_{i}"]["algo_config"]
                        ),
                        # observation_space=...,
                        # action_space=...,
                    )
                else:
                    policies[f"{team_name}_{i}"] = PolicySpec()

    return (
        get_trainable_cls(algo)
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: agent_id,
            policies_to_train=policies_to_train,
        )
        .training(
            model=model_config(
                framework=framework,
                lstm=lstm,
                custom_model_config={"teams": env_config["teams"], "training_scheme": env_config["training_scheme"]},
            ),
            lr=(lr or NotProvided),
        )
    )


class EvaluationCallbacks(DefaultCallbacks, Callback):
    """
    Custom Callback for Evaluation Metrics.

    Custom callback class for collecting and processing evaluation metrics
    during the training of the agents.
    """

    def on_episode_step(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Optional[Dict[PolicyID, Policy]] = None,
        episode: Union[Episode, EpisodeV2],
        env_index: Optional[int] = None,
        **kwargs,
    ):
        """
        Callback for on episode step.

        Parameters
        ----------
        worker : RolloutWorker
            The rollout worker.
        base_env : BaseEnv
            The base environment.
        policies : dict[PolicyID, Policy], optional
            The policies.
        episode : Union[Episode, EpisodeV2]
            The episode.
        env_index : int, optional
            The environment index.

        """
        info = episode._last_infos
        for a_key in info.keys():
            if a_key != "__common__":
                for b_key in info[a_key]:
                    try:
                        episode.user_data[f"{a_key}/{b_key}"].append(info[a_key][b_key])
                    except KeyError:
                        episode.user_data[f"{a_key}/{b_key}"] = [info[a_key][b_key]]

    def on_episode_end(
        self,
        *,
        worker: "RolloutWorker",
        base_env: BaseEnv,
        policies: Dict[PolicyID, Policy],
        episode: Union[Episode, EpisodeV2, Exception],
        env_index: Optional[int] = None,
        **kwargs,
    ):
        """
        Callback for on episode step.

        Parameters
        ----------
        worker : RolloutWorker
            The rollout worker.
        base_env : BaseEnv
            The base environment.
        policies : dict[PolicyID, Policy], optional
            The policies.
        episode : Union[Episode, EpisodeV2]
            The episode.
        env_index : int, optional
            The environment index.

        """
        info = episode._last_infos
        for a_key in info.keys():
            if a_key != "__common__":
                for b_key in info[a_key]:
                    metric = np.array(episode.user_data[f"{a_key}/{b_key}"])
                    episode.custom_metrics[f"{a_key}/{b_key}"] = np.sum(metric).item()


class RestoreWeightsCallback(DefaultCallbacks, Callback):
    """
    Custom Callback for Restoring Weights.

    Custom callback class for restoring policy weights from checkpoints during
    the training of the agents.
    """

    def __init__(
        self,
        load_dir: str,
        load_policy_names: list[str],
    ):
        """
        Initialize RestoreWeightsCallback.

        Parameters
        ----------
        load_dir : str
            The directory where the checkpoints are stored.
        load_policy_names : list[str]
            The list of names of policies to restore.
        """
        self.load_dir = load_dir
        self.load_policy_names = load_policy_names

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        """
        Callback for algorithm initialization.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being initialized.
        """

        algorithm.set_weights(self.restored_policy_weights)

    def setup(self, *args, **kwargs):
        """
        Setup callback, called once at the beginning of the training.

        Parameters
        ----------
        args : tuple
            Additional positional arguments.
        kwargs : dict
            Additional keyword arguments.
        """
        checkpoint_path = get_checkpoint_dir(self.load_dir)
        restored_policies = Policy.from_checkpoint(checkpoint_path)
        self.restored_policy_weights = {
            policy_name: restored_policies[policy_name].get_weights() for policy_name in self.load_policy_names
        }


class SelfPlayCallback(DefaultCallbacks, Callback):
    def __init__(
        self, policy_to_train: str = "red_0", opponent_policy: str = "blue_0", win_rate_threshold: float = 0.6
    ):
        super().__init__()
        # 0=RandomPolicy, 1=1st main policy snapshot,
        # 2=2nd main policy snapshot, etc..
        self.current_opponent = 0
        self.policy_to_train = policy_to_train
        self.opponent_policy = opponent_policy
        self.win_rate_threshold = win_rate_threshold

    def on_train_result(self, *, algorithm, result, **kwargs):
        # Get the win rate for the train batch.
        # Note that normally, one should set up a proper evaluation config,
        # such that evaluation always happens on the already updated policy,
        # instead of on the already used train_batch.
        main_rew = result["hist_stats"].pop(f"policy_{self.policy_to_train}_reward")
        # opponent_rew = result["hist_stats"]["policy_blue_0_reward"]
        opponent_rew = list(result["hist_stats"].values())[0]
        assert len(main_rew) == len(opponent_rew)
        won = 0
        for r_main, r_opponent in zip(main_rew, opponent_rew):
            if r_main > r_opponent:
                won += 1
        win_rate = won / len(main_rew)
        result["win_rate"] = win_rate
        print(f"Iter={algorithm.iteration} win-rate={win_rate} -> ", end="")
        # If win rate is good -> Snapshot current policy and play against
        # it next, keeping the snapshot fixed and only improving the "main"
        # policy.
        if win_rate > self.win_rate_threshold:
            self.current_opponent += 1
            new_pol_id = f"{self.policy_to_train}_v{self.current_opponent}"
            print(f"adding new opponent to the mix ({new_pol_id}).")

            # Re-define the mapping function, such that "main" is forced
            # to play against any of the previously played policies
            # (excluding "random").
            def policy_mapping_fn(agent_id, episode, worker, **kwargs):
                # agent_id = [0|1] -> policy depends on episode ID
                # This way, we make sure that both policies sometimes play
                # (start player) and sometimes agent1 (player to move 2nd).
                if agent_id == self.policy_to_train:
                    return self.policy_to_train
                else:
                    return (
                        self.opponent_policy
                        if episode.episode_id % 2 == hash(agent_id) % 2  # int(agent_id.split("_")[1])
                        else "{}_v{}".format(
                            self.policy_to_train, np.random.choice(list(range(1, self.current_opponent + 1)))
                        )
                    )

            new_config = copy.deepcopy(algorithm.get_policy(self.policy_to_train).config)
            new_config.pop("worker_index")
            new_config.pop("__policy_id")
            new_config.pop("__stdout_file__")
            new_policy = algorithm.add_policy(
                policy_id=new_pol_id,
                policy_cls=type(algorithm.get_policy(self.policy_to_train)),
                policy_mapping_fn=policy_mapping_fn,
                config=new_config,
                observation_space=algorithm.get_policy(self.policy_to_train).observation_space,
                action_space=algorithm.get_policy(self.policy_to_train).action_space,
            )

            # Set the weights of the new policy to the main policy.
            # We'll keep training the main policy, whereas `new_pol_id` will
            # remain fixed.
            main_state = algorithm.get_policy(self.policy_to_train).get_state()
            new_policy.set_state(main_state)
            # We need to sync the just copied local weights (from main policy)
            # to all the remote workers as well.
            algorithm.workers.sync_weights()
        else:
            print("not good enough; will keep learning ...")

        # +2 = red_0 + blue_0
        result["league_size"] = self.current_opponent + 2
