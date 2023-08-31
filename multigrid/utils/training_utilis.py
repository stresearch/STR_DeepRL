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
from ray.rllib.utils.framework import try_import_tf, try_import_torch
from ray.rllib.algorithms import AlgorithmConfig
from multigrid.rllib.models import TFModel, TorchModel, TorchLSTMModel, TorchCentralizedCriticModel
from ray.rllib.utils.from_config import NotProvided
from ray.tune.registry import get_trainable_cls
from gymnasium.envs import registry as gym_envs_registry
import ray.rllib.algorithms.callbacks as callbacks
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.tune.callback import Callback
from ray.rllib import BaseEnv, Policy, RolloutWorker
from typing import Dict, Optional, Union
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.utils.typing import AgentID, EnvType, PolicyID


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
    our_agent_ids: list[str] | None = None,
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
    our_agent_ids : list[str] or None, optional
        List of agent ids. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    AlgorithmConfig
        The RL algorithm configuration.

    """

    env_config = gym_envs_registry[env].kwargs

    return (
        get_trainable_cls(algo)
        .get_default_config()
        .environment(env=env, env_config=env_config)
        .framework(framework)
        .rollouts(num_rollout_workers=num_workers)
        .resources(num_gpus=num_gpus)
        .multi_agent(
            policies={team_name for team_name in list(env_config["teams"].keys())}
            if env_config["training_scheme"] == "CTCE"
            else {f"{team_name}_{i}" for team_name, team_num in env_config["teams"].items() for i in range(team_num)},
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
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
        policy_name: str,
    ):
        """
        Initialize RestoreWeightsCallback.

        Parameters
        ----------
        load_dir : str
            The directory where the checkpoints are stored.
        policy_name : str
            The name of the policy to restore.
        """
        self.load_dir = load_dir
        self.policy_name = policy_name

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        """
        Callback for algorithm initialization.

        Parameters
        ----------
        algorithm : Algorithm
            The algorithm being initialized.
        """
        algorithm.set_weights({self.policy_name: self.restored_policy_0_weights})

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
        policy_0_checkpoint_path = get_checkpoint_dir(self.load_dir)
        restored_policy_0 = Policy.from_checkpoint(policy_0_checkpoint_path)
        self.restored_policy_0_weights = restored_policy_0[self.policy_name].get_weights()
