from __future__ import annotations

""" Expected for restricted changes """


"""Script for Training Deep Reinforcement Learning agents in MultiGrid environment.

This script provides a streamlined way to configure and train RL agents
using Ray's RLlib library. The script allows for a variety of command-line options,
including algorithm selection, environment setup, and more.

Note: This script is expected to have restricted changes.

"""

import argparse
import json
import pathlib
from pprint import pprint
import git
from pathlib import Path
import os
import subprocess
import ray
from ray import tune
from ray.rllib.algorithms import AlgorithmConfig
from ray.tune import CLIReporter
from ray.air.integrations.mlflow import MLflowLoggerCallback

from multigrid.utils.training_utilis import (
    algorithm_config,
    get_checkpoint_dir,
    EvaluationCallbacks,
    RestoreWeightsCallback,
)
from multigrid.rllib.ctde_torch_policy import CentralizedCritic

# Set the working diretory to the repo root
REPO_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
os.chdir(REPO_ROOT)

# Constants
SUBMISSION_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/submission_config.json"), key=os.path.getmtime
)[-1]

with open(SUBMISSION_CONFIG_FILE, "r") as file:
    submission_config_data = file.read()

submission_config = json.loads(submission_config_data)

SUBMITTER_NAME = submission_config["name"]

TAGS = {"user_name": SUBMITTER_NAME, "git_commit_hash": git.Repo(REPO_ROOT).head.commit}

ALGORITHM_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/configs/algorithm_training_config.json"), key=os.path.getmtime
)[-1]

with open(ALGORITHM_CONFIG_FILE, "r") as file:
   algorithm_training_config_data = file.read()

algorithm_training_config = json.loads(algorithm_training_config_data)


# Initialize the CLI reporter for Ray
reporter = CLIReporter(max_progress_rows=10, max_report_frequency=30)


def configure_algorithm(args):
    """
    Create an algorithm configuration object based on command-line arguments.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments.

    Returns
    -------
    AlgorithmConfig
        The constructed algorithm configuration object.
    """
    config = algorithm_config(**vars(args))
    config.seed = args.seed
    config.callbacks(EvaluationCallbacks)
    config.environment(disable_env_checking=False)

    return config


def train(
    algo: str,
    config: AlgorithmConfig,
    stop_conditions: dict,
    save_dir: str,
    load_dir: str | None = None,
    local_mode: bool = False,
    experiment_name: str = "testing_experiment",
    training_scheme: str = "CTCE",
    policies_to_load: list[str] | None = None,
    restore_all_policies_from_checkpoint: bool = False,
):
    """Main training loop for RLlib algorithms.

    This function initializes Ray, runs the training loop, and handles
    checkpoints, logging, and custom callbacks.

    Parameters
    ----------
    algo : str
        The RL algorithm to use.
    config : AlgorithmConfig
        Configuration object for the algorithm.
    stop_conditions : dict
        Conditions to stop the training.
    save_dir : str
        Directory to save checkpoints and logs.
    load_dir : str, optional
        Directory to load pre-trained model checkpoints from.
    local_mode : bool, optional
        Whether to run Ray in local mode (for debugging).
    experiment_name : str, optional
        Descriptive name for the experiment, used for logging purposes.
    training_scheme : str, optional
        Training scheme, can be either 'CTCE', 'DTDE', or 'CTDE'.
    policies_to_load : list[str], optional
        List of policy names to restore from checkpoint if `load_dir` is provided.
    restore_all_policies_from_checkpoint : bool, optional
        If True, restores all policies from the specified checkpoint in `load_dir`. Otherwise, only restores specified policies in `policies_to_load`.

    Returns
    -------
    None
    """

    # Assemble the list of callbacks
    callbacks = [
        # Logger callback for MLflow integration
        MLflowLoggerCallback(
            tracking_uri="./submission/mlflow", experiment_name=experiment_name, tags=TAGS, save_artifact=True
        ),
    ]

    # Add a callback to restore specific policy weights if policies are specified
    if policies_to_load:
        callbacks.append(RestoreWeightsCallback(load_dir=load_dir, load_policy_names=policies_to_load))

    # Initialize Ray
    ray.init(num_cpus=(config.num_rollout_workers + 1), local_mode=local_mode)

    # Run the training loop using Ray's `tune` API
    tune.run(
        CentralizedCritic if training_scheme == "CTDE" else algo,
        stop=stop_conditions,
        config=config,
        local_dir=save_dir,
        verbose=1,
        # If `restore_all_policies_from_checkpoint` is True, restore all policies from checkpoint
        # This is helpful for continue to train your agent from last checkpoint
        # But ensure to update the stop_conditions to allow you train beyond the original conditions
        restore=get_checkpoint_dir(load_dir) if restore_all_policies_from_checkpoint else None,
        checkpoint_freq=10,
        checkpoint_at_end=True,
        progress_reporter=reporter,
        callbacks=callbacks,
        name=experiment_name,
    )

    # Shutdown Ray once training is completed
    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        "--framework", type=str, choices=["torch", "tf", "tf2"], default="torch", help="Deep learning framework to use."
    )
    parser.add_argument("--lstm", action="store_true", help="Use LSTM model.")
    parser.add_argument(
        "--env", type=str, default="MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1", help="MultiGrid environment to use."
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')",
    )
    # Future todo - have an more user friendly implementation of config, perhaps using ymal files
    parser.add_argument(
        "--algorithm-training-config",
        type=json.loads,
        default=algorithm_training_config, 
        help="Deep RL Algorithm Specific config dict, given as a JSON string (e.g. '{\"PG_params\": { \"lr\" : 0.001},  \"PPO_params\": { \"entropy_coeff\" : 0.001}}')",

    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Set the random seed of each worker. This makes experiments reproducible"
    )
    parser.add_argument("--num-workers", type=int, default=10, help="Number of rollout workers.")
    parser.add_argument("--num-gpus", type=int, default=0, help="Number of GPUs to train on.")
    parser.add_argument("--num-timesteps", type=int, default=1e6, help="Total number of timesteps to train.")
    parser.add_argument("--lr", type=float, help="Learning rate for training.")
    parser.add_argument(
        "--load-dir",
        type=str,
        default="submission/pretrained_checkpoints/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1_154ab_00000_0_2023-09-12_16-08-06/checkpoint_000250",
        help="Checkpoint directory for loading pre-trained policies.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="submission/ray_results/",
        help="Directory for saving checkpoints, results, and trained policies.",
    )
    parser.add_argument(
        "--name", type=str, default="<my_experinemnt>", help="Distinct name to track your experinemnt in save-dir"
    )
    parser.add_argument(
        "--local-mode", type=bool, default=False, help="Boolean value to set to use local mode for debugging"
    )
    parser.add_argument(
        "--policies-to-train", nargs="+", type=str, default=["red_0"], help="List of agent ids to train"
    )
    parser.add_argument(
        "--policies-to-load", nargs="+", type=str, default=["blue_0"], help="List of agent ids to train"
    )
    parser.add_argument("--training-scheme", type=str, default="DTDE", help="Can be either 'CTCE', 'DTDE' or 'CTDE'")

    args = parser.parse_args()
    args.multiagent = {}
    args.multiagent["policies_to_train"] = args.policies_to_train
    config = configure_algorithm(args)
    stop_conditions = {"timesteps_total": args.num_timesteps}

    print()
    print(f"Running with following CLI options: {args}")
    print("\n", "-" * 64, "\n", "Training with following configuration:", "\n", "-" * 64)
    print()
    pprint(config.to_dict())
    train(
        algo=args.algo,
        config=config,
        stop_conditions=stop_conditions,
        save_dir=args.save_dir,
        load_dir=args.load_dir,
        local_mode=args.local_mode,
        experiment_name=args.name,
        training_scheme=args.training_scheme,
        policies_to_load=args.policies_to_load,
    )
