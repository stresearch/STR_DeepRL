""" Expected for restricted changes """

"""Script for performing evaluation of a trained RL policies, aka RLlib checkpoint, by visualizing their behaviors and saving
metrics. It includes functionalities for visualizing trajectories of trained agents, saving frames as a GIF, and saving evaluation metrics
to a CSV file.

Note: This script is expected to have restricted changes.

"""


import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from ray.rllib.algorithms import Algorithm
from multigrid.utils.training_utilis import algorithm_config, get_checkpoint_dir
from multigrid.agents_pool import SubmissionPolicies
import subprocess
import git
import os

from ray.rllib.algorithms import Algorithm
from ray.rllib.utils.typing import AgentID
from typing import Any, Callable, Iterable


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


EVALUATION_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/configs/evaluation_config.json"), key=os.path.getmtime
)[-1]

with open(EVALUATION_CONFIG_FILE, "r") as file:
    evaluation_config_data = file.read()

evaluation_config = json.loads(evaluation_config_data)


def save_frames_to_gif(frames: List[np.ndarray], save_path: Path, filename: str) -> None:
    """
    Saves frames as a GIF.

    Parameters
    ----------
    frames : list[np.ndarray]
        List of frames to be saved as GIF.
    save_path : Path
        Directory where the GIF will be saved.
    filename : str
        Name of the output GIF file.
    """
    import imageio

    # write to file
    print(f"Saving GIF to {save_path / filename}")
    imageio.mimsave(save_path / filename, frames)

    return save_path / filename


def save_evaluation_metrics(episodes_data: List[Dict], save_path: Path, scenario_name: str) -> None:
    """
    Saves evaluation metrics to CSV.

    Parameters
    ----------
    episodes_data : list[dict]
        List of dictionaries containing episode data.
    save_path : Path
        Directory where the CSV files will be saved.
    scenario_name : str
        Name of the scenario, used as part of the file name.
    """
    # Save episodes statistics
    episodes_df = pd.DataFrame(episodes_data)
    episodes_df.to_csv(save_path / f"{scenario_name}_episodes_data.csv", index=False)

    mean_values = episodes_df[list(episodes_df.columns[:-1])].mean()
    solved_ratio = episodes_df["Solved"].sum() / len(episodes_df)

    # Save eval summary
    eval_summary_df = pd.DataFrame(mean_values).T
    eval_summary_df["Solved Ratio"] = solved_ratio
    eval_summary_df.to_csv(save_path / f"{scenario_name}_eval_summary.csv", index=False)

    print(f"Metrics saved to {save_path}")


def get_actions(
    agent_ids: Iterable[AgentID],
    evaluating_algorithms: list[Algorithm],
    observations: dict[AgentID, Any],
    states: dict[AgentID, Any],
    policies_to_eval: list[str],
) -> tuple[dict[AgentID, Any], dict[AgentID, Any]]:
    """
    Get actions for the given agents.

    Parameters
    ----------
    agent_ids : Iterable[AgentID]
        Agent IDs for which to get actions
    algorithm : Algorithm
        RLlib algorithm instance with trained policies
    policy_mapping_fn : Callable(AgentID) -> str
        Function mapping agent IDs to policy IDs
    observations : dict[AgentID, Any]
        Observations for each agent
    states : dict[AgentID, Any]
        States for each agent

    Returns
    -------
    actions : dict[AgentID, Any]
        Actions for each agent
    states : dict[AgentID, Any]
        Updated states for each agent
    """

    actions = {}
    for agent_id in agent_ids:
        if states[agent_id]:
            actions[agent_id], states[agent_id], _ = evaluating_algorithms[agent_id].compute_single_action(
                observations[agent_id], states[agent_id], policy_id=agent_id
            )
        else:
            if agent_id in policies_to_eval:
                actions[agent_id] = evaluating_algorithms[agent_id].compute_single_action(
                    observations[agent_id], policy_id=agent_id
                )
            else:
                actions[agent_id] = evaluating_algorithms[agent_id].compute_single_action(
                    observations[agent_id],
                    policy_id=policies_to_eval[0],  # HW3 NOTE - Act as another main policy e.g. "red_0"
                )

    return actions, states


def evaluation(
    evaluating_algorithms: list[Algorithm], num_episodes: int = 100, policies_to_eval: list[str] = ["red_0"]
) -> list[np.ndarray]:
    """
    Visualizes trajectories from trained agents and collects evaluation data.

    Parameters
    ----------
    algorithm : Algorithm
        The trained algorithm to be evaluated.
    num_episodes : int, optional
        Number of episodes to visualize. Default is 100.

    Returns
    -------
    list[np.ndarray]
        List of frames of the agent's trajectory.
    list[dict]
        List of dictionaries containing episode data.
    """

    frames = []
    episodes_data = []
    # HW3 NOTE - Using the the env from the main policy
    main_policy = policies_to_eval[0]
    main_team = main_policy.split("_")[0] if len(main_policy.split("_")[0]) > 1 else "red"
    # Update policies_to_eval
    policies_to_eval = [policy for policy in policies_to_eval if main_team in policy]
    env = evaluating_algorithms[main_policy].env_creator(evaluating_algorithms[main_policy].config.env_config)

    for episode in range(num_episodes):
        print("\n", "-" * 32, "\n", "Episode", episode, "\n", "-" * 32)

        episode_rewards = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminations, truncations = {"__all__": False}, {"__all__": False}
        observations, infos = env.reset()
        states = {
            agent_id: evaluating_algorithms[main_policy].get_policy(agent_id).get_initial_state()
            for agent_id in env.get_agent_ids()
        }

        step_count = 0
        while not terminations["__all__"] and not truncations["__all__"]:
            frames.append(env.get_frame())

            actions, states = get_actions(
                env.get_agent_ids(), evaluating_algorithms, observations, states, policies_to_eval
            )

            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in rewards:
                episode_rewards[agent_id] += rewards[agent_id]

            step_count += 1

        frames.append(env.get_frame())
        opponents = [agent for agent in env.env.env.agents if agent.name not in policies_to_eval]

        done_dict = {}
        for policy_name in policies_to_eval:
            done_dict = {**done_dict, **infos[policy_name]}

        solved = any([env.env.env.red_door.is_open, all([opponent.terminated for opponent in opponents])]) and (
            env.env.env.step_count < env.max_steps
        )
        print("\n", "Rewards:", episode_rewards)
        print("\n", "Total Time Steps:", env.env.env.step_count)
        print("\n", "Solved:", solved)

        # Set episode data
        episodes_data.append(
            {**episode_rewards, **done_dict, **{"Episode Length": env.env.env.step_count, "Solved": solved}}
        )

    env.close()

    return frames, episodes_data


# Custom sorting policy_ids function
def sort_policy_id(policy_id):
    # Split policy_id to extract version number
    parts = policy_id.split("_v")
    # If version number is present, convert to int, otherwise use -1
    version = int(parts[1]) if len(parts) > 1 else -1
    return version


def main_evaluation(args):
    """
    Main function for evaluation. Sets up the environment, restores the trained algorithm,
    runs the evaluation, and saves the results.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command line arguments.
    """
    args.env_config.update(render_mode=args.render_mode)
    team_policies_mapping = args.eval_config["team_policies_mapping"]

    # HW3 NOTE - Setup Policies
    evaluating_policies = {}
    for policy_id in args.policies_to_eval:
        policy_name = team_policies_mapping[policy_id]
        eval_policy = SubmissionPolicies[policy_name](policy_id=policy_id, policy_name=policy_name)
        evaluating_policies[policy_id] = eval_policy

    evaluating_algorithms = {}
    main_policy = args.policies_to_eval[0]  # HW3 - NOTE always put the main_policy at first order
    for policy_id, policy in evaluating_policies.items():
        policy.reward_schemes[main_policy] = policy.reward_schemes[policy_id]
        policy.algorithm_training_config[main_policy] = policy.algorithm_training_config[policy_id]
        policy.policy_id = main_policy

        config = algorithm_config(
            **vars(args),
            num_workers=0,
            num_gpus=0,
            policies_map={main_policy: policy},
        )
        config.explore = False
        config.environment(disable_env_checking=True)
        # config.rl_module( _enable_rl_module_api=False)
        # config.training(_enable_learner_api=False)
        algorithm = config.build()
        evaluating_algorithms[policy_id] = algorithm

    checkpoint = get_checkpoint_dir(args.load_dir)

    # Create a Path object for the directory
    save_path = Path(args.save_dir)

    # Make sure the directory exists; if not, create it
    save_path.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        from ray.rllib.policy.policy import Policy

        for policy_id, evaluating_algo in evaluating_algorithms.items():
            if policy_id == main_policy:
                print(f"Loading main checkpoint from {checkpoint}")
                evaluating_algo.restore(checkpoint)
            else:
                if "default_DTDE_1v1_opponent_checkpoint" in args.eval_config and "DTDE-1v1" in args.env:
                    opponent_checkpoint_path = args.eval_config["default_DTDE_1v1_opponent_checkpoint"]
                    print(f"Loading opponent checkpoint from {opponent_checkpoint_path}")
                    evaluating_algo.restore(opponent_checkpoint_path)
                elif "default_CTCE_2v2_opponent_checkpoint" in args.eval_config and "CTCE-2v2" in args.env:
                    opponent_checkpoint_path = args.eval_config["default_CTCE_2v2_opponent_checkpoint"]
                    print(f"Loading opponent checkpoint from {opponent_checkpoint_path}")
                    evaluating_algo.restore(opponent_checkpoint_path)
                else:
                    evaluating_algo.restore(checkpoint)

    frames, episodes_data = evaluation(
        evaluating_algorithms, num_episodes=args.num_episodes, policies_to_eval=args.policies_to_eval
    )

    scenario_name = args.gif  # args.env.split("-v3-")[1]
    save_evaluation_metrics(episodes_data=episodes_data, save_path=save_path, scenario_name=scenario_name)

    if args.gif:
        filename = args.gif if args.gif.endswith(".gif") else f"{args.gif}.gif"
        return save_frames_to_gif(frames=frames, save_path=save_path, filename=filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="PPO", help="The name of the RLlib-registered algorithm to use.")
    parser.add_argument(
        "--framework", type=str, choices=["torch", "tf", "tf2"], default="torch", help="Deep learning framework to use."
    )
    parser.add_argument("--lstm", action="store_true", help="Use LSTM model.")
    parser.add_argument(
        "--env", type=str, default="MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red", help="MultiGrid environment to use."
    )
    parser.add_argument(
        "--env-config",
        type=json.loads,
        default={},
        help="Environment config dict, given as a JSON string (e.g. '{\"size\": 8}')",
    )
    parser.add_argument(
        "--eval-config",
        type=json.loads,
        default=evaluation_config,
        help='Evaluation config dict, given as a JSON string (e.g. \'{"team_policies_mapping": {"red_0" : "your_policy_name" , "blued_0" : "your_policy_name" }}\')',
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to visualize.")
    parser.add_argument("--load-dir", type=str, help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument(
        "--policies-to-eval", nargs="+", type=str, default=["red_0", "blue_0"], help="List of agent ids to train"
    )
    parser.add_argument("--gif", type=str, help="Store output as GIF at given path.")
    parser.add_argument(
        "--name", type=str, default="<my_experinemnt>", help="Distinct name to track your experinemnt in save-dir"
    )
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="Can be either 'human' or 'rgb_array.'")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="submission/evaluation_reports/",
        help="Directory for saving evaluation results.",
    )

    parsed_args = parser.parse_args()
    main_evaluation(args=parsed_args)
