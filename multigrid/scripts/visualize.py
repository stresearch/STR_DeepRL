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


def evaluation(
    algorithm: Algorithm, num_episodes: int = 100, policies_to_eval: list[str] = ["red_0"]
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
    env = algorithm.env_creator(algorithm.config.env_config)

    for episode in range(num_episodes):
        print("\n", "-" * 32, "\n", "Episode", episode, "\n", "-" * 32)

        episode_rewards = {agent_id: 0.0 for agent_id in env.get_agent_ids()}
        terminations, truncations = {"__all__": False}, {"__all__": False}
        observations, infos = env.reset()
        states = {agent_id: algorithm.get_policy(agent_id).get_initial_state() for agent_id in env.get_agent_ids()}
        while not terminations["__all__"] and not truncations["__all__"]:
            frames.append(env.get_frame())

            actions = {}
            for agent_id in env.get_agent_ids():
                # Single-agent
                actions[agent_id] = algorithm.compute_single_action(
                    observations[agent_id], states[agent_id], policy_id=agent_id
                )

            observations, rewards, terminations, truncations, infos = env.step(actions)
            for agent_id in rewards:
                episode_rewards[agent_id] += rewards[agent_id]

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
    parts = policy_id.split('_v')
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

    # HW3 TODO - Setup Policies
    evaluating_policies = {}
    for policy_id in args.policies_to_eval:
        policy_name = team_policies_mapping[policy_id]
        eval_policy = SubmissionPolicies[policy_name](policy_id=policy_id, policy_name=policy_name)
        evaluating_policies[policy_id] = eval_policy

    config = algorithm_config(
        **vars(args),
        num_workers=0,
        num_gpus=0,
        policies_map = evaluating_policies,
    )
    config.explore = False
    config.environment(disable_env_checking=True)
    # config.rl_module( _enable_rl_module_api=False)
    # config.training(_enable_learner_api=False)
    algorithm = config.build()
    checkpoint = get_checkpoint_dir(args.load_dir)

    # Create a Path object for the directory
    save_path = Path(args.save_dir)

    # Make sure the directory exists; if not, create it
    save_path.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        from ray.rllib.policy.policy import Policy
        print(f"Loading checkpoint from {checkpoint}")
 
        if "default_DTDE_1v1_opponent_checkpoint" in args.eval_config and "DTDE-1v1" in args.env:
            restored_policies = Policy.from_checkpoint(args.eval_config["default_DTDE_1v1_opponent_checkpoint"])
        elif "default_CTCE_2v2_opponent_checkpoint" in args.eval_config and "CTCE-2v2" in args.env:
            restored_policies = Policy.from_checkpoint(args.eval_config["default_CTCE_2v2_opponent_checkpoint"])
        else:
            restored_policies = Policy.from_checkpoint(checkpoint)
        
        sorted_keys = sorted(restored_policies.keys(), key=sort_policy_id, reverse=True)
        opponent_policies=[policy for policy in algorithm.config.policies if policy not in args.policies_to_eval]

        for idx, agent_id in enumerate(opponent_policies):
            best_opponent_policy_id = sorted_keys[:len(opponent_policies)][idx]
            restored_policy_weights = restored_policies[best_opponent_policy_id].get_weights()
            algorithm.set_weights({agent_id: restored_policy_weights})

        for agent_id in args.policies_to_eval :
            restored_policy_weights = restored_policies[agent_id].get_weights()
            algorithm.set_weights({agent_id: restored_policy_weights})


        # algorithm.restore(checkpoint)
       
    frames, episodes_data = evaluation(
        algorithm, num_episodes=args.num_episodes, policies_to_eval=args.policies_to_eval
    )

    scenario_name = args.env.split("-v3-")[1] #str(checkpoint).split("/")[-2].split("_")[1].split("-v3-")[1]
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
        help="Evaluation config dict, given as a JSON string (e.g. '{\"team_policies_mapping\": {\"red_0\" : \"your_policy_name\" , \"blued_0\" : \"your_policy_name\" }}')",
    )
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to visualize.")
    parser.add_argument("--load-dir", type=str, help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument("--policies-to-eval", nargs="+", type=str, default=["red_0", "blue_0"], help="List of agent ids to train")
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
