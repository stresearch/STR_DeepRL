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
    algorithm: Algorithm,
    num_episodes: int = 100,
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
        solved = all([env.env.env.red_door.is_open, (env.env.env.step_count < env.max_steps)])
        print("\n", "Rewards:", episode_rewards)
        print("\n", "Total Time Steps:", env.env.env.step_count)
        print("\n", "Solved:", solved)

        # Set episode data
        episodes_data.append({**episode_rewards, **{"Episode Length": env.env.env.step_count, "Solved": solved}})

    env.close()

    return frames, episodes_data


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
    config = algorithm_config(
        **vars(args),
        num_workers=0,
        num_gpus=0,
    )
    config.environment(disable_env_checking=True)
    algorithm = config.build()
    checkpoint = get_checkpoint_dir(args.load_dir)

    # Create a Path object for the directory
    save_path = Path(args.save_dir)

    # Make sure the directory exists; if not, create it
    save_path.mkdir(parents=True, exist_ok=True)

    if checkpoint:
        from ray.rllib.policy.policy import Policy

        print(f"Loading checkpoint from {checkpoint}")
        algorithm.restore(checkpoint)

        scenario_name = str(checkpoint).split("/")[-2].split("_")[1].split("-v3-")[1]

        # NOTE - future fixme update checkpoint loading method
        # # New way
        # policy_name = f"policy_{args.our_agent_ids[1]}"
        # restored_policy_0 = Policy.from_checkpoint(checkpoint)
        # restored_policy_0_weights = restored_policy_0[policy_name].get_weights()
        # algorithm.set_weights({policy_name: restored_policy_0_weights})

    frames, episodes_data = evaluation(
        algorithm,
        num_episodes=args.num_episodes,
    )

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
    parser.add_argument("--num-episodes", type=int, default=10, help="Number of episodes to visualize.")
    parser.add_argument("--load-dir", type=str, help="Checkpoint directory for loading pre-trained policies.")
    parser.add_argument("--gif", type=str, help="Store output as GIF at given path.")
    parser.add_argument("--our-agent-ids", nargs="+", type=int, default=[0, 1], help="List of agent ids to evaluate.")
    parser.add_argument("--render-mode", type=str, default="rgb_array", help="Can be either 'human' or 'rgb_array.'")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="submission/evaluation_reports/",
        help="Directory for saving evaluation results.",
    )

    parsed_args = parser.parse_args()
    main_evaluation(args=parsed_args)
