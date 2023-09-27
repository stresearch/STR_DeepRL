import pytest

import argparse
import subprocess
import os
from pathlib import Path
from itertools import combinations
from multigrid.scripts.visualize import main_evaluation
from multigrid.utils.training_utilis import get_checkpoint_dir
from multigrid.envs import CONFIGURATIONS
from multigrid.agents_pool import SubmissionPolicies
from ray.rllib.policy.policy import Policy
import json


# import cProfile
# Set the working diretory to the repo root
REPO_ROOT = subprocess.check_output(["git", "rev-parse", "--show-toplevel"]).strip().decode("utf-8")
os.chdir(REPO_ROOT)

SUBMISSION_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/submission_config.json"), key=os.path.getmtime
)[-1]

with open(SUBMISSION_CONFIG_FILE, "r") as file:
    submission_config_data = file.read()

submission_config = json.loads(submission_config_data)

SUBMITTER_NAME = submission_config["name"]
SAVE_DIR = "submission/evaluation_reports/from_github_actions"


EVALUATION_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/configs/evaluation_config.json"), key=os.path.getmtime
)[-1]

with open(EVALUATION_CONFIG_FILE, "r") as file:
    evaluation_config_data = file.read()

evaluation_config = json.loads(evaluation_config_data)


def commit_and_push():
    # Setting git config
    # subprocess.run(['git', 'config', '--global', 'user.email', 'you@example.com'])
    # subprocess.run(['git', 'config', '--global', 'user.name', 'Your Name'])

    # Add all changed files
    subprocess.run(["git", "add", "submission/evaluation_reports/from_github_actions/*"])

    # Commit changes
    try:
        subprocess.run(["git", "commit", "-m", "Auto-commit evaluation reports"], check=True)
    except subprocess.CalledProcessError:
        print("Nothing to commit. Skipping git commit.")
        return

    # Push changes
    subprocess.run(["git", "push"])


def test_evaluation():
    # Create/check paths
    search_dir = "submission/ray_results"
    assert os.path.exists(search_dir), f"Directory {search_dir} does not exist!"

    checkpoint_dirs = [
        checkpoint_dir.parent
        for checkpoint_dir in sorted(Path(search_dir).expanduser().glob("**/result.json"), key=os.path.getmtime)
    ]

    checkpoint_paths = [
        sorted(Path(checkpoint_dir).expanduser().glob("**/*.is_checkpoint"), key=os.path.getmtime)[-1].parent
        for checkpoint_dir in checkpoint_dirs
    ]

    for checkpoint_path in checkpoint_paths:
        # Define parameters for the test
        env = str(checkpoint_path).split("/")[-2].split("_")[1] + "-Eval"
        scenario_name = env.split("-v3-")[1]
        gif = f"{scenario_name}_{SUBMITTER_NAME}"

        policies_to_eval = ["red_0"]
        if "2v2" in scenario_name:
            policies_to_eval = ["red"]
            evaluation_config["team_policies_mapping"] = {
                "red": "your_policy_name",
            }

        elif "CTDE-Red" in scenario_name:
            policies_to_eval = ["red_0", "red_1"]
            evaluation_config["team_policies_mapping"] = {
                "red_0": "your_policy_name",
                "red_1": "your_policy_name_v2",
            }

        # Set argument
        params = {
            "algo": "PPO",
            "framework": "torch",
            "lstm": False,
            "env": env,
            "env_config": {},
            "num_episodes": 10,
            "load_dir": checkpoint_path,
            "gif": gif,
            "render_mode": "rgb_array",
            "save_dir": SAVE_DIR,
            "policies_to_eval": policies_to_eval,
            "eval_config": evaluation_config,
        }

        args = argparse.Namespace(**params)

        # Call the evaluation function
        main_evaluation(args)

        # Check the generated evaluation reports
        eval_report_path = os.path.join(args.save_dir, f"{scenario_name}_eval_summary.csv")
        assert os.path.exists(eval_report_path), f"Expected evaluation report {eval_report_path} doesn't exist!"

    commit_and_push()


# cProfile.run('test_evaluation()', 'test_evaluation_output.prof')


def test_batch_evaluation():
    # Create/check paths
    search_dir = "submission/ray_results/1v1"
    assert os.path.exists(search_dir), f"Directory {search_dir} does not exist!"

    checkpoint_dirs = [
        checkpoint_dir.parent
        for checkpoint_dir in sorted(Path(search_dir).expanduser().glob("**/result.json"), key=os.path.getmtime)
    ]

    checkpoint_paths = [
        sorted(Path(checkpoint_dir).expanduser().glob("**/*.is_checkpoint"), key=os.path.getmtime)[-1].parent
        for checkpoint_dir in checkpoint_dirs
    ]

    agent_policy_ids = [policy_id for policy_id in SubmissionPolicies]

    # Generate all possible 1v1 matchups
    matchups = list(combinations(agent_policy_ids, 2))

    agent_policies_checkpoints = {}
    for checkpoint_path in checkpoint_paths:
        restored_policies = Policy.from_checkpoint(checkpoint_path)
        custom_policy_id = restored_policies["red_0"].config["env_config"]["team_policies_mapping"]["red_0"]
        agent_policies_checkpoints[custom_policy_id] = checkpoint_path

    # Matchups contains all the 1v1 competitions
    for match in matchups:
        policy1, policy2 = match

        checkpoint_path_1 = agent_policies_checkpoints[policy1]
        checkpoint_path_2 = agent_policies_checkpoints[policy2]

        envs = ["MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1", "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1-Death_Match"]

        for env in envs:

            if evaluation_config["using_eval_scenarios"]:
                env += "-Eval"

            scenario_name = env.split("-v3-")[1]

            # Fair evaluation for learned asymmetry behavior
            for _ in range(2):
                gif = f"{scenario_name}_{policy1}_as_Red_VS_{policy2}_as_Blue"

                evaluation_config["team_policies_mapping"] = {}
                evaluation_config["team_policies_mapping"]["red_0"] = policy1
                evaluation_config["team_policies_mapping"]["blue_0"] = policy2
                evaluation_config["default_DTDE_1v1_opponent_checkpoint"] = checkpoint_path_2

                policies_to_eval = ["red_0", "blue_0"]

                # Set argument
                params = {
                    "algo": "PPO",
                    "framework": "torch",
                    "lstm": False,
                    "env": env,
                    "env_config": {},
                    "num_episodes": 10,
                    "load_dir": checkpoint_path_1,
                    "gif": gif,
                    "render_mode": "rgb_array",
                    "save_dir": SAVE_DIR,
                    "policies_to_eval": policies_to_eval,
                    "eval_config": evaluation_config,
                }

                args = argparse.Namespace(**params)

                # Call the evaluation function
                main_evaluation(args)

                # Check the generated evaluation reports
                eval_report_path = os.path.join(args.save_dir, f"{gif}_eval_summary.csv")

                # Change Red-Blue asymmetry
                policy1, policy2 = policy2, policy1
                checkpoint_path_1, checkpoint_path_2 = checkpoint_path_2, checkpoint_path_1

        # commit_and_push()

