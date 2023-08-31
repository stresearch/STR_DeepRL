import pytest

import argparse
import subprocess
import os
from pathlib import Path
from multigrid.scripts.visualize import main_evaluation
from multigrid.utils.training_utilis import get_checkpoint_dir
from multigrid.envs import CONFIGURATIONS
import json


# import cProfile
# Set the working diretory to the repo root
REPO_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
os.chdir(REPO_ROOT)

SUBMISSION_CONFIG_FILE = sorted(
    Path("submission").expanduser().glob("**/submission_config.json"), key=os.path.getmtime
)[-1]

with open(SUBMISSION_CONFIG_FILE, "r") as file:
    submission_config_data = file.read()

submission_config = json.loads(submission_config_data)

SUBMITTER_NAME = submission_config["name"]
SAVE_DIR = "submission/evaluation_reports/from_github_actions"


def commit_and_push():
    # Setting git config
    # subprocess.run(['git', 'config', '--global', 'user.email', 'you@example.com'])
    # subprocess.run(['git', 'config', '--global', 'user.name', 'Your Name'])

    # Add all changed files
    subprocess.run(['git', 'add', 'submission/evaluation_reports/from_github_actions/*'])

    # Commit changes
    try:
        subprocess.run(['git', 'commit', '-m', 'Auto-commit evaluation reports'], check=True)
    except subprocess.CalledProcessError:
        print("Nothing to commit. Skipping git commit.")
        return

    # Push changes
    subprocess.run(['git', 'push'])  


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
        env = str(checkpoint_path).split("/")[-2].split("_")[1] 
        scenario_name = env.split("-v3-")[1]
        gif = f"{scenario_name}_{SUBMITTER_NAME}"

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
            "our_agent_ids": [0, 1],
            "render_mode": "rgb_array",
            "save_dir": SAVE_DIR,
        }

        args = argparse.Namespace(**params)

        # Call the evaluation function
        main_evaluation(args)

        # Check the generated evaluation reports
        eval_report_path = os.path.join(args.save_dir, f"{scenario_name}_eval_summary.csv")
        assert os.path.exists(eval_report_path), f"Expected evaluation report {eval_report_path} doesn't exist!"

    commit_and_push()

# cProfile.run('test_evaluation()', 'test_evaluation_output.prof')