

---
# Assignment 1: Intro to Deep RL with Single Agent Training Environments

## Due Date
- **Due Date:** Thursday, September 14, 6:00 PM

## Overview
This assignment aims to provide hands-on experience with key components of Reinforcement Learning (RL) environments. Upon completion, you'll be able to:

- Debug your environment by ensuring the following:
  - The environment has the correct reward scale.
  - The environment's termination conditions meet the learning objective.
  - Your agent utilizes the correct observation and action spaces for training.
- Begin training on your local machine or Google Colab.
- Familiarize yourself with Tensorboard and how to use custom metrics.
- Understand the assignment submission process.

The starter code for this assignment can be found [here](https://classroom.github.com/classrooms/123430433-rl2rl-deeprl/assignments/week-1-intro-to-deep-rl-and-agent-training-environments).

## Setup Instructions
Choose to run the code on either Google Colab or your local machine:
- **Local Setup**: For local execution, install the necessary Python packages by following the [INSTALLATION.md](INSTALLATION.md) guidelines.
- **Google Colab**: To run on Colab, the notebook will handle dependency installations. Try it by clicking below:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/heng2j/multigrid/blob/hw1_new/notebooks/homework1.ipynb)

## Recommended Steps to Get Familiar with the Codebase
We recommend reading the files in the following order. For some files, you will need to fill in the sections labeled `HW1 TODO` or `HW1 FIXME`.

- [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py)
- [envs/__init__.py](multigrid/envs/__init__.py)
- [wrappers.py](multigrid/wrappers.py)

Look for sections marked with `HW1` to understand how your changes will be utilized. You might also find the following files relevant:

- [scripts/manual_control.py](multigrid/scripts/manual_control.py)
- [scripts/visualize.py](multigrid/scripts/visualize.py)
- [scripts/train.py](multigrid/scripts/train.py)

Depending on your chosen setup, refer to [scripts/train.py](multigrid/scripts/train.py) and [scripts/visualize.py](multigrid/scripts/visualize.py) (if running locally), or [notebooks/homework1.ipynb](notebooks/homework1.ipynb) (if running on Colab).


If you're debugging, you might want to use VSCode's debugger. If you're running on Colab, adjust the `#@params` in the `Args` class as per the command-line arguments above.


---
# Assignment Task Breakdown

## Task 0 - Own Your Assignment By Configuring Submission Settings
Please change the `name` in [submission_config.json](submission/submission_config.json) to your name in CamelCase

---
## Task 1 - Familiarize Yourself with the Environment
Running the follwoing command to manually control your agent. The keyboard controls can be found in `key_handler()` in the `ManualControl` class

Command for Task 1:
```shell
python multigrid/scripts/manual_control.py --env-id MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single
```

**Tips:**
- You can view `CONFIGURATIONS` in [envs/__init__.py](multigrid/envs/__init__.py) to manually control different environments. Note that only single agent versions are playable; multi-agent versions are for viewing only.
- For more information on the training environment, visit the original [multigrid](https://github.com/ini/multigrid) repository that we forked, and the actively developed official [Minigrid](https://github.com/Farama-Foundation/Minigrid) managed by the Farama Foundation.


*If you have suggestions for manually controlling multi-agent environments, please let us know.*

***Notes:***

1. Experiment with other optional arguments to familiarize yourself with the environment.
2. The dense reward will be displayed at each time step after your input actions. The keyboard controls can be found in the `key_handler()` method in the `ManualControl` class.
3. A wrong action can lead to a penalty per step. A sparse reward will be added to the total episodic reward when you perform an associated operation, e.g., picking up the key from the floor.

**Task 1 Description:** Your agent can perform a wrong action by randomly using the pickup action at each time step. There is a penalty when the agent picks up the incorrect object. You should adjust this value in proportion to the game horizon (1280 time steps). Please fix the reward scale in the reward scheme that defined for `MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single` in [envs/__init__.py](multigrid/envs/__init__.py)  [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py).

There are many ways to debug the rewards. You can manually control the agent, collect performance and behavior analysis data by evaluating the trained agent, or observe the Training Status Reports or Tensorboard during training. Based on your observations in this exercise, in your own words, please briefly describe the impact of having the right scale of dense rewards in respect to the total horizon of the game. Please include your answer in the Notebooks or as a sepearte `HW1_answers.md` in the `submission/` folder.

---
## Task 2 - Debug Observations and Observations Space for Training
If you run the following training command to train an agent with Decentalized Training Decentalized Execution (DTDE) training scheme, you are expected to see ValueErrors from blanks that needed to be filled to fix the mismatching observation and observation space issue. Make sure to handle this exception and implement the correct observation to avoid it.


Command for Task 2:
```shell
python multigrid/scripts/train.py --local-mode False --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single --num-workers 10 --num-gpus 0 --name --training-scheme DTDE
```

**Tips:**
- You can set `--local-mode` to True and use the VSCode debugger to walk through the code for debugging.
- Check the original definition of `self.observation_space` in [agent.py](multigrid/core/agent.py) and the new requirements in `CompetativeRedBlueDoorWrapper` in [wrappers.py](multigrid/wrappers.py) to see how the observation for the agents should be defined in `MultiGrid-CompetativeRedBlueDoor-v3`. Then you will know how to match them with the observations you are generating.

For training. Your training batch size should be larger than the horizon so that you're collecting multiple rollouts when evaluating the performance of your trained policy. For example, if the horizon is 1000 and the training batch size is 5000, you'll collect approximately 5 trajectories (or more if any of them terminate early).

***Note:*** 

You might encounter a `ValueError` for mismatching observation and observation space if you run the above command. Make sure to handle this exception and implement the correct observation to avoid it.


---

## Task 3 - Monitor and Track Agent Training with Tensorboard and Save Out Visualization from Evaluation
Monitor and track your runs using Tensorboard with the following command:
```shell
tensorboard --logdir submission/ray_results/
```

**Tips:**
- You can filter the plots using the following filters:

```
episode_len_mean|ray/tune/episode_reward_mean|episode_reward_min|entropy|vf|loss|kl|cpu|ram
```


- To visualize a specific checkpoint, use the following command:
```shell
python multigrid/scripts/visualize.py --env MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single  --num-episodes 10  --load-dir submission/ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single_XXXX/checkpoint_YYY/checkpoint-YYY --render-mode human --gif DTDE-Red-Single
```
##### Replace `XXXX` and `YYY` with the corresponding number of your checkpoint.


- If running on Colab, use the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to achieve the same; see the [notebook](notebooks/homework1.ipynb) for more details.

---


## Task 4 - Homework Submission via Github Classroom

### Submission Requirements:

1. **RLlib Agents**: 
    - Commit and push your best-performing RLlib agents and checkpoints, ensuring they satisfy the minimum thresholds described in the Task, to [submission/ray_results](submission/ray_results).

2. **RLlib Agents Evaluation Reports**: 
    - Commit and push relevant RLlib agent evaluation results: `<my_experiment>_eval_summary.csv`, `<my_experiment>_episodes_data.csv`, and `<my_experiment>.gif` to [submission/evaluation_reports](submission/evaluation_reports).

3. **Answers to Questions**:
    - For question answers, either:
      - Update the provided [homework1.ipynb](notebooks/homework1.ipynb) notebook, or 
      - Submit a separate `HW1_Answer.md` file under [submission](submission).

4. **MLFlow Artifacts**:
    - Ensure you commit and push the MLFlow artifacts to [submission](submission) (Which should be automatic).


#### Tips:
- Retain only the top-performing checkpoints in [submission/ray_results](submission/ray_results).
    - Refer to the baseline performance thresholds specified for each agent training task.
    - Uploading numerous checkpoints, particularly underperforming ones, may cause the CI/CD to fail silently due to time constraints.
    
- Executing [tests/test_evaluation.py](tests/test_evaluation.py) with `pytest` should generate and push the necessary results to [submission/evaluation_reports](submission/evaluation_reports).

- For an exemplar submission that fulfills all the requirements and successfully passing the Autograding Github Actions, please checkout [Example Submission](https://github.com/STRDeepRL/week-1-intro-to-deep-rl-and-agent-training-environments-heng4str).

- Always place your submissions within the `submission/` directory. If opting for the notebook approach, please maintain your edited `homework1.ipynb` and related documents under `notebooks/`.

- **Honesty System**: If OS compatibility issues hinder task completion, you're permitted to modify files outside the `EXCEPTION_FILES` listed in [tests/test_codebase.py](tests/test_codebase.py). Add those modified files to the list in your own `test_codebase.py`. However, ensure these changes don't impact your Agent Training Performance, as the centralized evaluation in Week 4's Agent Competition won't consider these changes.

- If you would like to showcase your work at the begining of the class, please notify the class facilitators in advance.


***Note:*** 
Please beaware that the [File Size Check GitHub Action Workflow](.github/workflows/check_file_size.yml) will check the total files size for folers "submission/" "notebooks/", to ensure each of them will not exceed 5MBs. Please ensure to only submit the checkpoints, the notebooks and the MLFlow artifacts that are meant for grading by the Github Action CI/CD pipeline.

