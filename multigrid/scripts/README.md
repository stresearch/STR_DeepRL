# Training MultiGrid agents with RLlib

MultiGrid is compatible with RLlib's multi-agent API.

This folder provides scripts to train and visualize agents over MultiGrid environments.

## Requirements

Using MultiGrid environments with RLlib requires installation of [rllib](https://docs.ray.io/en/latest/rllib/index.html), and one of [PyTorch](https://pytorch.org/) or [TensorFlow](https://www.tensorflow.org/).

## Getting Started

Train 2 agents on the `MultiGrid-Empty-8x8-v0` environment using the PPO algorithm:

    python train.py --algo PPO --env MultiGrid-Empty-8x8-v0 --num-agents 2 --save-dir ~/saved/empty8x8/

Visualize behavior from trained agents policies:

    python visualize.py --algo PPO --env MultiGrid-Empty-8x8-v0 --num-agents 2 --load-dir ~/saved/empty8x8/

For more options, run ``python train.py --help`` and ``python visualize.py --help``.

## Environments

All of the environment configurations registered in [`multigrid.envs`](../envs/__init__.py) can also be used with RLlib, and are registered via `import multigrid.rllib`.

To use a specific MultiGrid environment configuration by name:

    >>> import multigrid.rllib
    >>> from ray.rllib.algorithms.ppo import PPOConfig
    >>> algorithm_config = PPOConfig().environment(env='MultiGrid-Empty-8x8-v0')

To convert a custom `MultiGridEnv` to an RLlib `MultiAgentEnv`:

    >>> from multigrid.rllib import to_rllib_env
    >>> MyRLLibEnvClass = to_rllib_env(MyEnvClass)
    >>> algorithm_config = PPOConfig().environment(env=MyRLLibEnvClass)



# Assignment 1: Intro to Deep RL with Single Agent Training Environments

### Due September 14, 11:30 pm


The goal of this assignment is to gain experimentally learning experiences on the key components of RL environments.

Knowing how to debug your environment 
- Ensure the env has the right reward scale
- Ensure the env’s done conditions are meeting the learning objective
- Ensure your agent is having the right observation and action spaces for training 

Kick start training on your laptop or on Google CoLab
Knowing how to use Tensorboard with custom metrics 
Get familiar with the assignment submission process


The starter-code for this assignment can be found at
https://classroom.github.com/classrooms/123430433-rl2rl-deeprl/assignments/week-1-intro-to-deep-rl-and-agent-training-environments



## Setup

You have the option of running the code either on Google Colab or on your own machine.

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [INSTALLATION.md](INSTALLATION.md) for instructions.

2. **Colab:** The first few sections of the notebook will install all required dependencies. You can try out the Colab option by clicking the badge below:

# TODO Add Colab Link
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/berkeleydeeprlcourse/homework_fall2022/blob/master/hw1/cs285/scripts/run_hw1.ipynb)



## Complete the code
We recommend that you read the files in the following order. For some files, you will need to fill in blanks, labeled TODO.

Fill in sections marked with `TODO` or `FIXME` . In particular, see
 - [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py)
 - [wrappers.py](multigrid/wrappers.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/manual_control.py](multigrid/scripts/manual_control.py)
 - [scripts/visualize.py](multigrid/scripts/visualize.py)
 - [scripts/train.py](multigrid/scripts/train.py)
 
 For this assignemnt please use: 
 - [scripts/train.py](multigrid/scripts/train.py) and [multigridscripts/visualize.py](multigrid/scripts/visualize.py) if running locally
 - or [notebooks/homework1.ipynb](notebooks/homework1.ipynb)if running on Colab



## Run the code

Tip: While debugging, you probably want to use VSCode's debugger. And here is a [sample Launch commands](sample_launch.json) once you have install the Python extention.


If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

### Task 1 - Get to Know your Learning Environment
Command for Task 1:

```shell
python scripts/manual_control.py
```




Notes:
1. Please feel free to also try another optional arguments to get familiar with your envinorment.
2. Dense reward will be shown in every time step after your input actions
    1. The keyboard controls can be found in key_handler() in the ManualControl class
    2. Wrong action can yield penalty per step
3. Sparse reward will be add to total episodic reward when you perform an associated operation i.e, picking up the key from the floor

Task 1 description:
Your agent can perform a bad action by randomly using the pickup action in every time step. There is a penalty when the agent does't pick up the right object. You should reset this value in proportion to the total horizon and the ultimate goal oriented reward. Please fix the reward scale in [envs/competative_red_blue_door.py](multigrid/envs/competative_red_blue_door.py)


### Task 2 - Debug Observations and Observations Space for Training

Command for task 2:

If you run the following training command, you are expected to see ValueError on mismatching observation and observation space. Make sure to handle this exception and implement the correct observation to avoid it.

```shell
python scripts/train.py
```

Note: Please feel free to also try another optional arguments to get familiar with the training and evaluation proccess. Your training batch size should be greater than horizon, such that you’re collecting multiple rollouts when evaluating the performance of your trained policy. For example, if the horizon is 1000 and training batch size is 5000, then you’ll be
collecting approximately 5 trajectories (maybe more if any of them terminate early)


## Task 3 - Monitoring and Tracking Agent Training with Tensorboard and Save out Visualization from Evaluation:


You can monitor and track your runs using tensorboard:
```
tensorboard --logdir ./ray_result
```

You will see scalar summaries as well as videos of your trained policies (in the 'images' tab).



You can also choose to visualize specific checkpoint with the following command:
```shell
python scripts/visualize.py --env MultiGrid-CompetativeRedBlueDoor-v0  --num-episodes 20  --load-dir ./ray_results/PPO/PPO_MultiGrid-CompetativeRedBlueDoor-v0_37eb5_00000_0_2023-07-10_11-12-43 --gif ./result.gif
```

If running on Colab, you will be using the `%tensorboard` [line magic](https://ipython.readthedocs.io/en/stable/interactive/magics.html) to do the same thing; see the [notebook](notebooks/homework1.ipynb) for more details.


## Task 4 - Submit your homework on Github Classroom





 Turning it in
1. Submitting the PDF. Make a PDF report containing: Table 1 for a table of results from Question
1.2, Figure 1 for Question 1.3. and Figure 2 with results from question 2.2.
You do not need to write anything else in the report, just include the figures with captions as described
in each question above. See the handout at
http://rail.eecs.berkeley.edu/deeprlcourse/static/misc/viz.pdf
for notes on how to generate plots.
2. Submitting the code and experiment runs. In order to turn in your code and experiment logs,
create a folder that contains the following:
• A folder named run logs with experiments for both the behavioral cloning (part 2, not part 3)
exercise and the DAgger exercise. Note that you can include multiple runs per exercise if you’d like,
but you must include at least one run (of any task/environment) per exercise. These folders can
be copied directly from the cs285/data folder into this new folder. Important: Disable video
logging for the runs that you submit, otherwise the files size will be too large! You
can do this by setting the flag --video log freq -1
• The cs285 folder with all the .py files, with the same names and directory structure as the original
homework repository. Also include the commands (with clear hyperparameters) that we need in
order to run the code and produce the numbers that are in your figures/tables (e.g. run “python
run hw1 behavior cloning.py –ep len 200” to generate the numbers for Section 2 Question 2) in the
form of a README file



3. If you are a Mac user, do not use the default “Compress” option to create the zip. It creates artifacts that the autograder does not like. You may use zip -vr submit.zip submit -x "*.DS Store"
from your terminal.
4. Turn in your assignment on Gradescope. Upload the zip file with your code and log files to HW1 Code,
and upload the PDF of your report to HW1.