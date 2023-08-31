import pytest 
from ray.tune.registry import register_env
from multigrid.envs import CONFIGURATIONS
from multigrid.rllib import to_rllib_env
from multigrid.wrappers import (
    CompetativeRedBlueDoorWrapper,
)
from multigrid.base import MultiGridEnv
from multigrid.core.constants import Color, Direction, State, Type
import numpy as np
import gymnasium as gym

def test_code_completion():

    env_id = "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single"

    env: MultiGridEnv = gym.make(
        env_id,
        render_mode="human",
        agents=1,
        screen_size=640,
    )

    try:
        env = CompetativeRedBlueDoorWrapper(env) 
        dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])
        env.observation_space["red_0"]["image"].shape[2] == sum(dim_sizes)
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")

