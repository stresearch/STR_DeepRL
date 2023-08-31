"""
************
Environments
************

This package contains implementations of several MultiGrid environments.

**************
Configurations
**************

* `Blocked Unlock Pickup <./multigrid.envs.blockedunlockpickup>`_
    * ``MultiGrid-BlockedUnlockPickup-v0``
* `Empty <./multigrid.envs.empty>`_
    * ``MultiGrid-Empty-5x5-v0``
    * ``MultiGrid-Empty-Random-5x5-v0``
    * ``MultiGrid-Empty-6x6-v0``
    * ``MultiGrid-Empty-Random-6x6-v0``
    * ``MultiGrid-Empty-8x8-v0``
    * ``MultiGrid-Empty-16x16-v0``
* `Locked Hallway <./multigrid.envs.locked_hallway>`_
    * ``MultiGrid-LockedHallway-2Rooms-v0``
    * ``MultiGrid-LockedHallway-4Rooms-v0``
    * ``MultiGrid-LockedHallway-6Rooms-v0``
* `Playground <./multigrid.envs.playground>`_
    * ``MultiGrid-Playground-v0``
* `Red Blue Doors <./multigrid.envs.redbluedoors>`_
    * ``MultiGrid-RedBlueDoors-6x6-v0``
    * ``MultiGrid-RedBlueDoors-8x8-v0``
"""

from .blockedunlockpickup import BlockedUnlockPickupEnv
from .competative_red_blue_door import (
    CompetativeRedBlueDoorEnvV2,
    CompetativeRedBlueDoorEnvV3,
)
from .empty import EmptyEnv
from .locked_hallway import LockedHallwayEnv
from .playground import PlaygroundEnv
from .redbluedoors import RedBlueDoorsEnv


CONFIGURATIONS = {
    "MultiGrid-BlockedUnlockPickup-v0": (BlockedUnlockPickupEnv, {}),
    "MultiGrid-CompetativeRedBlueDoor-v2": (CompetativeRedBlueDoorEnvV2, {"size": 8, "allow_agent_overlap": False}),
    "MultiGrid-CompetativeRedBlueDoor-v3": (
        CompetativeRedBlueDoorEnvV3,
        {"size": 8, "allow_agent_overlap": False, "has_obsticle": True, "teams": {"red": 2, "blue": 2}, "agents": 4},
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": False,
            "teams": {"red": 1},
            "agents": 1,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.2, # HW1 FIXME 1 - This value maybe too high for dense penalty 
                },
            },
        },
    ),
     "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Single-with-Obsticle": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 1},
            "agents": 1,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTCE-Red": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 2},
            "agents": 2,
            "training_scheme": "CTCE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 2},
            "agents": 2,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 2},
            "agents": 2,
            "training_scheme": "CTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTCE-Red-Eval": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 2},
            "agents": 2,
            "training_scheme": "CTCE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
            "randomization": True,
            "max_steps": 300,
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-Red-Eval": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 2},
            "agents": 2,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
            "randomization": True,
            "max_steps": 300,
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTDE-Red-Eval": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "teams": {"red": 2},
            "agents": 2,
            "training_scheme": "CTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
            "randomization": True,
            "max_steps": 300,
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-1v1": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": False,
            "death_match": True,
            "teams": {"red": 1, "blue": 1},
            "agents": 2,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTCE-1v1": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": False,
            "death_match": True,
            "teams": {"red": 1, "blue": 1},
            "agents": 2,
            "training_scheme": "CTCE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTCE-2v2": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "death_match": True,
            "teams": {"red": 2, "blue": 2},
            "agents": 4,
            "training_scheme": "CTCE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-2v2": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "death_match": True,
            "teams": {"red": 2, "blue": 2},
            "agents": 4,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTDE-2v2": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": True,
            "death_match": False,
            "teams": {"red": 2, "blue": 2},
            "agents": 4,
            "training_scheme": "CTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTDE-2v2-Death-Match": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": False,
            "death_match": True,
            "teams": {"red": 2, "blue": 2},
            "agents": 4,
            "training_scheme": "CTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-CTCE-2v2-Death-Match": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": False,
            "death_match": True,
            "teams": {"red": 2, "blue": 2},
            "agents": 4,
            "training_scheme": "CTCE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-CompetativeRedBlueDoor-v3-DTDE-2v2-Death-Match": (
        CompetativeRedBlueDoorEnvV3,
        {
            "size": 8,
            "allow_agent_overlap": False,
            "has_obsticle": False,
            "death_match": True,
            "teams": {"red": 2, "blue": 2},
            "agents": 4,
            "training_scheme": "DTDE",
            "reward_schemes": {
                "red_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "red_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_0": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
                "blue_1": {
                    "eliminated_opponent_sparse_reward": 0.5,
                    "key_pickup_sparse_reward": 0.5,
                    "ball_pickup_dense_reward": 0.5,
                    "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                    "invalid_pickup_dense_penalty": 0.001,
                },
            },
        },
    ),
    "MultiGrid-Empty-5x5-v0": (EmptyEnv, {"size": 5}),
    "MultiGrid-Empty-Random-5x5-v0": (EmptyEnv, {"size": 5, "agent_start_pos": None}),
    "MultiGrid-Empty-6x6-v0": (EmptyEnv, {"size": 6}),
    "MultiGrid-Empty-Random-6x6-v0": (EmptyEnv, {"size": 6, "agent_start_pos": None}),
    "MultiGrid-Empty-8x8-v0": (EmptyEnv, {}),
    "MultiGrid-Empty-16x16-v0": (EmptyEnv, {"size": 16}),
    "MultiGrid-LockedHallway-2Rooms-v0": (LockedHallwayEnv, {"num_rooms": 2}),
    "MultiGrid-LockedHallway-4Rooms-v0": (LockedHallwayEnv, {"num_rooms": 4}),
    "MultiGrid-LockedHallway-6Rooms-v0": (LockedHallwayEnv, {"num_rooms": 6}),
    "MultiGrid-Playground-v0": (PlaygroundEnv, {}),
    "MultiGrid-RedBlueDoors-6x6-v0": (RedBlueDoorsEnv, {"size": 6}),
    "MultiGrid-RedBlueDoors-8x8-v0": (RedBlueDoorsEnv, {"size": 8}),
}

# Register environments with gymnasium
from gymnasium.envs.registration import register

for name, (env_cls, config) in CONFIGURATIONS.items():
    register(id=name, entry_point=env_cls, kwargs=config)
