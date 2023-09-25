from __future__ import annotations

import copy
import numba as nb
import numpy as np
from numpy.typing import NDArray as ndarray
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ObservationWrapper

from .base import MultiGridEnv, AgentID, ObsType
from .core.constants import Color, Direction, State, Type
from .core.world_object import WorldObj


class FullyObsWrapper(ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding instead of agent view.

    Examples
    --------
    >>> import gymnasium as gym
    >>> import multigrid.envs
    >>> env = gym.make('MultiGrid-Empty-16x16-v0')
    >>> obs, _ = env.reset()
    >>> obs[0]['image'].shape
    (7, 7, 3)

    >>> from multigrid.wrappers import FullyObsWrapper
    >>> env = FullyObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'].shape
    (16, 16, 3)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)

        # Update agent observation spaces
        for agent in self.env.agents:
            agent.observation_space["image"] = spaces.Box(
                low=0, high=255, shape=(env.height, env.width, WorldObj.dim), dtype=int
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        img = self.env.grid.encode()
        for agent in self.env.agents:
            img[agent.state.pos] = agent.encode()

        for agent_id in obs:
            obs[agent_id]["image"] = img

        return obs


class OneHotObsWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import OneHotObsWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = OneHotObsWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            view_height, view_width, _ = agent.observation_space["image"].shape
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        for agent_id in obs:
            obs[agent_id]["image"] = self.one_hot(obs[agent_id]["image"], self.dim_sizes)

        return obs

    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class SingleAgentWrapper(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.observation_space = env.agents[0].observation_space
        self.action_space = env.agents[0].action_space

    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return tuple(item for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({self.agents[0].name: action})
        return tuple(item for item in result)


class CompetativeRedBlueDoorWrapper(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import CompetativeRedBlueDoorWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = CompetativeRedBlueDoorWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    # NOTE - Questions
    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.script_path = __file__

        # HW1 TODO 1:
        # Instead of directly using the RGB 3 channels  partially observable agent view
        # In this wrapper, we are applying one-hot encoding of a partially observable agent view
        # using Type, Color, State and Direction
        self.dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent observation spaces
        dim = sum(self.dim_sizes)
        for agent in self.env.agents:
            # HW3 NOTE:
            # Make a deep copy of agent's raw observation_space & action_spcae
            agent.raw_observation_space = copy.deepcopy(agent.observation_space)
            agent.raw_action_space = copy.deepcopy(agent.action_space)

            # Retrieve the shape of the original "image" observation_space
            view_height, view_width, _ = agent.observation_space["image"].shape
            # Reassign the "image" observation_space for one-hot encoding observations
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        # HW1 TODO 2:
        # For each agent_id in obs, update obs[agent_id]['image'] using the self.one_hot() method and 'image' from obs[agent_id].
        # If there's a type mismatch or one of the sub-observations is out of bounds, you might encounter an error like this:
        # ValueError: The observation collected from env.reset was not contained within your env's observation space.
        #             Its possible that there was a typemismatch (for example observations of np.float32 and a space ofnp.float64 observations),
        #             or that one of the sub-observations wasout of bounds.
        # Make sure to handle this exception and implement the correct observation to avoid it.

        for agent_id in obs:
            agent_observations = obs[agent_id]
            if isinstance(agent_observations, list):
                # If it is stacked observations from multiple agents
                for observation in agent_observations:
                    # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                    observation["image"] = self.one_hot(observation["image"], self.dim_sizes)
            else:
                # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                agent_observations["image"] = self.one_hot(agent_observations["image"], self.dim_sizes)

        return obs

    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class SingleAgentWrapperV2(gym.Wrapper):
    """
    Wrapper to convert a multi-agent environment into a
    single-agent environment.
    """

    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        # We only take the shape of the observation_space["image"] for this wrapper
        self.observation_space = env.agents[0].observation_space["image"]
        self.action_space = env.agents[0].action_space

    def reset(self, *args, **kwargs):
        """
        :meta private:
        """
        result = super().reset(*args, **kwargs)
        return tuple(item for item in result)

    def step(self, action):
        """
        :meta private:
        """
        result = super().step({self.agents[0].name: action})
        return tuple(item for item in result)


class CompetativeRedBlueDoorWrapperV2(ObservationWrapper):
    """
    Wrapper to get a one-hot encoding of a partially observable
    agent view as observation.

    Examples
    --------
    >>> from multigrid.envs import EmptyEnv
    >>> from multigrid.wrappers import CompetativeRedBlueDoorWrapper
    >>> env = EmptyEnv()
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0],
            [2, 5, 0]])
    >>> env = CompetativeRedBlueDoorWrapper(env)
    >>> obs, _ = env.reset()
    >>> obs[0]['image'][0, :, :]
    array([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0]],
            dtype=uint8)
    """

    # NOTE - Questions
    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.script_path = __file__

        # HW2 NOTE 1:
        # This basic implementation of PPO in CleanRL is not using spaces.Dict as the observation space
        # Instead, one of the many alternatives is use the one-hot encoding of a partially observable agent view
        # And add one more dimention to the shape of the observations to include the direction in the Box observation space
        # * If you think of a better way to enrich the observation to improve the performance of the agent, please feel free to make the changes
        self.dim_sizes = np.array([len(Type), len(Color), max(len(State), len(Direction))])

        # Update agent's observation spaces
        dim = sum(self.dim_sizes) + 1  # +1 for adding the current direction
        for agent in self.env.agents:
            # Retrieve the shape of the original "image" observation_space
            view_height, view_width, _ = agent.observation_space["image"].shape
            # Reassign the "image" observation_space for one-hot encoding observations
            agent.observation_space["image"] = spaces.Box(
                low=0, high=1, shape=(view_height, view_width, dim), dtype=np.uint8
            )

        # HW2 NOTE 2:
        # This basic implementation of PPO in CleanRL only works with single agent
        # That's why we are taking the single observation_space from the first agent
        self.observation_space = self.env.agents[0].observation_space["image"]

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        """
        :meta private:
        """
        agent_id = list(obs.keys())[0]
        for agent_id in obs:
            agent_observations = obs[agent_id]
            if isinstance(agent_observations, list):
                # If it is stacked observations from multiple agents
                for observation in agent_observations:
                    # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                    observation["image"] = self.one_hot(observation["image"], self.dim_sizes)
            else:
                # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                agent_observations["image"] = self.one_hot(agent_observations["image"], self.dim_sizes)

        # HW2 NOTE 3:
        # The obs we are receiveing from the unwrapped environment will not match the observation space that we defined for this wrapper
        # The original obs contain all raw observation features like "image" and "direction"
        # Here we simply using concatenation to combine both "image" and "direction"
        # which create an observation that match the observation space we defined for this wrapper
        obs[agent_id]["image"] = self.one_hot(obs[agent_id]["image"], self.dim_sizes)
        obs[agent_id]["direction"] = np.full(
            (obs[agent_id]["image"].shape[:2] + (1,)), obs[agent_id]["direction"]
        ).astype("uint8")
        obs = np.concatenate(
            (obs[agent_id]["direction"], obs[agent_id]["image"]),
            axis=2,
        )

        return obs

    @staticmethod
    @nb.njit(cache=True)
    def one_hot(x: ndarray[np.int], dim_sizes: ndarray[np.int]) -> ndarray[np.uint8]:
        """
        Return a one-hot encoding of a 3D integer array,
        where each 2D slice is encoded separately.

        Parameters
        ----------
        x : ndarray[int] of shape (view_height, view_width, dim)
            3D array of integers to be one-hot encoded
        dim_sizes : ndarray[int] of shape (dim,)
            Number of possible values for each dimension

        Returns
        -------
        out : ndarray[uint8] of shape (view_height, view_width, sum(dim_sizes))
            One-hot encoding

        :meta private:
        """
        out = np.zeros((x.shape[0], x.shape[1], sum(dim_sizes)), dtype=np.uint8)

        dim_offset = 0
        for d in range(len(dim_sizes)):
            for i in range(x.shape[0]):
                for j in range(x.shape[1]):
                    k = dim_offset + x[i, j, d]
                    out[i, j, k] = 1

            dim_offset += dim_sizes[d]

        return out


class MARLCompetativeRedBlueDoorWrapper(CompetativeRedBlueDoorWrapper):
    def __init__(self, env: MultiGridEnv):
        """ """
        super().__init__(env)
        self.script_path = __file__

        # HW3 NOTE - Use your favor debugger to take a look what you can do with self.policies_map
        self.policies_map
        # HW3 NOTE - You probably can change any of the following with self.policies_map
        self.reward_schemes
        self.agents
        self.observation_space
        self.action_space
        self.training_scheme

        # HW3 NOTE - You can can customize your observation_space, action_space and probably more here
        for agent in self.env.agents:
            if agent.name in self.policies_map:
                new_observation_space, new_action_space = self.policies_map[agent.name].custom_observation_space(
                    policy_id=agent.name,
                    raw_observation_space=agent.raw_observation_space,
                    raw_action_space=agent.raw_action_space,
                )
                if new_observation_space:
                    self.observation_space[agent.name] = new_observation_space
                if new_action_space:
                    self.action_space[agent.name] = new_action_space

    def observation(self, obs: dict[AgentID, ObsType]) -> dict[AgentID, ObsType]:
        for agent_id in obs:
            if agent_id in self.policies_map:
                obs[agent_id] = self.policies_map[agent_id].custom_observations(
                    obs=obs, policy_id=agent_id, wrapper=self
                )
            else:
                # HW3 NOTE - we still need to keep default observation conversion logic here
                # Future todo - Set up a default observation function
                agent_observations = obs[agent_id]
                if isinstance(agent_observations, list):
                    # If it is stacked observations from multiple agents
                    for observation in agent_observations:
                        # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                        observation["image"] = self.one_hot(observation["image"], self.dim_sizes)
                else:
                    # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                    agent_observations["image"] = self.one_hot(agent_observations["image"], self.dim_sizes)

        return obs
