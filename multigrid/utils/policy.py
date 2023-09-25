"""Template for custom policy implementations."""

import abc
from typing import Generic, Tuple, TypeVar
from gymnasium.core import ObservationWrapper

State = TypeVar("State")


class Policy(Generic[State], metaclass=abc.ABCMeta):
    """Abstract base class for a policy."""

    def __init__(self, policy_id: str, policy_name: str):
        # You can implement any init operations here or in setup()
        self.policy_id = policy_id  # future todo  - Should this be multiple or indiviaul, current is not individual
        self.policy_name = policy_name  # future todo  - Should this be multiple or indiviaul, current is not individual
        self.reward_schemes = {self.policy_id: {}}
        self.algorithm_training_config = {self.policy_id: {}}

    @staticmethod
    @abc.abstractmethod
    def custom_observation_space(policy_id, raw_observation_space, raw_action_space):
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def custom_observations(obs: dict[any], policy_id: str, wrapper: ObservationWrapper):
        raise NotImplementedError()

    @staticmethod
    @abc.abstractmethod
    def custom_handle_steps(agent, agent_index, action, reward, terminated, info, env):
        raise NotImplementedError()

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        del args, kwargs
        self.close()
