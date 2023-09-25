import numpy as np
from multigrid.utils.policy import Policy
from multigrid.base import AgentID, ObsType
from multigrid.core.agent import Agent, Mission
from multigrid.core import Action
from ray.rllib.utils.from_config import NotProvided
from ray.rllib.algorithms.ppo import PPOConfig
from gymnasium.core import ObservationWrapper


class YourPolicyNameV2_Policy(Policy):
    """
    Policy class for Meltingpot competition
    About Populations:
        We will make multiple instances of this class for every focal agent
        If you want to sample different agents for every population/episode
        add the required required randomization in the "initial_state" function
    """

    def __init__(self, policy_id: str, policy_name: str):
        # You can implement any init operations here or in setup()
        self.policy_id = policy_id  # Future todo - Should this be multiple or indiviaul, current is not individual
        self.policy_name = policy_name  # Future todo - Should this be multiple or indiviaul, current is not individual
        self.reward_schemes = {
            self.policy_id: {
                "eliminated_opponent_sparse_reward": 0.0,  # HW3 NOTE - No incentive to kill the other agents (Not applicatble for Collaborative Task)
                "key_pickup_sparse_reward": 0.5,
                "ball_pickup_dense_reward": 0.5,
                "dense_reward_discount_factor": {"ball_carrying_discount_factor": 0.9},
                "invalid_pickup_dense_penalty": 0.0015,
            }
        }

        self.algorithm_training_config = {
            self.policy_id: {
                "algo": "PPO",
                "algo_config_class": PPOConfig,
                "algo_config": {
                    "lr": 5e-5,
                    "gamma": 0.99,
                    "lambda_": 0.99,
                    "kl_coeff": 0.2,
                    "kl_target": 0.01,
                    "clip_param": 0.3,
                    "grad_clip": None,
                    "vf_clip_param": 10.0,
                    "vf_loss_coeff": 0.5,
                    "entropy_coeff": 0.001,
                    "sgd_minibatch_size": 128,
                    "num_sgd_iter": 30,
                },
            }
        }

    @staticmethod
    def custom_observation_space(policy_id, raw_observation_space, raw_action_space):
        policy_id = policy_id
        new_observation_space = None
        new_action_space = None
        return new_observation_space, new_action_space

    @staticmethod
    def custom_observations(obs: dict[AgentID, ObsType], policy_id: str, wrapper: ObservationWrapper):
        agent_observations = obs[policy_id]
        if isinstance(agent_observations, list):
            # If it is stacked observations from multiple agents
            for observation in agent_observations:
                # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
                observation["image"] = wrapper.one_hot(observation["image"], wrapper.dim_sizes)
        else:
            # update the given ["image"] observation with self.one_hot() with the updated self.dim_sizes
            agent_observations["image"] = wrapper.one_hot(agent_observations["image"], wrapper.dim_sizes)

        return agent_observations

    # HW3 NOTE - custom_handle_steps is a place for reward shaping. This will execute after the default actions being executed in the step() of the environemnt
    @staticmethod
    def custom_handle_steps(
        agent,
        agent_index,
        action,
        agent_observed_objects,
        agent_reward,
        agent_terminated,
        agent_info,
        reward_schemes,
        training_scheme,
    ):
        if action == Action.pickup:
            if (
                agent.carrying
                and (agent.carrying.type == "key")
                and (agent.carrying.is_available)
                and (agent.color == agent.carrying.color)
            ):
                agent.carrying.is_available = False
                agent.carrying.is_pickedup = True
                agent_reward += reward_schemes["key_pickup_sparse_reward"]

                if training_scheme == "DTDE" or "CTDE":
                    # Mimic communiations
                    agent.mission = Mission("Go open the door with the key")
                    for obj in agent_observed_objects:  # Mimic within range communication
                        if isinstance(obj, Agent) and obj.color == agent.color:
                            obj.mission = Mission("Go move away the ball")

            elif (
                agent.carrying
                and (agent.carrying.type == "ball")
                and (agent.front_pos == agent.carrying.init_pos)
                and (agent.color != agent.carrying.color)
            ):
                agent_reward += reward_schemes["ball_pickup_dense_reward"] * agent.carrying.discount_factor
                agent.carrying.discount_factor *= agent.carrying.discount_factor

                if training_scheme == "DTDE" or "CTDE":
                    # Mimic communiations
                    agent.mission = Mission("Go move away the ball")
                    for obj in agent_observed_objects:  # Mimic within range communication
                        if isinstance(obj, Agent) and obj.color == agent.color:
                            if obj.carrying and obj.carrying.type == "key" and obj.carrying.color == obj.color:
                                obj.mission = Mission("Go open the door with the key")
                            else:
                                obj.mission = Mission("Go pick up the key")

            else:
                # Invalid pickup action
                agent_reward -= reward_schemes["invalid_pickup_dense_penalty"]

        return agent_reward, agent_terminated, agent_info
