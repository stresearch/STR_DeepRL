""" Expected for restricted changes """

import numpy as np
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.annotations import override
from ray.rllib.utils.torch_utils import convert_to_torch_tensor

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
from ray.rllib.evaluation.postprocessing import compute_advantages
from ray.rllib.algorithms.ppo.ppo import PPO

# NOTE - Reference for setting CTDE Model
# https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/centralized_critic_models.py


GLOBAL_TEAM_OBS = "global_team_obs"
GLOBAL_TEAM_ACTION = "global_team_action"


class CentralizedValueMixin:
    """
    Mixin to add method for evaluating the central value function from the model.

    Methods
    -------
    __init__:
        Initialize the mixin.
    """
    def __init__(self):
        self.compute_central_vf = self.model.central_value_function


# Grabs the global team obs/act and includes it in the experience train_batch,
# and computes GAE using the central vf predictions.
def centralized_critic_postprocessing(policy, sample_batch, other_agent_batches=None, episode=None):
    """
    Process sample batch with centralized critic.

    Parameters
    ----------
    policy :
        The policy object.
    sample_batch : 
        The sample batch data.
    other_agent_batches : optional
        Data from other agents.
    episode : Episode, optional
        The current episode object.

    Returns
    -------
        Post-processed sample batch.
    """
   
    # Prep for our agent
    policy_id = policy._Policy__policy_id
    team_name = policy_id.split("_")[0]
    team_num = policy.config["env_config"]["teams"][team_name]
    team_members_agents = [f"{team_name}_{i}" for i in range(team_num) if f"{team_name}_{i}" != policy_id]

    pytorch = policy.config["framework"] == "torch"
    if (pytorch and hasattr(policy, "compute_central_vf")) or (not pytorch and policy.loss_initialized()):
        assert other_agent_batches is not None

        # also record the global team obs and actions in the trajectory
        sample_batch[GLOBAL_TEAM_OBS] = np.concatenate(
            [other_agent_batches[agent_id][2][SampleBatch.CUR_OBS] for agent_id in team_members_agents], axis=1
        )

        sample_batch[GLOBAL_TEAM_ACTION] = np.stack(
            [other_agent_batches[agent_id][2][SampleBatch.ACTIONS] for agent_id in team_members_agents], axis=1
        )

        # overwrite default VF prediction with the central VF
        sample_batch[SampleBatch.VF_PREDS] = (
            policy.compute_central_vf(
                convert_to_torch_tensor(sample_batch[SampleBatch.CUR_OBS], policy.device),
                convert_to_torch_tensor(sample_batch[GLOBAL_TEAM_OBS], policy.device),
                convert_to_torch_tensor(sample_batch[GLOBAL_TEAM_ACTION], policy.device),
            )
            .cpu()
            .detach()
            .numpy()
        )

    else:
        # Policy hasn't been initialized yet, use zeros.
        sample_batch[GLOBAL_TEAM_OBS] = np.zeros(
            (sample_batch[SampleBatch.CUR_OBS].shape[0], sample_batch[SampleBatch.CUR_OBS].shape[1] * (team_num - 1))
        )  # np.zeros_like(sample_batch[SampleBatch.CUR_OBS])
        sample_batch[GLOBAL_TEAM_ACTION] = np.zeros(
            (sample_batch[SampleBatch.ACTIONS].shape[0], team_num - 1),
        )  # np.zeros_like(sample_batch[SampleBatch.ACTIONS]) #  np.zeros((sample_batch[SampleBatch.ACTIONS].shape[0], sample_batch[SampleBatch.ACTIONS].shape[1] * (team_num-1))) #np.zeros_like(sample_batch[SampleBatch.ACTIONS])
        sample_batch[SampleBatch.VF_PREDS] = np.zeros_like(sample_batch[SampleBatch.REWARDS], dtype=np.float32)

    completed = sample_batch[SampleBatch.TERMINATEDS][-1]
    if completed:
        last_r = 0.0
    else:
        last_r = sample_batch[SampleBatch.VF_PREDS][-1]

    train_batch = compute_advantages(
        sample_batch,
        last_r,
        policy.config["gamma"],
        policy.config["lambda"],
        use_gae=policy.config["use_gae"],
    )
    return train_batch


# Copied from PPO but optimizing the central value function.
def loss_with_central_critic(policy, base_policy, model, dist_class, train_batch):
    """
    Compute loss using the central critic.

    Parameters
    ----------
    policy :
        The policy object.
    base_policy :
        The base policy object.
    model :
        The model object.
    dist_class :
        The distribution class.
    train_batch : 
        Training batch data.

    Returns
    -------
    tensor
        Loss value.
    """
    
    
    # Save original value function.
    vf_saved = model.value_function

    # Calculate loss with a custom value function.
    model.value_function = lambda: policy.model.central_value_function(
        train_batch[SampleBatch.CUR_OBS],
        train_batch[GLOBAL_TEAM_OBS],
        train_batch[GLOBAL_TEAM_ACTION],
    )
    policy._central_value_out = model.value_function()
    loss = base_policy.loss(model, dist_class, train_batch)

    # Restore original value function.
    model.value_function = vf_saved

    return loss


class CTDEPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    """
    Custom policy class for CTDE-PPO in PyTorch.

    Methods
    -------
    __init__(self, observation_space, action_space, config):
        Initialize the policy.
    loss(self, model, dist_class, train_batch):
        Compute loss.
    postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        Post-process trajectory.
    """
    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    @override(PPOTorchPolicy)
    def loss(self, model, dist_class, train_batch):
        return loss_with_central_critic(self, super(), model, dist_class, train_batch)

    @override(PPOTorchPolicy)
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        return centralized_critic_postprocessing(self, sample_batch, other_agent_batches, episode)


class CentralizedCritic(PPO):
    """
    Custom PPO class with centralized critic.

    Methods
    -------
    get_default_policy_class(cls, config):
        Get the default policy class for the agent.
    """
    @classmethod
    @override(PPO)
    def get_default_policy_class(cls, config):
        return CTDEPPOTorchPolicy
