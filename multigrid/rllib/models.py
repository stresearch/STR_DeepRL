""" Expected for restricted changes """

from gymnasium import spaces
from ray.rllib.models.tf.complex_input_net import ComplexInputNetwork as TFComplexInputNetwork
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.torch.complex_input_net import ComplexInputNetwork as TorchComplexInputNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.policy.rnn_sequencing import add_time_dimension
from ray.rllib.utils.framework import try_import_torch

from ray.rllib.utils.annotations import override

torch, nn = try_import_torch()


class TFModel(TFModelV2):
    """
    Basic tensorflow model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        """
        See ``TFModelV2.__init__()``.
        """
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        self.model = TFComplexInputNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.forward = self.model.forward
        self.value_function = self.model.value_function


class TorchModel(TorchModelV2, nn.Module):
    """
    Basic torch model to use with RLlib.

    Essentially a wrapper for ``ComplexInputNetwork`` that correctly deals with
    ``Dict`` observation spaces.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        """
        See ``TorchModelV2.__init__()``.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.model = TorchComplexInputNetwork(obs_space, action_space, num_outputs, model_config, name)
        self.forward = self.model.forward
        self.value_function = self.model.value_function


class TorchLSTMModel(TorchModelV2, nn.Module):
    """
    Torch LSTM model to use with RLlib.

    Processes observations with a ``ComplexInputNetwork`` and then passes
    the output through an LSTM layer.

    For configuration options (i.e. ``model_config``),
    see https://docs.ray.io/en/latest/rllib/rllib-models.html.
    """

    def __init__(
        self,
        obs_space: spaces.Space,
        action_space: spaces.Space,
        num_outputs: int,
        model_config: dict,
        name: str,
        **kwargs,
    ):
        """
        See ``TorchModelV2.__init__()``.
        """
        nn.Module.__init__(self)
        super().__init__(
            obs_space,
            action_space,
            num_outputs,
            model_config,
            name,
        )

        # Base
        self.base_model = TorchComplexInputNetwork(
            obs_space,
            action_space,
            None,
            model_config,
            f"{name}_base",
        )

        # LSTM
        self.lstm = nn.LSTM(
            self.base_model.post_fc_stack.num_outputs,
            model_config.get("lstm_cell_size", 256),
            batch_first=True,
        )

        # Action & Value
        self.action_model = nn.Linear(self.lstm.hidden_size, num_outputs)
        self.value_model = nn.Linear(self.lstm.hidden_size, 1)

        # Current LSTM output
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        # Base
        x, _ = self.base_model(input_dict, state, seq_lens)

        # LSTM
        x = add_time_dimension(
            x,
            seq_lens=seq_lens,
            framework="torch",
            time_major=False,
        )
        h, c = state[0].unsqueeze(0), state[1].unsqueeze(0)
        x, [h, c] = self.lstm(x, [h, c])

        # Out
        self._features = x.reshape(-1, self.lstm.hidden_size)
        logits = self.action_model(self._features)
        return logits, [h.squeeze(0), c.squeeze(0)]

    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return self.value_model(self._features).flatten()

    def get_initial_state(self):
        return [torch.zeros(self.lstm.hidden_size), torch.zeros(self.lstm.hidden_size)]


# NOTE - Reference for setting CTDE Model
# https://github.com/ray-project/ray/blob/master/rllib/examples/centralized_critic.py
# https://github.com/ray-project/ray/blob/master/rllib/examples/models/centralized_critic_models.py


from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import numpy as np


class TorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """
    Multi-agent model that implements a centralized value function (VF).

    Attributes
    ----------
    num_team_members : int
        The number of team members.
    model : nn.Module
        The base neural network model.
    central_vf : nn.Module
        Neural network for the centralized value function.
    
    Methods
    -------
    __init__(obs_space, action_space, num_outputs, model_config, name)
        Initialize the model.
    forward(input_dict, state, seq_lens)
        Forward pass through the model.
    central_value_function(obs, team_obs, team_actions)
        Compute the centralized value function.
    value_function()
        Get the value function from the base model (not used in this custom model).
    """


    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        """
        Initialize the model.

        Parameters
        ----------
        obs_space : gym.Space
            Observation space.
        action_space : gym.Space
            Action space.
        num_outputs : int
            Number of output units.
        model_config : dict
            Configuration dictionary for the model.
        name : str
            Name of the model.
        """
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)

        self.num_team_members = 1  # NOTE - future fix for scalbility with values from custom policy spec model_config["custom_model_config"] and name to indidate the team name

        # Base of the model
        self.model = TorchComplexInputNetwork(obs_space, action_space, num_outputs, model_config, name)

        # Central VF maps (obs, team_obs, team_act) -> vf_pred
        # Calculate input size based on observation size, number of team members and action space
        obs_size = np.prod(obs_space.shape)
        act_size = action_space.n

        # input_size = my agent's obs + team member's obs + team member's actions
        input_size = obs_size * (self.num_team_members + 1) + act_size * self.num_team_members

        self.central_vf = nn.Sequential(
            SlimFC(input_size, 16, activation_fn=nn.Tanh),
            SlimFC(16, 1),
        )

    @override(ModelV2)
    def forward(self, input_dict, state, seq_lens):
        """
        Forward pass through the model.

        Parameters
        ----------
        input_dict : dict
            Dictionary of model inputs.
        state : list
            List of RNN hidden states.
        seq_lens : tensor
            Tensor of sequence lengths.

        Returns
        -------
        tuple
            Tuple of model outputs and new RNN states.
        """
        model_out, _ = self.model(input_dict, state, seq_lens)
        return model_out, []

    def central_value_function(self, obs, team_obs, team_actions):
        """
        Compute the centralized value function.

        Parameters
        ----------
        obs : tensor
            Observations for the agent.
        team_obs : tensor
            Observations for the team.
        team_actions : tensor
            Actions taken by the team.

        Returns
        -------
        tensor
            Centralized value function predictions.
        """
        input_ = torch.cat(
            [
                obs,
                team_obs,
                torch.nn.functional.one_hot(team_actions.squeeze(1).long(), self.action_space.n).float(),
            ],
            1,
        )

        return torch.reshape(self.central_vf(input_), [-1])

    @override(ModelV2)
    def value_function(self):
        """
        Get the value function from the base model (not used in this custom model).

        Returns
        -------
        tensor
            Value function predictions.
        """
        return self.model.value_function()  # not used
