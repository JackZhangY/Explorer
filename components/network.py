import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal


activations = {
  'Linear': nn.Identity(),
  'ReLU': nn.ReLU(),
  'LeakyReLU': nn.LeakyReLU(),
  'Tanh': nn.Tanh(),
  'Sigmoid': nn.Sigmoid(),
  'Softmax-1': nn.Softmax(dim=-1),
  'Softmax0': nn.Softmax(dim=0),
  'Softmax1': nn.Softmax(dim=1),
  'Softmax2': nn.Softmax(dim=2)
}


def layer_init(layer, init_type='default', nonlinearity='relu', w_scale=1.0):
  nonlinearity = nonlinearity.lower()
  # Initialize all weights and biases in layer and return it
  if init_type in ['uniform_', 'normal_']:
    getattr(nn.init, init_type)(layer.weight.data)
  elif init_type in ['xavier_uniform_', 'xavier_normal_', 'orthogonal_']:
    # Compute the recommended gain value for the given nonlinearity
    gain = nn.init.calculate_gain(nonlinearity)
    getattr(nn.init, init_type)(layer.weight.data, gain=gain)
  elif init_type in ['kaiming_uniform_', 'kaiming_normal_']:
    getattr(nn.init, init_type)(layer.weight.data, mode='fan_in', nonlinearity=nonlinearity)
  else: # init_type == 'default'
    return layer
  layer.weight.data.mul_(w_scale)
  nn.init.zeros_(layer.bias.data)
  return layer


class MLP(nn.Module):
  '''
  Multilayer Perceptron
  '''
  def __init__(self, layer_dims, hidden_act='ReLU', output_act='Linear', init_type='orthogonal_', w_scale=1.0, last_w_scale=1.0):
    super().__init__()
    # Create layers
    layers = []
    for i in range(len(layer_dims)-1):
      act = hidden_act if i+2 != len(layer_dims) else output_act
      w_s = w_scale if i+2 != len(layer_dims) else last_w_scale
      layers.append(
        layer_init(
          nn.Linear(layer_dims[i], layer_dims[i+1], bias=True), 
          init_type=init_type, 
          nonlinearity=act, 
          w_scale=w_s
        )
      )
      layers.append(activations[act])
    self.mlp = nn.Sequential(*layers) 
  
  def forward(self, x):
    for layer in self.mlp:
      x = layer(x)
    return x


class Conv2d_Atari(nn.Module):
  '''
  2D convolution neural network for Atari games
  '''
  def __init__(self, in_channels=4, feature_dim=512):
    super().__init__()
    self.conv1 = layer_init(nn.Conv2d(in_channels, 32, kernel_size=8, stride=4))
    self.conv2 = layer_init(nn.Conv2d(32, 64, kernel_size=4, stride=2))
    self.conv3 = layer_init(nn.Conv2d(64, 64, kernel_size=3, stride=1))
    linear_input_size = 7 * 7 * 64
    self.fc4 = layer_init(nn.Linear(linear_input_size, feature_dim))

  def forward(self, x):
    y = F.relu(self.conv1(x))
    y = F.relu(self.conv2(y))
    y = F.relu(self.conv3(y))
    y = y.view(y.size(0), -1)
    y = F.relu(self.fc4(y))
    return y


class Conv2d_MinAtar(nn.Module):
  '''
  2D convolution neural network for MinAtar games
  '''
  def __init__(self, in_channels, feature_dim=128):
    super().__init__()
    self.conv1 = layer_init(nn.Conv2d(in_channels, 16, kernel_size=3, stride=1))
    def size_linear_unit(size, kernel_size=3, stride=1):
      return (size - (kernel_size - 1) - 1) // stride + 1
    linear_input_size = size_linear_unit(10) * size_linear_unit(10) * 16
    self.fc2 = layer_init(nn.Linear(linear_input_size, feature_dim))
    
  def forward(self, x):
    y = F.relu(self.conv1(x))
    y = y.view(y.size(0), -1)
    y = F.relu(self.fc2(y))
    return y


class NetworkGlue(nn.Module):
  '''
  Glue two networks
  '''
  def __init__(self, net1, net2):
    super().__init__()
    self.net1 = net1
    self.net2 = net2

  def forward(self, x):
    y = self.net2(self.net1(x))
    return y


class DQNNet(nn.Module):
  def __init__(self, feature_net, value_net):
    super().__init__()
    self.feature_net = feature_net
    self.value_net = value_net

  def forward(self, obs):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute action values for all actions
    q = self.value_net(phi)
    return q


class MLPCritic(nn.Module):
  def __init__(self, layer_dims, hidden_act='ReLU', output_act='Linear', last_w_scale=1e-3):
    super().__init__()
    self.value_net = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act=output_act, last_w_scale=last_w_scale)

  def forward(self, phi):
    return self.value_net(phi).squeeze(-1)


class MLPQCritic(nn.Module):
  def __init__(self, layer_dims, hidden_act='ReLU', output_act='Linear', last_w_scale=1e-3):
    super().__init__()
    self.Q = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act=output_act, last_w_scale=last_w_scale)

  def forward(self, phi, action):
    phi_action = torch.cat([phi, action], dim=-1)
    q = self.Q(phi_action).squeeze(-1)
    return q


class MLPDoubleQCritic(nn.Module):
  def __init__(self, layer_dims, hidden_act='ReLU', output_act='Linear', last_w_scale=1e-3):
    super().__init__()
    self.Q1 = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act=output_act, last_w_scale=last_w_scale)
    self.Q2 = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act=output_act, last_w_scale=last_w_scale)

  def forward(self, phi, action):
    phi_action = torch.cat([phi, action], dim=-1)
    q1 = self.Q1(phi_action).squeeze(-1)
    q2 = self.Q2(phi_action).squeeze(-1)
    return q1, q2


class Actor(nn.Module):
  def distribution(self, phi):
    raise NotImplementedError

  def log_prob_from_distribution(self, action_distribution, action):
    raise NotImplementedError

  def forward(self, phi, action=None):
    # Compute action distribution and the log_prob of given actions
    action_distribution = self.distribution(phi)
    if action is None:
      action = action_distribution.sample()
    log_prob = self.log_prob_from_distribution(action_distribution, action)
    return action_distribution, action, log_prob


class MLPCategoricalActor(Actor):
  def __init__(self, layer_dims, hidden_act='ReLU', output_act='Linear', last_w_scale=1e-3):
    super().__init__()
    self.logits_net = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act=output_act, last_w_scale=last_w_scale)

  def distribution(self, phi):
    logits = self.logits_net(phi)
    return Categorical(logits=logits)

  def log_prob_from_distribution(self, action_distribution, action):
    return action_distribution.log_prob(action)


class MLPGaussianActor(Actor):
  def __init__(self, action_lim, layer_dims, hidden_act='ReLU', log_std_bounds=(-20, 2), last_w_scale=1e-3):
    super().__init__()
    self.actor_net = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act='Tanh', last_w_scale=last_w_scale)
    # The action std is independent of states
    self.action_log_std = nn.Parameter(last_w_scale*torch.zeros(layer_dims[-1]))
    self.log_std_min, self.log_std_max = log_std_bounds
    self.action_lim = action_lim

  def distribution(self, phi):
    action_mean = self.action_lim * self.actor_net(phi)
    # Constrain log_std inside [log_std_min, log_std_max]
    action_log_std = torch.clamp(self.action_log_std, self.log_std_min, self.log_std_max)
    return Normal(action_mean, action_log_std.exp())
    
  def log_prob_from_distribution(self, action_distribution, action):
    # Last axis sum needed for Torch Normal distribution
    return action_distribution.log_prob(action).sum(axis=-1)


class MLPSquashedGaussianActor(Actor):
  def __init__(self, action_lim, layer_dims, hidden_act='ReLU', log_std_bounds=(-20, 2), last_w_scale=1e-3):
    super().__init__()
    self.actor_net = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act='Linear', last_w_scale=last_w_scale)
    self.log_std_min, self.log_std_max = log_std_bounds
    self.action_lim = action_lim

  def distribution(self, phi):
    action_mean, action_log_std = self.actor_net(phi).chunk(2, dim=-1)
    # Constrain log_std inside [log_std_min, log_std_max]
    action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
    return action_mean, Normal(action_mean, action_log_std.exp())

  def log_prob_from_distribution(self, action_distribution, action):
    # NOTE: Check out the original SAC paper and https://github.com/openai/spinningup/issues/279 for details
    log_prob = action_distribution.log_prob(action).sum(axis=-1)
    log_prob -= (2*(math.log(2) - action - F.softplus(-2*action))).sum(axis=-1)
    return log_prob

  def forward(self, phi, deterministic=False):
    # Compute action distribution and the log_prob of given actions
    action_mean, action_distribution = self.distribution(phi)
    if deterministic:
      action = action_mean
    else:
      action = action_distribution.rsample()
    # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
    log_prob = self.log_prob_from_distribution(action_distribution, action)
    action = self.action_lim * torch.tanh(action)
    return action_distribution, action, log_prob


class MLPDeterministicActor(Actor):
  def __init__(self, action_lim, layer_dims, hidden_act='ReLU', last_w_scale=1e-3):
    super().__init__()
    self.actor_net = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act='Tanh', last_w_scale=last_w_scale)
    self.action_lim = action_lim
  
  def forward(self, phi):
    return self.action_lim * self.actor_net(phi)


class MLPRepGaussianActor(MLPDeterministicActor):
  def __init__(self, action_lim, layer_dims, hidden_act='ReLU', log_std_bounds=(-20, 2), last_w_scale=1e-3):
    super().__init__(action_lim, layer_dims, hidden_act, last_w_scale)
    self.actor_net = MLP(layer_dims=layer_dims, hidden_act=hidden_act, output_act='Linear', last_w_scale=last_w_scale)
    self.log_std_min, self.log_std_max = log_std_bounds
    self.action_lim = action_lim

  def distribution(self, phi):
    action_mean, action_log_std = self.actor_net(phi).chunk(2, dim=-1)
    # 
    action_mean = self.action_lim * torch.tanh(action_mean)
    # Constrain log_std inside [log_std_min, log_std_max]
    action_log_std = torch.clamp(action_log_std, self.log_std_min, self.log_std_max)
    return action_mean, Normal(action_mean, action_log_std.exp())

  def forward(self, phi, deterministic=False):
    # Compute action distribution and the log_prob of given actions
    action_mean, action_distribution = self.distribution(phi)
    if deterministic:
      action = action_mean
    else:
      action = action_distribution.rsample()
    return action


class REINFORCENet(nn.Module):
  def __init__(self, feature_net, actor_net):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    self.actor_params = list(self.feature_net.parameters()) + list(self.actor_net.parameters())

  def forward(self, obs, action=None):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    _, action, log_prob = self.actor_net(phi, action)
    return {'action': action, 'log_prob': log_prob}


class ActorCriticNet(nn.Module):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__()
    self.feature_net = feature_net
    self.actor_net = actor_net
    self.critic_net = critic_net
    self.actor_params = list(self.feature_net.parameters()) + list(self.actor_net.parameters())
    self.critic_params = list(self.feature_net.parameters()) + list(self.critic_net.parameters())

  def forward(self, obs, action=None):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute state value
    v = self.critic_net(phi)
    # Sample an action
    action_distribution, action, log_prob = self.actor_net(phi, action)
    return {'action': action, 'log_prob': log_prob, 'v': v}


class SACNet(ActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs, deterministic=False):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action_distribution, action, log_prob = self.actor_net(phi, deterministic)
    # Compute state-action value
    q1, q2 = self.critic_net(phi, action)
    return {'action': action, 'log_prob': log_prob, 'q1': q1, 'q2': q2}
  
  def get_q(self, obs, action):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute state-action value
    q1, q2 = self.critic_net(phi, action)
    return q1, q2


class DeterministicActorCriticNet(ActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action = self.actor_net(phi)
    # Compute state-action value
    q = self.critic_net(phi, action)
    return {'action': action, 'q': q}

  def get_q(self, obs, action):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute state-action value
    q = self.critic_net(phi, action)
    return q


class TD3Net(SACNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action = self.actor_net(phi)
    # Compute state-action value
    q1, q2 = self.critic_net(phi, action)
    return {'action': action, 'q1': q1, 'q2': q2}


class RepActorCriticNet(DeterministicActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs, deterministic=False):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action = self.actor_net(phi, deterministic)
    # Compute state-action value
    q = self.critic_net(phi, action)
    return {'action': action, 'q': q}


class RPGNet(ActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs, deterministic=False):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action = self.actor_net(phi, deterministic)
    # Compute state-action value
    reward = self.critic_net(phi, action)
    return {'action': action, 'reward': reward}
  
  def get_reward(self, obs, action):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute predicted reward
    reward = self.critic_net(phi, action)
    return reward


class DRPGNet(ActorCriticNet):
  def __init__(self, feature_net, actor_net, critic_net):
    super().__init__(feature_net, actor_net, critic_net)

  def forward(self, obs, deterministic=False):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Sample an action
    action = self.actor_net(phi, deterministic)
    # Compute state-action value
    reward1, reward2 = self.critic_net(phi, action)
    return {'action': action, 'reward1': reward1, 'reward2': reward2}
  
  def get_reward(self, obs, action):
    # Generate the latent feature
    phi = self.feature_net(obs)
    # Compute predicted reward
    reward1, reward2 = self.critic_net(phi, action)
    return reward1, reward2