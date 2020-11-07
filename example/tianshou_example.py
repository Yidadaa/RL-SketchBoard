from typing import Iterable, List, Union, Tuple, Any, Iterable

import gym
import numpy as np

import tianshou as ts
from tianshou import trainer
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import DummyVectorEnv

import torch
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

class TianshouExampleNet(nn.Module):
  """A simple net extracts features from envs."""

  def __init__(self, state_shape: Union[Iterable[int], int], action_shape: Union[Iterable[int], int]) -> None:
    """Init net with action shape and state shape."""
    super().__init__()
    self.model = nn.Sequential(
      nn.Linear(np.prod(state_shape, dtype=np.int), 128), nn.ReLU(inplace=True),
      nn.Linear(128, 128), nn.ReLU(inplace=True),
      nn.Linear(128, 128), nn.ReLU(inplace=True),
      nn.Linear(128, np.prod(action_shape, dtype=np.int))
    )

  def forward(self, obs: np.array, state: Any = None, info: dict = {}) -> Tuple[torch.Tensor, Any]:
    """Forward model."""
    batch_size = obs.shape[0]
    if type(obs) == np.ndarray: obs = torch.Tensor(obs) # type: torch.Tensor
    return self.model(obs.view(batch_size, -1)), state


class TianshouExample():
  """A simple example shows how tianshou works."""

  def __init__(self, env_name: str, train_env_count: int = 8, test_env_count: int = 100) -> None:
    """Init env with env name.

    Args:
      env_name: name of the environment
      train_env_count: number of envs to train
      test_env_count: number of envs to test
    """
    self.env_name = env_name
    self.train_env_count = train_env_count
    self.test_env_count = test_env_count

    self.env = gym.make(env_name)
    self.train_envs = self.make_envs(self.train_env_count)
    self.test_envs = self.make_envs(self.test_env_count)

    self.net = self.build_net()
    self.optim = optim.Adam(self.net.parameters(), lr=1e-3)
    self.policy = ts.policy.DQNPolicy(self.net, self.optim, discount_factor=0.9,
      estimation_step=3, target_update_freq=320)

    self.train_collector = Collector(self.policy, self.train_envs, ReplayBuffer(size=20000))
    self.test_collector = Collector(self.policy, self.test_envs)

  def make_envs(self, env_count: int) -> DummyVectorEnv:
    """Create {env_count} environments.

    Args:
      env_count: number of envs

    Returns:
      envs: vector of envs
    """
    return ts.env.DummyVectorEnv([lambda: gym.make(self.env_name) for _ in range(env_count)])

  def build_net(self) -> nn.Module:
    """Build net for rl algo."""
    state_shape = self.env.observation_space.shape # type: Iterable[int]
    action_shape = self.env.action_space.shape # type: Iterable[int]
    return TianshouExampleNet(state_shape=state_shape, action_shape=action_shape)

  def train(self) -> None:
    """Train the policy with envs."""
    result = trainer.offpolicy_trainer(self.policy, self.train_collector, self.test_collector,
      max_epoch=1, step_per_epoch=1000, collect_per_step=10, episode_per_test=100, batch_size=64,
      train_fn=lambda _, __: self.policy.set_eps(0.1),
      test_fn=lambda _, __: self.policy.set_eps(0.05),
      stop_fn=lambda mean_rewards: mean_rewards >= self.env.spec.reward_threshold, 
      writer=SummaryWriter('log/dqn'))
    print(f'Training finished in {result["duration"]}')

  def eval(self, n_episode: int = 10) -> None:
    """Eval policy."""
    self.policy.eval()
    self.policy.set_eps(0.05)
    Collector(self.policy, self.env).collect(n_episode=n_episode, render=1 / 35)


if __name__ == '__main__':
  ts_example = TianshouExample('CartPole-v0')
  ts_example.train()
  ts_example.eval()