import gym
from gym import spaces
from gym.utils import seeding

from typing import List, Tuple


class PChainEnv(gym.Env):
    """parametrized-Chain environment
    This game presents moves along a linear chain of states, with two actions:
     0) forward, which moves along the chain but returns no reward
     1) backward, which returns to the beginning and has a small reward
    The end of the chain, however, presents a large reward, and by moving
    'forward' at the end of the chain this large reward can be repeated.
    At each action, there is a small probability that the agent 'slips' and the
    opposite transition is instead taken.
    The observed state is the current state in the chain (0 to n-1) one hot encoded.

    The main difference with the n-Chain environment is that the probability to pass
    from one state to the next and the probability to get a reward depends on the
    parameter linked with this action.

    http://ceit.aut.ac.ir/~shiry/lecture/machine-learning/papers/BRL-2000.pdf
    """

    def __init__(self, n=4, small=2, large=10, epsilon=0.25, sigma=0.25):
        self.n = n
        self.small = small  # payout for 'backwards' action
        self.large = large  # payout at end of chain for 'forwards' action
        self.state = self.one_hot_encoder(0)  # Start at beginning of the chain
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Discrete(self.n)
        self.seed()
        self.means = self.np_random.uniform(-1, 1, (n, 2))  # mean of the gaussian distributions

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: Tuple[int, List[float]]):
        assert self.action_space.contains(action)

        if self.np_random.rand() < self.probability(action):
            if action[0]:  # 'backwards': go back to the beginning, get small reward
                reward = self.small
                self.state = 0
            elif self.state < self.n - 1:  # 'forwards': go up along the chain
                reward = 0
                self.state += 1
            else:  # 'forwards': stay at the end of the chain, collect large reward
                reward = self.large

        done = False
        return self.state, reward, done, {}

    def reset(self):
        self.state = 0
        return self.state

    def one_hot_encoder(self, n: int) -> List[int]:
        return [1 if i == n else 0 for i in range(self.n)]

    def probability(self, action) -> float:
        return 0.0