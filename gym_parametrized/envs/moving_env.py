import gym
from gym import spaces
from gym.utils import seeding
from collections import namedtuple
from typing import Tuple
import numpy as np

Action = namedtuple('Action', ['id', 'parameter'])
Target = namedtuple('Target', ['x', 'y', 'radius'])


class ActionName:
    ACCELERATE = 0
    BREAK = 1
    TURN = 2


class Agent:
    def __init__(self, max_speed=1.0):
        self.x = None
        self.y = None
        self.theta = None
        self.speed = None
        self.max_speed = max_speed

    def accelerate(self, value):
        self.speed += value

    def break_(self):
        self.speed = 0 if self.speed < 0.1 else self.speed - 0.1

    def turn(self, value):
        self.theta = (self.theta + value) % (2 * np.pi)

    def step(self, delta_t=1.0):
        self.x += delta_t * self.speed * np.cos(self.theta)
        self.y += delta_t * self.speed * np.sin(self.theta)

    def reset(self, x, y, direction):
        self.x = x
        self.y = y
        self.speed = 0
        self.theta = direction


class MovingEnv:
    def __init__(self, seed=None):
        # Agent Parameters
        self.max_turn = 1.0
        self.max_speed = 1.0
        self.max_acceleration = 0.5

        # Environment Parameters
        self.delta_t = 0.1
        self.max_step = 500
        self.field_size = 1.0
        self.target_radius = 0.1

        # Initialization
        self.step = None
        self.target = None
        self.agent = Agent(self.max_speed)
        self.seed(seed)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> list:
        limit = self.field_size-self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))
        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))
        return self.get_state()

    def step(self, raw_action: Tuple[int, list]):
        action = Action(*raw_action)
        self.step += 1

        if action.id == ActionName.ACCELERATE:
            acceleration = max(min(action.parameter, self.max_speed), 0)
            self.agent.accelerate(acceleration)
        elif action.id == ActionName.BREAK:
            self.agent.break_()
        elif action.id == ActionName.TURN:
            self.agent.turn(*action.parameter)
        self.agent.step(self.delta_t)

        distance = self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)
        if distance < self.target_radius and self.agent.speed == 0:
            reward = 10
            done = True
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y) > self.field_size or self.step > self.max_step:
            reward = -10
            done = True
        else:
            reward = 0
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self) -> list:
        return []

    def get_reward(self) -> float:
        return 1.

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
