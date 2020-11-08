from gym import spaces
from gym.utils import seeding
from collections import namedtuple
from typing import Tuple
import numpy as np

Action = namedtuple('Action', ['id', 'parameters'])
Target = namedtuple('Target', ['x', 'y', 'radius'])

# Action Id
TURN = 0
ACCELERATE = 1
BREAK = 2


class Agent:
    def __init__(self):
        self.x = None
        self.y = None
        self.theta = None
        self.speed = None

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
        self.max_acceleration = 0.5

        # Environment Parameters
        self.delta_t = 0.1
        self.max_step = 200
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = 0.05

        # Initialization
        self.current_step = None
        self.target = None
        self.agent = Agent()
        self.seed(seed)

        parameters_min = np.array([-self.max_turn, 0])
        parameters_max = np.array([+self.max_turn, self.max_acceleration])

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(parameters_min, parameters_max)))
        self.observation_space = spaces.Box(np.ones(10), -np.ones(10))

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
        last_distance = self.distance
        self.current_step += 1

        if action.id == TURN:
            rotation = max(min(action.parameters[TURN], self.max_turn), -self.max_turn)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = max(min(action.parameters[ACCELERATE], self.max_acceleration), 0)
            self.agent.accelerate(acceleration)
        elif action.id == BREAK:
            self.agent.break_()

        self.agent.step(self.delta_t)

        if self.distance < self.target_radius and self.agent.speed == 0:
            reward = self.get_reward(last_distance, True)
            done = True
        elif abs(self.agent.x) > self.field_size or abs(self.agent.y) > self.field_size or self.current_step > self.max_step:
            reward = -1
            done = True
        else:
            reward = self.get_reward(last_distance)
            done = False

        return self.get_state(), reward, done, {}

    def get_state(self) -> list:
        state = [
            self.agent.x,
            self.agent.y,
            self.agent.speed,
            np.cos(self.agent.theta),
            np.sin(self.agent.theta),
            self.target.x,
            self.target.y,
            self.distance,
            0 if self.distance > self.target_radius else 1,
            self.current_step / self.max_step
        ]
        return state

    def get_reward(self, last_distance, goal=False) -> float:
        return self.distance - last_distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self):
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1, y1, x2, y2):
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))
