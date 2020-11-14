import numpy as np
from typing import Tuple
from collections import namedtuple

import gym
from gym import spaces
from gym.utils import seeding
gym.logger.set_level(40)

Target = namedtuple('Target', ['x', 'y', 'radius'])

# Action Id
TURN = 1
ACCELERATE = 0
BREAK = 2


class Agent:
    def __init__(self, break_value: float = 0.1):
        self.x = None
        self.y = None
        self.theta = None
        self.speed = None
        self.break_value = break_value

    def accelerate(self, value: float) -> None:
        self.speed += value

    def break_(self) -> None:
        self.speed = 0 if self.speed < self.break_value else self.speed - self.break_value

    def turn(self, value: float) -> None:
        self.theta = (self.theta + value) % (2 * np.pi)

    def step(self, delta_t: float = 0.1) -> None:
        self.x += delta_t * self.speed * np.cos(self.theta)
        self.y += delta_t * self.speed * np.sin(self.theta)

    def reset(self, x: float, y: float, direction: float) -> None:
        self.x = x
        self.y = y
        self.speed = 0
        self.theta = direction


class Action:
    def __init__(self, id_: int, parameters: list):
        self.id = id_
        self.parameters = parameters

    @property
    def parameter(self) -> float:
        if len(self.parameters) == 2:
            return self.parameters[self.id]
        else:
            return self.parameters[0]


class MovingEnv(gym.Env, ):
    def __init__(self, seed=None):
        # Agent Parameters
        self.max_turn = np.pi/2
        self.max_acceleration = 0.5
        self.break_value = 0.1

        # Environment Parameters
        self.delta_t = 0.005
        self.max_step = 200
        self.field_size = 1.0
        self.target_radius = 0.1
        self.penalty = 0.001

        # Initialization
        self.seed(seed)
        self.target = None
        self.viewer = None
        self.current_step = None
        self.agent = Agent(self.break_value)

        parameters_min = np.array([0, -1])
        parameters_max = np.array([1, +1])

        self.action_space = spaces.Tuple((spaces.Discrete(3),
                                          spaces.Box(parameters_min, parameters_max)))
        self.observation_space = spaces.Box(np.ones(10), -np.ones(10))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self) -> list:
        self.current_step = 0

        limit = self.field_size-self.target_radius
        low = [-limit, -limit, self.target_radius]
        high = [limit, limit, self.target_radius]
        self.target = Target(*self.np_random.uniform(low, high))

        low = [-self.field_size, -self.field_size, 0]
        high = [self.field_size, self.field_size, 2 * np.pi]
        self.agent.reset(*self.np_random.uniform(low, high))

        return self.get_state()

    def step(self, raw_action: Tuple[int, list]) -> Tuple[list, float, bool, dict]:
        action = Action(*raw_action)
        last_distance = self.distance
        self.current_step += 1

        if action.id == TURN:
            rotation = self.max_turn * max(min(action.parameter, 1), -1)
            self.agent.turn(rotation)
        elif action.id == ACCELERATE:
            acceleration = self.max_acceleration * max(min(action.parameter, 1), 0)
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

    def get_reward(self, last_distance: float, goal: bool = False) -> float:
        return last_distance - self.distance - self.penalty + (1 if goal else 0)

    @property
    def distance(self) -> float:
        return self.get_distance(self.agent.x, self.agent.y, self.target.x, self.target.y)

    @staticmethod
    def get_distance(x1: float, y1: float, x2: float, y2: float) -> float:
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def render(self, mode='human'):
        screen_width = 400
        screen_height = 400
        unit_x = screen_width / 2
        unit_y = screen_height / 2
        agentradius = 0.05

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            agent = rendering.make_circle(unit_x * agentradius)
            self.agenttrans = rendering.Transform(translation=(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y)))
            agent.add_attr(self.agenttrans)
            agent.set_color(0.1, 0.3, 0.9)
            self.viewer.add_geom(agent)

            t, r, m = 0.1 * unit_x, 0.04 * unit_y, 0.06 * unit_x
            arrow = rendering.FilledPolygon([(t, 0), (m, r), (m, -r)])
            self.arrowtrans = rendering.Transform(rotation=self.agent.theta)
            arrow.add_attr(self.arrowtrans)
            arrow.add_attr(self.agenttrans)
            arrow.set_color(0, 0, 0)
            self.viewer.add_geom(arrow)

            target = rendering.make_circle(unit_x * self.target_radius)
            targettrans = rendering.Transform(translation=(unit_x * (1 + self.target.x), unit_y * (1 + self.target.y)))
            target.add_attr(targettrans)
            target.set_color(1, 0.5, 0.5)
            self.viewer.add_geom(target)

        self.arrowtrans.set_rotation(self.agent.theta)
        self.agenttrans.set_translation(unit_x * (1 + self.agent.x), unit_y * (1 + self.agent.y))

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
