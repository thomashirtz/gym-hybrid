import pytest
from gym_hybrid.agents import SlidingAgent
from math import pi


def get_sliding_agent(
        break_value: float,
        delta_t: float,
        x: float,
        y: float,
        speed: float,
        theta: float,
        phi: float,
) -> SlidingAgent:
    agent = SlidingAgent(break_value=break_value, delta_t=delta_t)
    agent.x = x
    agent.y = y
    agent.phi = phi
    agent.speed = speed
    agent.theta = theta
    return agent
