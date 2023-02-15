import pytest
from gym_hybrid.agents import MovingAgent
from math import pi


def get_moving_agent(
        break_value: float,
        delta_t: float,
        x: float,
        y: float,
        speed: float,
        theta: float,
) -> MovingAgent:
    agent = MovingAgent(break_value=break_value, delta_t=delta_t)
    agent.x = x
    agent.y = y
    agent.speed = speed
    agent.theta = theta
    return agent


@pytest.mark.parametrize(
    'initial_speed, break_value, resulting_speed',
    [(2, 1, 1), (1, 1, 0), (1, 2, 0), (2, 0, 2), (0, 1, 0)]
)
def test_break(initial_speed, break_value, resulting_speed):
    agent = get_moving_agent(
        speed=initial_speed,
        break_value=break_value,
        delta_t=1, x=0, y=0, theta=0,
    )
    agent.break_()
    assert agent.speed == resulting_speed


@pytest.mark.parametrize(
    'initial_speed, acceleration_value, resulting_speed',
    [(2, 1, 3), (1, 1, 2), (2, 3, 0)]
)
def test_accelerate(initial_speed, acceleration_value, resulting_speed):
    agent = get_moving_agent(
        speed=initial_speed,
        break_value=0, delta_t=1, x=0, y=0, theta=0,
    )
    agent.accelerate(value=acceleration_value)
    assert agent.speed == resulting_speed


@pytest.mark.parametrize(
    'initial_angle, turn_value, resulting_angle',
    [(pi, pi, 0), (1, 1, 2), (1, 2 * pi, 1)]
)
def test_turn(initial_angle, turn_value, resulting_angle):
    agent = get_moving_agent(
        theta=initial_angle,
        speed=0, break_value=0, delta_t=1, x=0, y=0,
    )
    agent.turn(value=turn_value)
    assert agent.theta == resulting_angle


@pytest.mark.parametrize(
    'theta, delta_t, speed, expected_x, expected_y',
    [(pi, 1, 1, -1, 0), (0, 1, 1, 1, 0)]
)
def test_step(theta, delta_t, speed, expected_x, expected_y):
    agent = get_moving_agent(
        theta=theta,
        speed=speed, break_value=0, delta_t=delta_t, x=0, y=0,
    )
    agent._step()
    assert agent.x == pytest.approx(expected_x)
    assert agent.y == pytest.approx(expected_y)
