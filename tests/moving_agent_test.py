import pytest
from gym_hybrid.agents import MovingAgent


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
