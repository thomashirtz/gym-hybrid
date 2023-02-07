from gym_hybrid.agents import SlidingAgent


def test_sliding_agent__accelerate():
    agent = SlidingAgent(
        break_value=1,
        delta_t=1,
    )
    agent.reset(
        x=0,
        y=0,
        direction=0,
    )


def test_sliding_agent__turn():
    pass


def test_sliding_agent__break():
    pass
