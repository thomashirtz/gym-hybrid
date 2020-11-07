from gym.envs.registration import register

register(
    id='NChain-v0',
    entry_point='gym_parametrized.envs:PChain',
)

register(
    id='Moving-v0',
    entry_point='gym_parametrized.envs:MovingEnv',
)

register(
    id='Transition-v0',
    entry_point='gym_parametrized.envs:TransitionEnv',
)