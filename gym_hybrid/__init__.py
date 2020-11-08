from gym.envs.registration import register

register(
    id='Moving-v0',
    entry_point='gym_hybrid.envs:MovingEnv',
)