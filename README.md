# gym-hybrid

Repository containing a collection of environment for reinforcement learning task possessing discrete-continuous hybrid action space.

## "Moving-v0" 

<img align="right" width="250"  src="moving-v0.jpg"> 

"Moving-v0" is a sandbox environment for parameterized action-space algorithms. The goal of the agent is to stop inside a target area.The field is a square
with side length 2 and the target area is a circle with radius 0.1. There is three discrete actions: turn, accelerate, and break; as well as 2 possible parameters: acceleration and rotation. The state is constituted of a list of 10 elements, including the speed, the position, the direction, the position of the target, etc.  
The reward is the distance of the agent from the target of the last step minus the current distance. It is possible to add a penalty to the reward to incentivize the learning algorithm to score as quickly as possible. When the Agent is stopped in the target area, it receives a reward of one. If the agent leaves the area or takes too long (maximum step set at 200), the reward is set at minus one and the episode terminates.

### Basics
Make and initialize an environment:
```
import gym
import gym_parametrized
env = gym.make('Moving-v0')
env.reset()
```

Get the action space and the observation space:
```
ACTION_SPACE = env.action_space[0].n
PARAMETERS_SPACE = env.action_space[1].shape[0]
OBSERVATION_SPACE = env.observation_space.shape[0]
```

Run a random agent:
```
done = False
while not done:
    state, reward, done, info = env.step(env.action_space.sample())
    print(f'State: {state} Reward: {reward} Done: {done}')
```


### Action

The action ids are: 
1. Turn
2. Accelerate
3. Break

The parameters are:
1. Acceleration value
2. Rotation value

**There is two distinct way to generate an action:**

Action with all the parameters (convenient if the model output all the parameters): 
```
action = (action_id, [value_rotation, value_acceleration])
```
Example of a valid actions:
```
action = (0, [0.1, 0.4])
action = (1, [0.0, 0.2])
action = (2, [0.1, 0.3])
```

Action with only the parameter related to the action id (convenient for algorithms that output only the parameter
of the chosen action, since it doesn't require to pad the action): 
```
action = (0, [value_rotation])
action = (1, [value_acceleration])
action = (2, [])
```
Example of valid actions:
```
action = (0, [0.1])
action = (1, [0.2])
action = (2, [])
```

### Disclaimer 
Even though the mechanics of the environment are done, maybe the hyperparameter will need some further adjustments.

### Reference
This environment is described in several papers such as:  
[[Parametrized Deep Q-Networks Learning]](https://arxiv.org/pdf/1810.06394.pdf)  
[[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space]](https://arxiv.org/pdf/1903.01344.pdf)  
*The figure comes from the second reference.

## Requirements
gym  
numpy

## Installation

<details open>
    <summary>Using PIP and github url</summary>
    Direct Installation from github:
    ```
    pip install git+https://github.com/thomashirtz/gym-hybrid#egg=gym-hybrid
    ```  
</details>

<details>
    <summary>Downloading and using pip</summary>
    Download the repository and run the command:
    ```
    python -m pip install -e place-where-the-file-is-located\gym-hybrid
    ```  
</details>

<details>
    <summary>Using git clone and pip</summary>
    Run the git command:
    ```
    git clone https://github.com/thomashirtz/gym-hybrid
    ```
    Then, from the cloned repository:
    ```
    pip install .
    ```
</details>



