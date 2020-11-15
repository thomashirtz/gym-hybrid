# gym-hybrid

Repository containing a collection of environment for reinforcement learning task possessing discrete-continuous hybrid action space.

## "Moving-v0" 

<img align="right" width="300"  src="moving_v0.gif"> 

"Moving-v0" is a sandbox environment for parameterized action-space algorithms. The goal of the agent is to stop inside a target area.  
The field is a square with a side length of 2. The target area is a circle with radius 0.1. There is three discrete actions: turn, accelerate, and break. In addition to the action, there is 2 possible complementary parameters: acceleration and rotation. 
The episode terminates if one of the three condition is filled: the agent stop inside the target area, the agent leaves the field, the step count is higher than the limit (set by default at 200).

### State
The state is constituted of a list of 10 elements. The environment related values are: the current step divided by the maximum step, and the position of the target (x and y). The player related values are the position (x and y), the speed, the direction (cosine and sine), the distance related to the target, and an indicator that becomes 1 if the player is inside the target zone.

### Reward
The reward is the distance of the agent from the target of the last step minus the current distance. There is a penalty (set by default at a low value) to incentivize the learning algorithm to score as quickly as possible. A bonus reward of one is added if the player achieve to stop inside the target area. A malus of one is applied if the step count exceed the limit or if the player leaves the field.

### Actions

**The action ids are:**
1. Accelerate
2. Turn
3. Break

**The parameters are:**
1. Acceleration value
2. Rotation value

**There is two distinct way to format an action:**

Action with all the parameters (convenient if the model output all the parameters): 
```
action = (action_id, [acceleration_value, rotation_value])
```
Example of a valid actions:
```
action = (0, [0.1, 0.4])
action = (1, [0.0, 0.2])
action = (2, [0.1, 0.3])
```
Note: Only the parameter related to the action chosen will be used.

Action with only the parameter related to the action id (convenient for algorithms that output only the parameter
of the chosen action, since it doesn't require to pad the action): 
```
action = (0, [acceleration_value])
action = (1, [rotation_value])
action = (2, [])
```
Example of valid actions:
```
action = (0, [0.1])
action = (1, [0.2])
action = (2, [])
```
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
### Parameters
The parameter that can be modified during the initialization are:
* `seed` (default = None)
* `max_turn`, angle in radi that can be achieved in one step (default = np.pi/2)
* `max_acceleration`, acceleration that can be achieved in one step (if the input parameter is 1) (default = 0.5)
* `delta_t`, time step of one step (default = 0.005)
* `max_step`, limit of the number of step before the end of an environment (default = 200)
* `penalty`, value substracted to the reward each step to incentivise the agent to finish the environment quicker (default = 0.001)

Initialization with custom parameters:
```
env = gym.make('Moving-v0', seed=0, max_turn=np.pi/4, max_acceleration=1.0, delta_t=0.001, max_step=500, penalty=0.01)
```

### Render & Recording
Two testing files are avalaible to show users how to render and record the environment:
* [Python file example for recording](tests/moving_record.py)
* [Python file example for rendering](tests/moving_render.py)

### Disclaimer 
Even though the mechanics of the environment are done, maybe the hyperparameter will need some further adjustments.

### Reference
This environment is described in several papers such as:  
[Parametrized Deep Q-Networks Learning, Xiong et al., 2018](https://arxiv.org/pdf/1810.06394.pdf)  
[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space, Fan et al., 2019](https://arxiv.org/pdf/1903.01344.pdf)  

## "Slide-v0" 
Under development (Variation of the Moving-v0 with conservation of energy)

## Requirements
gym  
numpy

## Installation

Direct Installation from github using pip by running this command:
```
pip install git+https://github.com/thomashirtz/gym-hybrid#egg=gym-hybrid
```  


