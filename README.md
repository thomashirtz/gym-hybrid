# gym-parametrized

Repository containing collection of environment for reinforcement learning task possessing parametrized action space.

## "Moving-v0" 
"Moving-v0" is a sandbox environment for parametrized agent. It consist in a 2x2 field, with circle target of 0.1 radius. 
The goal is to stop the player inside the target. There is three discrete action: turn, accelerate, and break; as well as 
2 possible parameters: acceleration and rotation. The state is constitued of a list of 10 elements, including the speed, the 
position, the direction.

This environment is described in several paper such as:  
[[Parametrized Deep Q-Networks Learning]](https://arxiv.org/pdf/1810.06394.pdf)  
[[Hybrid Actor-Critic Reinforcement Learning in Parameterized Action Space]](https://arxiv.org/pdf/1903.01344.pdf)





