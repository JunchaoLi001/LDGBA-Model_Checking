# LDGBA-Model_Checking
This is the implementation of Model Checking on POMDP with Limit-Deterministic Generalized B ̈uchi Automaton introduced in the work "Model-Based Motion Planning in POMDPs
with Temporal Logic Specifications".

If you find it useful in your research, please cite it using :

```
@article{li2023model,
  title={Model-based motion planning in POMDPs with temporal logic specifications},
  author={Li, Junchao and Cai, Mingyu and Wang, Zhaoan and Xiao, Shaoping},
  journal={Advanced Robotics},
  volume={37},
  number={14},
  pages={871--886},
  year={2023},
  publisher={Taylor \& Francis}
}
```
## Folders
#### 10by10 Simulation
  - Product POMDP on 10 by 10 grid world with LDGBA and frontier tracking function solved by DQN.
  - Product POMDP on 10 by 10 grid world with LDGBA and reward constraint solved by SARSOP.
  
#### Office Simulation
  - Product POMDP on 4 by 4 office environment with LDGBA and reward constraint solved by SARSOP with full observation.
  - Product POMDP on 4 by 4 office environment with LDGBA and reward constraint solved by SARSOP with single observation.
  - The video of Case 1 Turtlebot simulation with full observation.
  
## Installation:
  - Python 3.5+
  - Julia 1.6.3+ (needed for [SARSOP](https://github.com/JuliaPOMDP/SARSOP.jl))
  - Tensorflow 2.7.0 (needed for DQN)
  - [Pybullet 3](https://github.com/bulletphysics/bullet3) (needed for Pybullet Turtlebot simulation)
  
