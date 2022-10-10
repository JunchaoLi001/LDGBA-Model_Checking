# LDGBA-Model_Checking-SARSOP
This is the implementation of Model Checking on POMDP with Limit-Deterministic Generalized B ̈uchi Automaton introduced in the work "Model-Based Motion Planning in POMDP
with Temporal Logic Specifications".

If you find it useful in your research, please cite it using :

```
@inproceedings{
  title={Model-Based Motion Planning in POMDP with Temporal Logic Specifications},
  author={Junchao Li, Shaoping Xiao, Mingyu Cai and Zhaoan Wang},
  year={2022},
}
```
## Folders
#### 10by10 Simulation
  - Product POMDP simulation on 10 by 10 grid world solved by DQN with LDGBA with frontier tracking function.
  - Product POMDP simulation on 10 by 10 grid world solved by SARSOP with reward constraint.
  
#### Office Simulation
  - Product POMDP simulation on 4 by 4 office environment solved by SARSOP with full observation.
  - Product POMDP simulation on 4 by 4 office environment solved by SARSOP with single observation.
  - The video of Case 1 Turtlebot simulation with full observation.
  
## Installation:
  - Python 3.5+
  - Julia 1.6.3+ (needed for [SARSOP](https://github.com/JuliaPOMDP/SARSOP.jl))
  - Tensorflow 2.7.0 (needed for DQN)
  - [Pybullet 3](https://github.com/bulletphysics/bullet3) (needed for Pybullet Turtlebot simulation)
  
