using Revise
using Distributed
# addprocs(2)

# @everywhere begin
using Base
using Random
using POMDPs
using Spot
using POMDPModelChecking
import Cairo
using POMDPModelTools
using POMDPModels
using POMDPSimulators
using POMDPGifs
using ProgressMeter
using BeliefUpdaters
using SARSOP
using FileIO
using JLD2
using POMDPs
using SARSOP
using POMDPModels # this contains the TigerPOMDP model
using POMDPModels: TabularPOMDP
using NPZ
using POMDPModelTools
using PointBasedValueIteration
using BeliefGridValueIteration


## import the R, O, T matrices
R = npzread("r_matrix.npy") # R = [27,4]
O = npzread("o_matrix.npy") # O = zeros(27,4,27)
T = npzread("t_matrix.npy") # T = zeros(27,4,27)
### generate the pomdp
pomdp = TabularPOMDP(T, R, O, 0.99)


## define the initial belief
### firstly define the belief updater as the discrete belief updater
up = BeliefUpdaters.DiscreteUpdater(pomdp)
### define the belief distribtuion as what we want: here, 16*4=64 product states, we want the first 16 states are uniformly distributed, rest of them are '0'.
b = zeros(length(states(pomdp)))
b[1:2] .= 1/2 # for office case
#b[97:98] .= 1/2 # for 10by10 case
#b[7] = 0
#b[16] = 0
### input the defined initial_belief to define the discrete belief
discrete_belief = BeliefUpdaters.DiscreteBelief(pomdp, ordered_states(pomdp), b)
### initialize the initial belief
POMDPs.initialize_belief(up, discrete_belief) # (updater type, belief type)
## define the initial state

POMDPs.initialstate(pomdp::TabularPOMDP) = DiscreteBelief(pomdp, b)



## SARSOP Solver for solving the policy of POMDPs
sarsop = SARSOPSolver(timeout=20)

policy = solve(sarsop, pomdp)



## the policy will be saved to a file and can be loaded in an other julia session as follows:
#policy = load_policy(pomdp, "policy.out")

## Policy can be used to map belief to actions
#=
# simulate the SARSOP policy
simulator = SARSOPSimulator(sim_num = 15, sim_len = 15,
                            policy_filename = "policy.out",
                            pomdp_filename = "model.pomdpx")
simulate(simulator)
=#





## simulate the policy and print out the path
num_steps = 150

hr = HistoryRecorder(max_steps=num_steps)
h = simulate(hr, pomdp, policy)
#h = simulate(hr, pomdp, policy, [updater [init_belief [13]]])
path = zeros(num_steps)

for i in 1:num_steps
    path[i] = h[i][:s]
    println("state")
    println(h[i][:s]) # state
    #println("action")
    #println(h[i][:a]) # action
end

## write the path as a 'npy' file for python
npzwrite("the_final_path.npy", path)

print("the starting state is: ")
println(h[1][:s])
print("Simulation ended...")
