"""Control Synthesis using Reinforcement Learning.
"""
import numpy as np
import matplotlib
from itertools import product
from .pomdp import GridPOMDP
import os
import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib.pyplot as plt
    
if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact



class ControlSynthesis:
    """This class is the implementation of our main control synthesis algorithm.
    
    Attributes
    ----------
    shape : (n_pairs, n_qs, n_rows, n_cols, n_actions)
        The shape of the product POMDP.
    
    reward : array, shape=(n_pairs,n_qs,n_rows,n_cols)
        The reward function of the product POMDP.
        
    transition_probs : array, shape=(n_pairs,n_qs,n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.
    
    Parameters
    ----------
    pomdp : pomdp.GridPOMDP
        The POMDP that models the environment.
        
    oa : oa.OmegaAutomatan
        The OA obtained from the LTL specification.
        
    discount : float
        The discount factor.
    
    discountB : float
        The discount factor.
    
    """
    def __init__(self, pomdp, oa, discount=0.95, discountB=0.9):
        self.pomdp = pomdp
        self.oa = oa
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.shape = oa.shape + pomdp.shape + (len(pomdp.A)+oa.shape[1],)
        self.initial_track = [0,1]
        self.prod_b_state_size = self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]
        self.last_visited_q = []
        
        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=np.object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(pomdp.A))) + [len(pomdp.A)+e_a for e_a in oa.eps[q]]
        
        # Create the reward matrix
        self.a_array = np.array([[('a',),       ()]],dtype=np.object)
        self.b_array = np.array([[('b',),       ()]],dtype=np.object)
        self.c_array = np.array([[('c',),       ()]],dtype=np.object)
        
        
        # standard reward
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            label_array = pomdp.label[r,c]
            if oa.acc[q][label_array][i]:
                self.reward[i,q,r,c] = 10.#-self.discountB  # oa.acc[q][label_array[0],][i] else 0
            #if label_array==self.c_array[0][0] or q==2:
            #    self.reward[i,q,r,c] = -1-self.discountB
        self.reshaped_reward_init = np.reshape(self.reward,(1,self.prod_b_state_size)) # 'a' and 'b' all have rewards
        
        
        # E-LDGBA reward + standard reward (commet out this section if you don't want E-LDGBA)
        self.reward_a = np.zeros(self.shape[:-1]) # reward on 'a'
        for i,q,r,c in self.states():
            label_array = pomdp.label[r,c]
            if oa.acc[q][label_array][i] and label_array==self.a_array[0][0]:
                self.reward_a[i,q,r,c] = 10.
            
        self.reward_b = np.zeros(self.shape[:-1]) # reward on 'b'
        for i,q,r,c in self.states():
            label_array = pomdp.label[r,c]
            if oa.acc[q][label_array][i] and label_array==self.b_array[0][0]:
                self.reward_b[i,q,r,c] = 10.
        
        self.reshaped_reward_a = np.reshape(self.reward_a,(1,self.prod_b_state_size)) # 'a' has rewards
        self.reshaped_reward_b = np.reshape(self.reward_b,(1,self.prod_b_state_size)) # 'b' has rewards
        
        
        # Create the transition matrix
        self.transition_probs = np.empty(self.shape,dtype=np.object)  # Enrich the action set with epsilon-actions
        for i,q,r,c in self.states():
            for action in self.A[i,q,r,c]:
                if action < len(self.pomdp.A): # MDP actions
                    label_array = pomdp.label[r,c]
                    q_ = oa.delta[q][label_array]  # OA transition, [label_array[0],]
                    pomdp_states, probs = pomdp.get_transition_prob((r,c),pomdp.A[action])  # POMDP transition
                    self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in pomdp_states], probs  
                else:  # epsilon-actions
                    self.transition_probs[i,q,r,c][action] = ([(i,action-len(pomdp.A),r,c)], [1.])
        
                    
        # create the belief_state (product POMDPs)
        self.belief_state = np.zeros(self.shape[:-1],dtype="float16") #self.belief_state = np.zeros(self.shape[:-1])
        
        for i,q,r,c in self.states():
            if q==0:
                self.belief_state[i,q,r,c] = 1.
        
        self.belief_state = self.belief_state/(sum(sum(sum(sum(self.belief_state)))))
            
        # create the belief_transition_probs matrix (product POMDPs)
        self.belief_transition_probs = np.zeros((*self.shape,*self.shape[:-1]),dtype=np.object)
        for i,q,r,c in self.states():
            for action in self.A[i,q,r,c]:
                prod_states, prod_probs = self.transition_probs[i,q,r,c][action]
                for s in prod_states:
                    self.belief_transition_probs[i,q,r,c][action][s] = prod_probs[prod_states.index(s)]
                    
        
        # create the belief_observation_probs (product POMDPs)
        self.belief_observation_probs = np.zeros((*self.shape[:-1], self.shape[2], self.shape[3]),dtype=np.object)
        #print('shape'+str(np.shape(self.belief_observation_probs)))
        for i,q,r,c in self.states():
            obsv_states, obsv_probs = self.pomdp.get_observation_prob((r,c))
            for s in obsv_states:
                self.belief_observation_probs[i,q,r,c][s] = obsv_probs[obsv_states.index(s)]
        
    
    def states(self):
        """State generator.
        
        Yields
        ------
        state: tuple
            State coordinates (i,q,r,c)).
        """
        n_pomdps, n_qs, n_rows, n_cols, n_actions = self.shape
        for i,q,r,c in product(range(n_pomdps),range(n_qs),range(n_rows),range(n_cols)):
            yield i,q,r,c
    
    def random_state(self):
        """Generates a random state coordinate.
        
        Returns
        -------
        state: tuple
            A random state coordinate (i,q,r,c).
        """
        n_pomdps, n_qs, n_rows, n_cols, n_actions = self.shape
        pomdp_state = np.random.randint(n_rows),np.random.randint(n_cols)
        return (np.random.randint(n_pomdps),np.random.randint(n_qs)) + pomdp_state
    
    # tracking frontier function + reward function
    def Tf(self, state, next_state, reshaped_reward):
        """update tracking frontier function + reward function
        
        update 'self.track'
        
        Returns
        -------
        state: array
            the updated reward function.
        """
        reshaped_reward_update = reshaped_reward
        label_array = self.pomdp.label[state[2],state[3]] # read the label
        if label_array==self.a_array[0][0] or label_array==self.b_array[0][0]: # when label 'a' or 'b' is visited
            if next_state[1] in self.track: # check if q of next state is in frontier set
                self.track.remove(next_state[1]) # remove the q
                self.last_visited_q = next_state[1] # record the last visited q state
                #print('q state '+str(next_state[1])+' is removed (in function)')
                if label_array==self.a_array[0][0]: # if the visited label is 'a'
                    reshaped_reward_update = self.reshaped_reward_b # the reward of b remains, which all lead to q_0
                    #print('reward set as reward_b (in function)')
                if label_array==self.b_array[0][0]: # if the visited label is 'b'
                    reshaped_reward_update = self.reshaped_reward_a # the reward of a remains, which all lead to q_1
                    #print('reward set as reward_a (in function)')
        last_q = self.last_visited_q
        if self.track==[]: # if empty, then reset
            self.track=[0,1] # reset frontier set
            #print('last_q'+str(last_q))
            self.track.remove(last_q) # remove the last visited q state, keep the another q state
            #print('track reset (in function)')
            if last_q==0:
                reshaped_reward_update = self.reshaped_reward_a
            else:
                reshaped_reward_update = self.reshaped_reward_b
        return reshaped_reward_update
    
    
    def train_DQN(self,start=None,EPISODES=None,num_steps=None):
        """Performs the deep Q-learning networks returns the action values.
        
        Parameters
        ----------
        start : (r,c) = POMDP state
            The start state of the product POMDPs.
            
        EPISODES : int
            The number of episodes.
        
        num_steps : int 
            The episode length.
           
        
        """
        
        from dqn_cnn import DQNAgent
        EPISODES=EPISODES if EPISODES else 500
        num_steps=num_steps if num_steps else 200
        batch_size = 32 #num_steps
        min_steps=num_steps # start to compare with the largest num of steps
        
        # the defined belief_state size and action size
        belief_state_size = np.shape(self.belief_state)
        
        # find the size of belief_state for np.reshape
        prod_b_state_size = 1
        for i in range(len(belief_state_size)):
            prod_b_state_size = prod_b_state_size * belief_state_size[i]
        
        # action size
        prod_action_size = 4
        
        agent = DQNAgent(prod_b_state_size, self.shape[2], self.shape[3], self.shape[1], prod_action_size, self.discount, self.discountB)
        #agent.load("./save/DQN_CNN_10_frontier.h5")
        done = False
        num_episode_for_reward = 100 # print the accumulated reward per num of episode
        # initialize the list for plot
        accumulated_rewards=[]
        exploration_rate=[]
        average_rewards_hundred_steps = []
        average_epsilon = []
        
        import time
        start = time.perf_counter() # record the starting time
        
        for e in range(EPISODES):
            accumulated_rewards_per_episode=0
            done = False
            
            pomdp_state = self.pomdp.random_state()
            while self.pomdp.label[pomdp_state[0],pomdp_state[1]] == ('c',) or self.pomdp.structure[(pomdp_state[0],pomdp_state[1])]=='B':
                #print('state in c and B, state is regenerated')
                pomdp_state = self.pomdp.random_state()
            
            state = (self.shape[0]-1,self.oa.q0)+pomdp_state # select the start product state
            #print('start state: '+str(state))
            belief_state = self.belief_state # initialize the belief state
            
            self.track = [0,1] #self.initial_track # initialize frontier set = [0,1]
            #print('begin new episode, self.track is reset')
            #print('frontier reset as: [' + str(self.track)[1:-1] + ']')
            reshaped_reward = self.reshaped_reward_init
            #print('begin new episode, reward is reset to initial reward')
            
            for step in range(num_steps):
                
                # reshape the belief state as the input to acquire the action
                input_b_state = np.reshape(belief_state,(1, self.shape[2], self.shape[3], self.shape[1]))
                #print('STATE: '+str(state))
                
                # verify the existence of action and select the action from the belief_state
                action_probs = agent.act(input_b_state)
                action_probs = np.reshape(action_probs,(prod_action_size,1))
                
                i = 0
                possible_actions = []
                for i in range(len(self.A[state])):
                    possible_actions.append(action_probs[self.A[state][i]])
                action = self.A[state][np.argmax(possible_actions)]
                #print('action selected: '+str(action))
                
                ################## The agnet on POMDP simualtion
                
                # agent moves to the next state
                states, probs = self.transition_probs[state][action]
                next_state = states[np.random.choice(len(states),p=probs)]
                # find the observation states' list and the corresponding probabilities
                obsv_states, obsv_probs = self.pomdp.get_observation_prob(next_state[-2:])
                # observe the next state
                obsv_state = self.pomdp.generate_obsv_state(obsv_states, obsv_probs)
                
                ################## The belief_state update with the loops
                
                # temproraily store the current belief state
                current_belief_state = belief_state
                
                # multiply the transition probability matrix
                belief_state_after_transition = []
                for s in self.states():
                    belief_state_after_transition.append(belief_state[s]*self.belief_transition_probs[s][action])
                belief_state_after_transition = sum(belief_state_after_transition)
                # update the belief state with the observation probability matrix
                updated_belief_state = belief_state_after_transition
                for s in self.states():
                    updated_belief_state[s] = updated_belief_state[s]*self.belief_observation_probs[s][obsv_state[0], obsv_state[1]]
                belief_state = updated_belief_state/sum(sum(sum(sum(updated_belief_state))))
                
                ################# The training process
                
                # reshape all the belief to match the model requirements
                input_current_b_state = np.reshape(current_belief_state,(1, self.shape[2], self.shape[3], self.shape[1]))
                input_next_b_state = np.reshape(belief_state,(1, self.shape[2], self.shape[3], self.shape[1]))
                
                # Update the frontier set + update reward setup accordingly
                reshaped_reward = self.Tf(state, next_state, reshaped_reward)
                #print('frontier set: [' + str(self.track)[1:-1] + ']')
                
                # calculate E-LDGBA reward according to the label
                reward = np.sum(np.reshape(belief_state,(1,prod_b_state_size))*reshaped_reward)
                reward = np.float16(reward)
                ###print('reward: '+str(reward))
                
                #action_input = action
                action_input = np.zeros((1,prod_action_size),dtype="float16")
                action_input[0][action]=1.0
                ###print('action input: '+str(action_input))
                
                agent.memorize(input_current_b_state, action_input, reward, input_next_b_state, done)
                # reverse 'done value back to 'False'
                done=False
                state = next_state
                accumulated_rewards_per_episode = accumulated_rewards_per_episode + reward
                
                if step > batch_size and step%batch_size==0:
                    agent.replay(batch_size, EPISODES, e)
                    
                if step%50==0:
                    agent.target_update()
            
            print("episode: {}/{}, steps: {}, e: {:.2}".format(e, EPISODES, step+1, agent.epsilon))
            print('accumulated_rewards_per_episode: '+str(accumulated_rewards_per_episode))
            accumulated_rewards.append(accumulated_rewards_per_episode)
            exploration_rate.append(agent.epsilon)
            
            if len(accumulated_rewards)>=num_episode_for_reward:
                average_rewards_hundred_steps.append(np.average(accumulated_rewards))
                accumulated_rewards = []
                
                average_epsilon.append(np.average(exploration_rate))
                exploration_rate = []
                
        agent.save("./save/DQN_CNN_10_frontier.h5")
        
        ######### print out the total time used for computing
        finish = time.perf_counter() # record the finish time
        print(f'Finished in {round(finish-start, 4)} second(s)')
        
        #import matplotlib.pyplot as plt
        #t1 = np.arange(0, EPISODES, 1)
        t2 = np.arange(0, len(average_rewards_hundred_steps)*num_episode_for_reward, num_episode_for_reward)
        
        return average_epsilon, t2, average_rewards_hundred_steps
