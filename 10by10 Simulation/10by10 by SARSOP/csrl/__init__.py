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
        The reward function of the star-POMDP. self.reward[state] = 1-discountB if 'state' belongs to B, 0 otherwise.
        
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
        The discount factor applied to B states.
    
    """
    def __init__(self, pomdp, oa, discount=0.99999, discountB=0.99):
        self.pomdp = pomdp
        self.oa = oa
        self.discount = discount
        self.discountB = discountB  # We can also explicitly define a function of discount
        self.shape = oa.shape + pomdp.shape + (len(pomdp.A)+oa.shape[1],)
        
        
        # Create the action matrix
        self.A = np.empty(self.shape[:-1],dtype=np.object)
        for i,q,r,c in self.states():
            self.A[i,q,r,c] = list(range(len(pomdp.A))) + [len(pomdp.A)+e_a for e_a in oa.eps[q]]
        
        # Create the reward matrix
        a_array = np.array([[('a',),       ()]],dtype=np.object)
        c_array = np.array([[('c',), ()]],dtype=np.object)
        
        self.reward = np.zeros(self.shape[:-1])
        for i,q,r,c in self.states():
            label_array = pomdp.label[r,c]
            if oa.acc[q][pomdp.label[r,c]][i]:
                self.reward[i,q,r,c] = 10 # oa.acc[q][label_array[0],][i] else 0
        
        
        # Create the transition matrix
        self.transition_probs = np.empty(self.shape,dtype=np.object)  # Enrich the action set with epsilon-actions
        for i,q,r,c in self.states():
            for action in self.A[i,q,r,c]:
                if action < len(self.pomdp.A): # MDP actions
                    label_array = pomdp.label[r,c]
                    q_ = oa.delta[q][label_array]  # OA transition, [label_array[0],]
                    pomdp_states, probs = pomdp.get_transition_prob((r,c),pomdp.A[action])  # MDP transition
                    self.transition_probs[i,q,r,c][action] = [(i,q_,)+s for s in pomdp_states], probs  
                else:  # epsilon-actions
                    self.transition_probs[i,q,r,c][action] = ([(i,action-len(pomdp.A),r,c)], [1.])
        
        
        ############ below is the part for Julia Tabular code ############
        self.num_prod_states = self.shape[0]*self.shape[1]*self.shape[2]*self.shape[3]
        self.prod_states_idx = np.zeros(self.shape[:-1],dtype = 'int')
        
        count = 0
        for i,q,r,c in self.states():
            self.prod_states_idx[i,q,r,c]=count
            count=count+1
        
        # define the Julia Tabular product POMDPs transition matrix
        self.T_matrix = np.zeros((self.num_prod_states, 4, self.num_prod_states)) # T[s', a, s]
        for i,q,r,c in self.states():
            for action in range(4):
                prod_states, prod_probs = self.transition_probs[i,q,r,c][action]
                for s in prod_states:
                    self.T_matrix[self.prod_states_idx[s[0],s[1],s[2],s[3]]][action][self.prod_states_idx[i,q,r,c]] += prod_probs[prod_states.index(s)]
        #np.around(self.T_matrix, decimals=2)
        
        # define the Julia Tabular product POMDPs observation matrix
        self.O_matrix = np.zeros((self.num_prod_states, 4, self.num_prod_states)) # O[o, a, s']
        for i,q,r,c in self.states():
            obsv_states, obsv_probs = self.pomdp.get_observation_prob((r,c))
            
            for s in obsv_states:
                for q_state in range(self.shape[1]):
                    for action in range(4):
                        self.O_matrix[self.prod_states_idx[i,q_state,s[0],s[1]]][action][self.prod_states_idx[i,q,r,c]] = obsv_probs[obsv_states.index(s)]/(self.shape[1])
        
        # define the Julia Tabular product POMDPs reward matrix
        self.R_matrix = np.zeros((self.num_prod_states, 4)) # R[s, a]
        for i,q,r,c in self.states():
            for action in range(4):
                self.R_matrix[self.prod_states_idx[i,q,r,c]][action] = self.reward[i,q,r,c]
        
        
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
