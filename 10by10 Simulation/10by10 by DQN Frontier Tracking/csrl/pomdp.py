"""Grid World POMDP Implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import importlib

if importlib.util.find_spec('matplotlib'):
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    #matplotlib.font_manager._rebuild()
    
if importlib.util.find_spec('ipywidgets'):
    from ipywidgets.widgets import IntSlider
    from ipywidgets import interact

# Up, Down, Right, Left
Actions = ['U','D','R','L'] 

class GridPOMDP():
    """This class implements a Markov Decision Process where an agent can move up, down, right or left in a 2D grid world.
    
    Attributes
    ----------
    shape : (n_rows, n_cols)
        The shape of the grid.
        
    transition_probs : array, shape=(n_rows,n_cols,n_actions)
        The transition probabilities. self.transition_probs[state][action] stores a pair of lists ([s1,s2,..],[p1,p2,...]) that contains only positive probabilities and the corresponding transitions.


    Parameters
    ----------
    shape : (n_rows, n_cols)
        The shape of the grid.
    
    structure : array, shape=(n_rows,n_cols)
        The structure of the grid function, structure[i][j] stores the type of the cell (i,j). 
        If structure[i,j] is 'E' it means the cell is empty and the agent is free to move in any direction. If it is 'B' then the cell is blocked, the agent cannot go there.
        If it is one of 'U','D','R' and 'L', the agent is free to enter the cell in any direction, but it cannot leave the cell in the opposite direction of the label.
            For example, if the label is 'D', then the agent cannot go up as if there is an obstacle there.
        
    reward : array, shape = (n_rows, n_cols)
        The reward function, reward[i,j] is the reward for the state (i,j). If reward[i,j] is None, then the state is occupied by an obstacle.
        The default value is None.

    label : array, shape = (n_rows, n_cols)
        The labeling function, label[i,j] is the set of atomic propositions the state (i,j) is labeled with.
        The default value is None.
        
    A: list
        The list of actions represented by a string.
    
    p : float, optional
        The probability that the agent moves in the intended direction. It moves in one of the perpendicular direction with probability (1-p).
        The default value is 0.9.
        
    obsv_p : float, optional
        The probability that the agent observes the true state. The other possible observed states share the probability (1-p).
        The default value is 0.9.
    
    """
    
    def __init__(self, shape, structure=None, reward=None, label=None, A=Actions, p = 0.9, obsv_p = 0.9):
        self.shape = shape
        n_rows, n_cols = shape
        self.structure = structure if structure is not None else np.full(shape,'E')
        self.reward = reward if reward is not None else np.zeros((n_rows,n_cols))
        self.label = label if label is not None else np.empty(shape,dtype=np.object); self.label.fill(()) if label is None else None
        self.p = p
        self.obsv_p = obsv_p
        self.A = A
        
        # Create the transition matrix
        self.transition_probs = np.empty((n_rows, n_cols, len(A)),dtype=np.object)
        for state in self.states():
            for action, action_name in enumerate(A):
                self.transition_probs[state][action] = self.get_transition_prob(state,action_name)
        
        
    def states(self):
        """State generator.
        
        Yields
        ------
        state: tuple
            State coordinates (i,j).
        """
        n_rows, n_cols = self.shape
        for state in product(range(n_rows),range(n_cols)):
            yield state
        
    def random_state(self):
        """Generates a random state coordinate.
        
        Returns
        -------
        state: tuple
            A random state coordinate (i,j).
        """
        n_rows, n_cols = self.shape
        state = np.random.randint(n_rows),np.random.randint(n_cols)
        return state
        
    def get_transition_prob(self,state,action_name):
        """Returns the list of possible next states with their probabilities when the action is taken (next_states,probs).
        The agent moves in the intented direction with a probability self.p; it can move sideways with a probability (1-self.p)/2. 
        If the direction is blocked by an obtacle or the agent is in a trap state then the agent stays in the same position.
    
        Parameters
        ----------
        state : tuple
            The coordinate of the state (i,j),
        
        action_name: str
            The name of the action.
        
        Returns
        -------
        out: (states,probs)
            The list of possible next states and their probabilities.
        """
        cell_type = self.structure[state]
        if cell_type in ['B', 'T']:
            return [state], np.array([1.])
        
        n_rows, n_cols = self.shape
        states, probs = [], []
        
        # South
        if action_name!='U' and state[0]+1 < n_rows and self.structure[state[0]+1][state[1]] != 'B' and cell_type != 'U':
            states.append((state[0]+1,state[1]))
            probs.append(self.p if action_name=='D' else (1-self.p)/2)
        # North
        if action_name!='D' and state[0]-1 >= 0 and self.structure[state[0]-1][state[1]] != 'B' and cell_type != 'D':
            states.append((state[0]-1,state[1]))
            probs.append(self.p if action_name=='U' else (1-self.p)/2) 
        # West
        if action_name!='R' and state[1]-1 >= 0 and self.structure[state[0]][state[1]-1] != 'B' and cell_type != 'R':
            states.append((state[0],state[1]-1))
            probs.append(self.p if action_name=='L' else (1-self.p)/2)
        # East
        if action_name!='L' and state[1]+1 < n_cols and self.structure[state[0]][state[1]+1] != 'B' and cell_type != 'L':
            states.append((state[0],state[1]+1))
            probs.append(self.p if action_name=='R' else (1-self.p)/2)
        
        # If the agent cannot move in some of the directions
        probs_sum = np.sum(probs)
        if probs_sum<1:
            states.append(state)
            probs.append(1-probs_sum)
    
        return states, probs
    
    def get_observation_prob(self,state):
        """Returns the list of possible observed states with their probabilities when the current state is the input (obsv_states,obsv_probs).
        The agent observes one of the states with their probability self.p; 
        If the observed state is blocked by an obtacle then the agent can't observe that state.
    
        Parameters
        ----------
        state : tuple
            The coordinate of the state (i,j)
        
        Returns
        -------
        out: (obsv_states,obsv_probs)
            The list of possible observed states and their probabilities.
        """
        
        n_rows, n_cols = self.shape
        obsv_states, obsv_probs = [], []
        
        # find all potential coordinates of the observed states
        for i in range(state[0]-1, state[0]+2):
            for j in range (state[1]-1, state[1]+2):
                if n_rows-1>=i>=0 and n_cols-1>=j>=0:
                    if self.structure[(i,j)]!='B':
                        obsv_states.append((i,j))
        
        if (state[0],state[1]) in obsv_states:
            for obsv_state in obsv_states:
                obsv_probs.append(self.obsv_p if obsv_state == state else (1.-self.obsv_p)/(len(obsv_states)-1))
        else: 
            for obsv_state in obsv_states:
                obsv_probs.append(1./len(obsv_states))
        return obsv_states, obsv_probs
    
    def generate_obsv_state(self,states,probs):
        """Returns the actual observed state with the input of observed states and their probabilbity
        
        Parameters
        ----------
        states : list
            The list of the coordinate of the state (i,j)
            
        probs : list
            The list of the probabilities of all the states
        
        Returns
        -------
        out: obsv_state
            
        """
        # find the index of obsv_p in the probs list
        if self.obsv_p in probs:
            index = probs.index(self.obsv_p)
            if np.random.random()<=self.obsv_p: # observe one of the other states
                obsv_state = states[index]
            else: # observe the current state
                other_states = states[:index] + states[index+1 :]
                obsv_index = np.random.choice([*range(0,len(other_states))])
                obsv_state = other_states[obsv_index]
        else:
            obsv_index = np.random.choice([*range(0,len(states))])
            obsv_state = states[obsv_index]
        
        return obsv_state
        
    
    