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
        
    delta : list of dicts
        The transition function of the DRA. delta[q][label_set] is the number of the state that the DRA makes a transition to when it consumes the label_set in the state q.
        
    acc : array, shape=(n_qs,n_pairs)
        The n_qs x n_pairs matrix that represents the accepting condition. If acc[q][i] is false then it means that q belongs to the first set of ith Rabin pair,
        if it is true, then q belongs to the second set and if it is none q doesn't belong either of them.
        
    spot_dra : spot.twa_graph
        The spot twa_graph object of the DRA.
        
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
        If it is 'T', then the cell is a trap cell, which means if the agent cannot leave the cell once it reaches it.
        The default value is None.
        
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
        The default value is 0.8.
        
    obsv_p : float, optional
        The probability that the agent observes the true state. The other possible observed states share the probability (1-p).
        The default value is 0.8.
    
    
    """
    
    def __init__(self, shape, structure=None, reward=None, label=None, A=Actions, p = 0.9, obsv_p = 0.9):
        self.shape = shape
        n_rows, n_cols = shape
        
        # Create the default structure, reward and label if they are not defined.
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
    
    
    ######## Office Problem
    
    def s_property(self,row,col,direct):
        n_rows, n_cols = self.shape
        #property('hallway'==0, 'wall'==1, 'door'==2, 'window'==3)
        s_property = np.zeros((n_rows, n_cols, 4),dtype = 'int') # n_rows, n_cols, directions('U', 'D', 'R', 'L')
        s_property[0,0,0],s_property[0,0,1],s_property[0,0,2],s_property[0,0,3]=1,1,2,3
        s_property[0,1,0],s_property[0,1,1],s_property[0,1,2],s_property[0,1,3]=1,0,2,2
        s_property[0,2,0],s_property[0,2,1],s_property[0,2,2],s_property[0,2,3]=1,1,2,2
        s_property[0,3,0],s_property[0,3,1],s_property[0,3,2],s_property[0,3,3]=1,2,1,2

        s_property[1,0,0],s_property[1,0,1],s_property[1,0,2],s_property[1,0,3]=1,1,2,1
        s_property[1,1,0],s_property[1,1,1],s_property[1,1,2],s_property[1,1,3]=0,0,2,2
        s_property[1,2,0],s_property[1,2,1],s_property[1,2,2],s_property[1,2,3]=1,1,1,2
        s_property[1,3,0],s_property[1,3,1],s_property[1,3,2],s_property[1,3,3]=2,1,1,1

        s_property[2,0,0],s_property[2,0,1],s_property[2,0,2],s_property[2,0,3]=1,1,2,1
        s_property[2,1,0],s_property[2,1,1],s_property[2,1,2],s_property[2,1,3]=0,0,1,2
        s_property[2,2,0],s_property[2,2,1],s_property[2,2,2],s_property[2,2,3]=1,1,2,1
        s_property[2,3,0],s_property[2,3,1],s_property[2,3,2],s_property[2,3,3]=1,0,1,2

        s_property[3,0,0],s_property[3,0,1],s_property[3,0,2],s_property[3,0,3]=1,1,2,3
        s_property[3,1,0],s_property[3,1,1],s_property[3,1,2],s_property[3,1,3]=0,1,0,2
        s_property[3,2,0],s_property[3,2,1],s_property[3,2,2],s_property[3,2,3]=1,1,0,0
        s_property[3,3,0],s_property[3,3,1],s_property[3,3,2],s_property[3,3,3]=0,1,1,0
        
        return s_property[row,col,direct]
    
    
    # transition probability
    def get_transition_prob(self,state,action_name):
        """Returns the list of possible next states with their probabilities when the action is taken (next_states,probs).
        The agent moves in the intented direction with a probability self.p; it can move other available states with a probability ((1-self.p)/num_available_states). 
        If the direction is not available then the agent stays at the same position with P = self.p. 
    
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
        
        n_rows, n_cols = self.shape
        states, probs = [], []
        direct_index = []
        
        for i in range(4):
            if self.s_property(state[0],state[1],i)==0 or self.s_property(state[0],state[1],i)==2:
                direct_index.append(i)
        count=len(direct_index) # count how many direction(s) is/are out
        
        # Up
        if action_name=='U':
            if 0 in direct_index:
                if count == 1:
                    states.append((state[0]-1,state[1]))
                    probs.append(1.0)
                else:
                    states.append((state[0]-1,state[1]))
                    probs.append(self.p)
                    for i in direct_index:
                        if i==1:
                            states.append((state[0]+1,state[1]))
                            probs.append((1.-self.p)/(count-1))
                        if i==2:
                            states.append((state[0],state[1]+1))
                            probs.append((1.-self.p)/(count-1))
                        if i==3:
                            states.append((state[0],state[1]-1))
                            probs.append((1.-self.p)/(count-1))
            else:
                states.append((state[0],state[1]))
                probs.append(self.p)
                for i in direct_index:
                    if i==1:
                        states.append((state[0]+1,state[1]))
                        probs.append((1.-self.p)/(count))
                    if i==2:
                        states.append((state[0],state[1]+1))
                        probs.append((1.-self.p)/(count))
                    if i==3:
                        states.append((state[0],state[1]-1))
                        probs.append((1.-self.p)/(count))
                
        # Down
        if action_name=='D':
            if 1 in direct_index:
                if count == 1:
                    states.append((state[0]+1,state[1]))
                    probs.append(1.0)
                else:
                    states.append((state[0]+1,state[1]))
                    probs.append(self.p)
                    for i in direct_index:
                        if i==0:
                            states.append((state[0]-1,state[1]))
                            probs.append((1.-self.p)/(count-1))
                        if i==2:
                            states.append((state[0],state[1]+1))
                            probs.append((1.-self.p)/(count-1))
                        if i==3:
                            states.append((state[0],state[1]-1))
                            probs.append((1.-self.p)/(count-1))
            else:
                states.append((state[0],state[1]))
                probs.append(self.p)
                for i in direct_index:
                    if i==0:
                        states.append((state[0]-1,state[1]))
                        probs.append((1.-self.p)/(count))
                    if i==2:
                        states.append((state[0],state[1]+1))
                        probs.append((1.-self.p)/(count))
                    if i==3:
                        states.append((state[0],state[1]-1))
                        probs.append((1.-self.p)/(count))
        # Right
        if action_name=='R':
            if 2 in direct_index:
                if count == 1:
                    states.append((state[0],state[1]+1))
                    probs.append(1.0)
                else:
                    states.append((state[0],state[1]+1))
                    probs.append(self.p)
                    for i in direct_index:
                        if i==0:
                            states.append((state[0]-1,state[1]))
                            probs.append((1.-self.p)/(count-1))
                        if i==1:
                            states.append((state[0]+1,state[1]))
                            probs.append((1.-self.p)/(count-1))
                        if i==3:
                            states.append((state[0],state[1]-1))
                            probs.append((1.-self.p)/(count-1))
            else:
                states.append((state[0],state[1]))
                probs.append(self.p)
                for i in direct_index:
                    if i==0:
                        states.append((state[0]-1,state[1]))
                        probs.append((1.-self.p)/(count))
                    if i==1:
                        states.append((state[0]+1,state[1]))
                        probs.append((1.-self.p)/(count))
                    if i==3:
                        states.append((state[0],state[1]-1))
                        probs.append((1.-self.p)/(count))
        # Left
        if action_name=='L':
            if 3 in direct_index:
                if count == 1:
                    states.append((state[0],state[1]-1))
                    probs.append(1.0)
                else:
                    states.append((state[0],state[1]-1))
                    probs.append(self.p)
                    for i in direct_index:
                        if i==0:
                            states.append((state[0]-1,state[1]))
                            probs.append((1.-self.p)/(count-1))
                        if i==1:
                            states.append((state[0]+1,state[1]))
                            probs.append((1.-self.p)/(count-1))
                        if i==2:
                            states.append((state[0],state[1]+1))
                            probs.append((1.-self.p)/(count-1))
            else:
                states.append((state[0],state[1]))
                probs.append(self.p)
                for i in direct_index:
                    if i==0:
                        states.append((state[0]-1,state[1]))
                        probs.append((1.-self.p)/(count))
                    if i==1:
                        states.append((state[0]+1,state[1]))
                        probs.append((1.-self.p)/(count))
                    if i==2:
                        states.append((state[0],state[1]+1))
                        probs.append((1.-self.p)/(count))
        
        #print('states'+str(states))
        #print('probs'+str(probs))
        
        return states, probs
    
    
    ### This observation function is for scenario with observation of 'hallway', 'door', 'wall' and 'window' combination on all four directions
    
    ### Office scenario full observation
    def get_observation_prob(self,state):
        """Returns the list of possible observed states with their probabilities when the current state is the input (obsv_states,obsv_probs).
        The agent observes the state according to the 'hallway'=0, 'wall'=1 and 'door'=2 combination on four directions in order;
        (ex. s_property=1,1,2,1 can be state [0,0],[1,0],[2,0],[2,2] or [3,0]. Agent located at one of these states can have probability=0.2 to observe each state)
    
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
        count=0
        for i in range(n_rows):
            for j in range(n_cols):
                for k in range(4):
                    if self.s_property(state[0],state[1],k)==self.s_property(i,j,k):
                        count=count+1
                if count==4:
                    obsv_states.append((i,j))
                count = 0
                
        for obsv_state in obsv_states:
            obsv_probs.append(1./len(obsv_states))
        
        return obsv_states, obsv_probs
    