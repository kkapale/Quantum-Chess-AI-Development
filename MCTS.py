#following are needed for MCTS
import numpy as np
import copy
import time
from collections import defaultdict


#from heuristics import material

def node_value(game_state, AI_player):
    move_code = game_state.movecode.value
    AI_player = AI_player % 2
    """To be used only on terminal node.
        Returns 1 if player wins. -1 if opponent wins and 0 is a draw"""
    if game_state.is_game_over():
        if move_code ==  2: #White win
            if AI_player == 0: #AI player is White
                return 1
            elif AI_player == 1: #AI player is Black
                return -1
        if move_code == 3: #Black win
            if AI_player == 0: #AI player is White
                return -1
            elif AI_player == 1: #AI player is Black
                return 1
        if move_code in [4,5,6]:
            return 0
    else:
        raise Exception("Someting wrong. Terminal node not reached")

class MCTS_Node():

    def __init__(self, gamestate, parent=None, parent_action=None):
        self.game_state = gamestate #GameState with board, turn, player, opponent and methods
        self.parent = parent
        self.parent_action = parent_action
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[-1] = 0
        self._results[0] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions()
        

    def untried_actions(self):
        moves_list = self.game_state.get_legal_moves() #adapt to the game
        #print(f"MCTS legal moves {moves_list}")
        self._untried_actions = moves_list
        self._untried_actions.reverse() #reversal is used as last moves are used first to expand
        return self._untried_actions

    """
    def q(self) defines the value system for the AI. it currently values a win and draw the same.
    This can definitely be improved, but be careful because anything not directly associated with wining the game may distract the AI
    """
    def q(self):
        wins = self._results[1]
        draws = self._results[0]
        loses = self._results[-1]
        return wins + draws - loses

    def n(self):
        return self._number_of_visits        #Returns the number of times each node is visited.

    def value(self):
        return self.q() / self.n() if self.n() != 0 else 0

    def expand(self):
        action = self._untried_actions.pop()
        next_state = self.game_state.__copy__()  #need deepcopy: copy.deepcopy(self.game_state)
        next_state.do_move(action) #do_move modifies the current state and returns it.
        action = next_state.get_history()[-1]  #Get the actual action (with measurement info) than the tried action
        # The above line can be modified to include the other result of the measurement if measurement is involved.
        child_node = MCTS_Node(next_state, parent=self, parent_action=action)
        
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.game_state.is_game_over() #Returns True if the game has a winner or results in a draw, False otherwise

    def rollout(self,player):

        current_rollout_state = self.game_state.__copy__() #Do need deepcopy in the rollout otherwise backpropagation does not work.
        
        """
        The following logic leads to creation of new game state from the Quantum Game State: SLOW
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_moves()
            action = self.rollout_policy(possible_moves)
            current_rollout_state.do_move(action)  #do_move modifies the current state and returns
        """
        while not current_rollout_state.is_game_over():
            current_rollout_state.random_rollout() #This creates only one QC game and does rollout: very FAST
            #backpropagate(player)
        
        return node_value(current_rollout_state, player)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)  #repeats for parent node if there is one

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.4):
        #classic MCTS equation, c_param could probably be tweaked a bit
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children] 
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def tic1(self): #these functions are used to make a timeout for the ai so it doesen't go too long, 
        #but could probably use a removal of that global variable
        global _start_time1
        _start_time1 = time.time()
        return
        
    def toc1(self):
        t_diff1 = time.time() - _start_time1
        return t_diff1
    
    def best_action(self, player, simVar = 3, timeout = 100):
        #print(f"{simVar} in file")
        l=len(self._untried_actions)
        simulation_no = simVar*l #adaptive simulations
        
        for i in range(simulation_no):
            v = self._tree_policy()
            if i == 0:
                v.tic1()
                
            reward = v.rollout(player)
            v.backpropagate(reward)
            
            if v.toc1() > timeout:
                print('hit time limit')
                break
        return self.best_child().parent_action, self.best_child().value()
