from QuantumChessGame import *
from QuantumChessUtils import *
import functools
import time
import gc
import numpy as np




class MinimalQuantumChessGameState():
    """
    New class to attain minimal state representation for MCTS.
    initial_state_fen, move_history, and move_code from the previous move are all that are needed.
    This class interacts with the QuantumChessEngine through the QuantuChessGameWithAI class 
    for performing moves and to determine the end of game and game result.
    """
    def __init__(self,initial_state="", move_history=[], move_code=1):
        self.initial_state_fen = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves " if initial_state == "" else initial_state
        self.move_history = move_history
        self.movecode = move_code

    def create_QC_game(self):
        print(f"in: {time.time()}")
        new_QC_game = QuantumChessGame()
        new_QC_game.new_game({'initial_state_fen':self.initial_state_fen, 'force_turn':True, 'force_win':True, 'rest_url':"", 'max_split_moves': [10,10]})
        move_history = self.move_history
        if move_history != None:
            for move in move_history:
                _ , move_code = new_QC_game.do_move(move)
                if move_code == 0:
                    print("Error in creating QC Gamestate: move failed")
        print(f"out: {time.time()}")
        return new_QC_game
    
    def create_QC_game_fast(self):
        #print(f"in: {time.time()}")
        new_QC_game = QuantumChessGame()
        new_QC_game.new_game({'initial_state_fen':self.initial_state_fen, 'force_turn':True, 'force_win':True, 'rest_url':"", 'max_split_moves': [10,10]})
        move_history = self.move_history
        _  = [new_QC_game.do_move(move) for move in move_history] 
        #print(f"new QC game created")
        #print(f"out: {time.time()}")
        return new_QC_game
    
    def get_legal_moves(self):
        newQCgame=self.create_QC_game_fast()
        legal_moves = newQCgame.get_legal_moves()
        moves = compacted_moves(newQCgame,legal_moves)
        newQCgame.delete_game()
        #gc.collect()
        return moves

    def do_move_new_copy(self, move_str):
        newQCgame=self.create_QC_game_fast()
        _ , move_code = newQCgame.do_move(move_str)
        move_history = newQCgame.get_history()
        init_state, move_history, move_code = self.initial_state_fen, move_history, move_code
        new_game_state = MinimalQuantumChessGameState(init_state, move_history, move_code)
        newQCgame.delete_game()
        return new_game_state

    def do_move(self, move_str):
        newQCgame=self.create_QC_game_fast()
        _ , move_code = newQCgame.do_move(move_str)
        move_history = newQCgame.get_history()
        self.move_history, self.movecode = move_history, move_code
        #new_game_state = MinimalQuantumChessGameState(init_state, move_history, move_code)
        newQCgame.delete_game()
        #gc.collect()
        return self
    
    def get_history(self):
        return self.move_history

    def __copy__(self):
        init_state, move_history, move_code = self.initial_state_fen, self.move_history, self.movecode
        new_game_state = MinimalQuantumChessGameState(init_state, move_history, move_code)
        return new_game_state
    
    def game_result(self,AI_player):
        """Returns 1 if player wins. -1 if opponent wins. 0 is a draw."""
        
        move_code = self.movecode

        if move_code ==  2: #White win
            if AI_player == "White": #AI player is White
                result = 1
            elif AI_player == "Black": #AI player is Black
                result = -1
        if move_code == 3: #Black win
            if AI_player == "White": #AI player is White
                result = -1
            elif AI_player == "Black": #AI player is Black
                result = 1
        if move_code in [4,5]:
            result = 0  #Game drawn
        
        return result

    def is_game_over(self):
        if int(self.movecode) in [2, 3, 4, 5]:
            return True
        else:
            return False

    def random_rollout(self):
        newQCgame=self.create_QC_game_fast()
        
        while not newQCgame.is_game_over():
            possible_moves = compacted_moves(newQCgame,newQCgame.get_legal_moves())
            action = possible_moves[np.random.randint(len(possible_moves))]
            _ , move_code = newQCgame.do_move(action)

        move_history = newQCgame.get_history()
        
        self.move_history, self.movecode = move_history, move_code
        newQCgame.delete_game()
        
        return self
    
    def __str__(self):
        gamestatestr = self.initial_state_fen + "moves:" + str(self.move_history) + "move_code:" + str(self.move_code)
        return gamestatestr

    