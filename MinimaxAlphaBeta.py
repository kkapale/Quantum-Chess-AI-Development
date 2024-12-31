from QuantumChessGame import *
from QuantumChess import *

#from QuantumChess import QuantumChessGame, MoveCode, format_moves
#from QuantumChess import is_quantum_move, is_split_move
#from QuantumChess import apply_move_corrections, square_number_to_str
#from QuantumChess import number_to_piece, piece_to_number, MoveType
import functools
import time
from numba import njit, jit
#
#
# Some optimization experiments with python: Method 1
#
maxCacheSize = 1024
def ignore_unhashable(func): 
    uncached = func.__wrapped__
    attributes = functools.WRAPPER_ASSIGNMENTS + ('cache_info', 'cache_clear')
    @functools.wraps(func, assigned=attributes) 
    def wrapper(*args, **kwargs): 
        try: 
            return func(*args, **kwargs) 
        except TypeError as error: 
            if 'unhashable type' in str(error): 
                return uncached(*args, **kwargs) 
            raise 
    wrapper.__uncached__ = uncached
    return wrapper
#
#Custom Decorator function
def listToTuple(function):
    def wrapper(*args):
        args = [tuple(x) if type(x) == list else x for x in args]
        result = function(*args)
        result = tuple(result) if type(result) == list else result
        return result
    return wrapper
#
def tupleToList(function):
    def wrapper2(*args2):
        args2 = [list(x) if type(x) == tuple else x for x in args2]
        result2 = function(*args2)
        result2 = list(result2) if type(result2) == tuple else result2
        return result2
    return wrapper2
#
# 
# Some optimization experiments with python: Method 2
# # Inspired by https://gist.github.com/harel/9ced5ed51b97a084dec71b9595565a71
# # Taken from https://gist.github.com/adah1972/f4ec69522281aaeacdba65dbee53fade
#
from collections import namedtuple
import json
import six

Serialized = namedtuple('Serialized', 'json')


def hashable_cache(cache):
    def hashable_cache_internal(func):
        def deserialize(value):
            if isinstance(value, Serialized):
                return json.loads(value.json)
            else:
                return value

        def func_with_serialized_params(*args, **kwargs):
            _args = tuple([deserialize(arg) for arg in args])
            _kwargs = {k: deserialize(v) for k, v in six.viewitems(kwargs)}
            return func(*_args, **_kwargs)

        cached_func = cache(func_with_serialized_params)

        @functools.wraps(func)
        def hashable_cached_func(*args, **kwargs):
            _args = tuple([
                Serialized(json.dumps(arg, sort_keys=True))
                if type(arg) in (list, dict) else arg
                for arg in args
            ])
            _kwargs = {
                k: Serialized(json.dumps(v, sort_keys=True))
                if type(v) in (list, dict) else v
                for k, v in kwargs.items()
            }
            return cached_func(*_args, **_kwargs)
        hashable_cached_func.cache_info = cached_func.cache_info
        hashable_cached_func.cache_clear = cached_func.cache_clear
        return hashable_cached_func

    return hashable_cache_internal
"""# Example usage below

from cachetools.func import lru_cache

@hashable_cache(functools.lru_cache())
def fib(n):
    assert n >= 0
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fib(n - 1) + fib(n - 2)
"""
#
# New functions needed to the augment QuantumChessGame class
#
def str_to_square_number(str) -> int:
    col=ord(str[0])-97
    row=int(str[1])-1
    square_number = row * 8 + col 
    return square_number
#
def is_square_empty(self,square_number):
    pce = chr(self.gamedata.pieces[square_number])
    if pce == 'x':
        return True
    else:
        return False
#
def are_split_targets_empty(self,target1_str,target2_str):
    target1_square = str_to_square_number(target1_str)
    target2_square = str_to_square_number(target2_str)
    if self.is_square_empty(target1_square) and self.is_square_empty(target2_square):
        return True
    else:
        return False
#
def is_spit_move_symmetric(self,move):
    source, target = move.split("^")
    if len(target)==4 and self.are_split_targets_empty(target[0:2],target[2:4]):
        return True
    else:
        return False

#method 1: something here does not work
#@ignore_unhashable
#@listToTuple
#@functools.lru_cache(maxsize=maxCacheSize)
#@tupleToList
#method 2: yet to be fully tested.
#@hashable_cache(functools.lru_cache())
def remove_symmetric_splits(self, moves_list):
    for move in moves_list:
        if "^" in move:
            if self.is_spit_move_symmetric(move):
                source, target = move.split("^")
                symmetric_move = source + "^" + target[2:4] + target[0:2]
                moves_list.remove(symmetric_move)
    return moves_list

#
def print_board_and_probabilities(self):
    """Renders a ASCII diagram showing the board probabilities and pieces."""
    s = ''
    s += ' +-------------------------------------------------+\n'
    for y in reversed(range(8)):
        s += str(y + 1) + '| '
        for x in range(8):
            bit = y * 8 + x
            prob = str(int(100 * self.gamedata.probabilities[bit]))
            bp=bit
            pce = chr(self.gamedata.pieces[bp])
            if len(prob) <= 2:
                s += ' '
            if prob == '0':
                s += ' . '
            else:
                s += prob
                s += ':'
                s += (pce)
            if len(prob) < 2:
                s += ' '
            s += ' '
        s += '|\n'
    s += ' +-------------------------------------------------+\n'
    
    for x in range(8):
        s +=  '     ' + chr(ord('a') + x)

    print(s)
#
def print_board(self):
    """Renders a ASCII diagram showing the board pieces."""
    s = ''
    s += ' +----------------------------------+\n'
    for y in reversed(range(8)):
        s += str(y + 1) + '| '
        for x in range(8):
            bit = y * 8 + x
            pce = chr(self.gamedata.pieces[bit])
            s += ' '
            if pce == 'x':
                s += '.'
            else:
                s += pce
            s += ' '
            s += ' '
        s += ' |\n'
    s += ' +----------------------------------+\n    '
    for x in range(8):
        s += chr(ord('a') + x) + '   '

    print(s)
#
#
#@functools.lru_cache(maxsize=1024)
#@jit(forceobj=True)
def get_board_score_PB(self, Player):
    """
    Reference: https://www.chessprogramming.org/Simplified_Evaluation_Function
    This function implements improved board scoring using the piece values based on their positions. Layout is [a1b1...a8b8.h8]
    Returns the board score using piece values and probabilities for the specified maximizer: White or Black
    """
    
    empty_list = [  0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0,
                    0,  0,  0,  0,  0,  0,  0,  0]
                    
    pawn_value_White_list = [ 0,   0,   0,   0,   0,   0,   0,   0,
                              5,  10,  10, -20, -20,  10,  10,   5,
                              5,  -5, -10,   0,   0, -10,  -5,   5,
                              0,   0,   0,  20,  20,   0,   0,   0,
                              5,   5,  10,  25,  25,  10,   5,   5,
                             10,  10,  20,  30,  30,  20,  10,  10,
                             50,  50,  50,  50,  50,  50,  50,  50,
                              0,   0,   0,   0,   0,   0,   0,   0]
                         
    pawn_value_Black_list = list(reversed(pawn_value_White_list))
    
    knight_value_White_list = [
                          -50, -40, -30, -30, -30, -30, -40, -50,
                          -40, -20,   0,   0,   0,   0, -20, -40,
                          -30,  0,   10,  15,  15,  10,   0, -30,
                          -30,  5,   15,  20,  20,  15,   5, -30,
                          -30,  0,   15,  20,  20,  15,   0, -30,
                          -30,  5,   10,  15,  15,  10,   5, -30,
                          -40, -20,   0,   5,   5,   0, -20, -40,
                          -50, -40, -30,- 30, -30, -30, -40, -50]
    knight_value_Black_list = list(reversed(knight_value_White_list)) #not really needed
        
    bishop_value_White_list = [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10,   5,   0,   0,   0,   0,   5, -10,
                -10,  10,  10,  10,  10,  10,  10, -10,
                -10,   0,  10,  10,  10,  10,   0, -10,
                -10,   5,   5,  10,  10,   5,   5, -10,
                -10,   0,   5,  10,  10,   5,   0, -10,
                -10,   0,   0,   0,   0,   0,   0, -10,
                -20, -10, -10, -10, -10, -10, -10, -20]
    bishop_value_Black_list = list(reversed(bishop_value_White_list))
    
    rook_value_White_list = [ 0,  0,  0,  5,  5,  0,  0,  0,
                             -5,  0,  0,  0,  0,  0,  0, -5,
                             -5,  0,  0,  0,  0,  0,  0, -5,
                             -5,  0,  0,  0,  0,  0,  0, -5,
                             -5,  0,  0,  0,  0,  0,  0, -5,
                             -5,  0,  0,  0,  0,  0,  0, -5,
                              5, 10, 10, 10, 10, 10, 10,  5,
                              0,  0,  0,  0,  0,  0,  0,  0]
    rook_value_Black_list = list(reversed(rook_value_White_list))
    
    queen_value_White_list = [
            -20, -10, -10, -5, -5, -10, -10, -20,
            -10,   0,   0,  0,  0,   0,   0, -10,
            -10,   0,   5,  5,  5,   5,   0, -10,
             -5,   0,   5,  5,  5,   5,   0,  -5,
              0,   0,   5,  5,  5,   5,   0,  -5,
            -10,   5,   5,  5,  5,   5,   0, -10,
            -10,   0,   5,  0,  0,   0,   0, -10,
            -20, -10, -10, -5, -5, -10, -10, -20]
    queen_value_Black_list = list(reversed(queen_value_White_list))
    
    king_middle_game_White_list = [
            20,  30,  10,   0,   0,  10,  30,  20,
            20,  20,   0,   0,   0,   0,  20,  20,
           -10, -20, -20, -20, -20, -20, -20, -10,
            20, -30, -30, -40, -40, -30, -30, -20,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30,
           -30, -40, -40, -50, -50, -40, -40, -30]
    king_middle_game_Black_list = list(reversed(king_middle_game_White_list))
    
    king_end_game_White_list = [
            50, -30, -30, -30, -30, -30, -30, -50,
           -30, -30,   0,   0,   0,   0, -30, -30,
           -30, -10,  20,  30,  30,  20, -10, -30,
           -30, -10,  30,  40,  40,  30, -10, -30,
           -30, -10,  30,  40,  40,  30, -10, -30,
           -30, -10,  20,  30,  30,  20, -10, -30,
           -30, -20, -10,   0,   0, -10, -20, -30,
           -50, -40, -30, -20, -20, -30, -40, -50]
                           
    king_end_game_Black_list = list(reversed(king_end_game_White_list))
    
    piece_value_position_dict = {
        "P" :  pawn_value_White_list,
        "p" :  pawn_value_Black_list,
        "N" :  knight_value_White_list,
        "n" :  knight_value_White_list,
        "B" :  bishop_value_White_list,
        "b" :  bishop_value_Black_list,
        "R" :  rook_value_White_list,
        "r" :  rook_value_Black_list,
        "Q" :  queen_value_White_list,
        "q" :  queen_value_Black_list,
        "Km" : king_middle_game_White_list,
        "km" : king_middle_game_Black_list,
        "Ke" : king_end_game_White_list,
        "ke" : king_end_game_Black_list,
        "x"  : empty_list
        }
        
    piece_value = { "P":  -100, #change back eventually
                    "p":  -100,
                    "N":   320,
                    "n":  -320,
                    "B":   325,
                    "b":  -325,
                    "R":   500,
                    "r":  -500,
                    "Q":   975,
                    "q":  -975,
                    "K": 32767,
                    "k":-32767,
                    "x":     0
    }
    score = float(0)
    end_game = True
    
    position = 0
    position_value = 0
    for i in self.gamedata.pieces:
            #first determine standared material score
            key = str(chr(i))
            score += piece_value[key] * self.gamedata.probabilities[position]
            #Now determing position dependent material score
            if key =="K" or key =="k":
                if end_game==True:
                    key = key+"e"
                else:
                    key = key+"m"
            piece_value_list = piece_value_position_dict[key]
            position_value += piece_value_list[position] * self.gamedata.probabilities[position]
            #score += position_value
            position += 1
    if Player=='White':
        return score + position_value
    else:
        return -score + position_value
#
def get_move_value(self, move, Player):
        pre_move_score = get_board_score_PB(self, Player)
        gamedata, move_code = self.do_move(move)
        if move_code == 0 or move_code == 6:
            #print(f'Encountered issue in getting score for move: {move} while ordering. Move Code: 0')
            #print('Check if the move is possible on the board below:')
            #self.print_board_and_probabilities()
            #print(f"{Player}'s Score for {move} is 100000")
            return 100000
        elif move_code == 6:
            #print(f'Encountered issue in getting score for move: {move} while ordering. Move Code: 6')
            #self.print_board_and_probabilities()
            self.undo_move()
            #print(f"{Player}'s Score for {move} is 100000")
            return 100000
        else:
            post_move_score = get_board_score_PB(self, Player)
            self.undo_move()
            move_score = post_move_score - pre_move_score
            #if "^" in move and move_score !=0:
            #print(f"{Player}'s Score for {move} is {move_score}")
            return move_score
## method 1
#@ignore_unhashable
#@listToTuple
#@functools.lru_cache(maxsize=maxCacheSize)
#@tupleToList
#method 2: This is better as it maintains the nature of the input and output lists
#@hashable_cache(functools.lru_cache())
def order_moves_N(self, moves_list, Player, isMax):
    """ 
    Carries out move ordering in the descending order of the move score.
    """

    if Player == "White" and isMax == True:
        order_moves_for = "White"
    elif Player == "White" and isMax == False:
        order_moves_for = "Black"
    elif Player == "Black" and isMax == True:
        order_moves_for = "Black"
    elif Player == "Black" and isMax == False:
        order_moves_for = "White"

    if moves_list != []:
        moves_in_order = sorted(moves_list, key = lambda x: get_move_value(self, x, order_moves_for), reverse=True)
        #print(f'{moves_in_order}')
        return moves_in_order
    else:
        #print([])
        return []
#
#@ignore_unhashable
#@listToTuple
#@functools.lru_cache(maxsize=maxCacheSize)
#@tupleToList
def order_moves(self, moves_list, Player, isMax):
    """ Carries out move ordering in the descending order of the move score.
    """

    if Player == "White" and isMax == True:
        order_moves_for = "White"
    elif Player == "White" and isMax == False:
        order_moves_for = "Black"
    elif Player == "Black" and isMax == True:
        order_moves_for = "Black"
    elif Player == "Black" and isMax == False:
        order_moves_for = "White"

    def get_move_value(move, Player):
        pre_move_score = self.get_board_score(Player)
        gamedata, move_code = self.do_move(move)

        if move_code == 0 or move_code == 6:
            #print(f'Encountered issue in getting score for move: {move} while ordering. Move Code: {move_code}')
            #self.print_board_and_probabilities()
            #move_history = format_moves(move_history)
            #print("Move history so far:")
            #print(" ".join(move_history[:]))
            #print(f'Current Player: {self.current_player}')
            #print(f'pre_move_score: {pre_move_score}')
            #self.undo_move()
            return -100000
        post_move_score = self.get_board_score(Player)
        self.undo_move()
        move_score = post_move_score - pre_move_score
        #if "^" in move and move_score !=0:
        #    print(f"{Player}'s Score for {move} is {move_score}")
        return move_score
        
    moves_in_order = sorted(moves_list, key = lambda x: get_move_value(x, order_moves_for), reverse=True)
    return moves_in_order
#


#@functools.lru_cache(maxsize=maxCacheSize)
#@jit(forceobj=True)
def get_board_score(self, Player):
    """Returns the board score using piece values and probabilities for the specified maximizer: White or Black"""
    piece_value = { "P":  -100, #change back eventually *******
                    "p":  -100,
                    "N":   320,
                    "n":  -320,
                    "B":   325,
                    "b":  -325,
                    "R":   500,
                    "r":  -500,
                    "Q":   975,
                    "q":  -975,
                    "K": 32767,
                    "k":-32767,
                    "x":     0
    }
    score = int(0)
    position = 0
    for i in self.gamedata.pieces:
            key = chr(i)
            score += piece_value[key] * self.gamedata.probabilities[position]
            position += 1
    if Player=='White':
        return score
    else:
        return -score
#
def get_fen_code(self):
    """
    Gives the fen code for the current state of the gamedata
    Feature Needed: The last piece of the code based on White or Black
    player's turn.
    """
    s = ''
    s += 'position fen '
    for rank in range(8):
        for file in range(8):
            position = rank * 8 + file
            s += chr(self.gamedata.pieces[position])
        if rank != 7:
            s += '/'
    
    """ # The folllowing feature is not implemented correctly."""
    if self.current_player == 0:
        s += '  w - - 0 1 moves'
    else:
        s += '  b - - 0 1 moves' 

    replace_this = ["xxxxxxxx","xxxxxxx","xxxxxx","xxxxx","xxxx","xxx","xx","x"]
    new_s = ["","","","","","","","",""]
    new_s[0] = s
    for i in range(1,9):
        num = 9 - i
        new_s[i] = new_s[i-1].replace(replace_this[i-1], str(num))
        
    return new_s[8]
#
def get_classical_moves(self):
    """Get all the legal classical moves"""

    classical_moves = [x 
        for x in format_moves(self.get_legal_moves()) if '^' not in x]
    
    classical_moves_no_duplicates = []
    for i in classical_moves:
        if i not in classical_moves_no_duplicates:
            classical_moves_no_duplicates.append(i)
    
    return classical_moves_no_duplicates
#
def get_quantum_moves(self):
    """Get all the legal quantum moves"""
    quantum_moves = [x for x in format_moves(self.get_legal_moves()) if '^' in x]
    return quantum_moves
#
def separate_split_merge_moves(self):
    """Get all the legal split and merge moves as seperate sets"""
    split_moves=[]
    merge_moves=[]
    all_quantum_moves = self.get_quantum_moves()
    for move in all_quantum_moves:
        source, target = move.split("^")
        if len(target)==4:
            split_moves.append(move)
        elif len(target)==2:
            merge_moves.append(move)
    return split_moves, merge_moves
#
def get_compacted_moves(self):
    """Get all the legal moves with symmetric quantum moves removed"""
    all_valid_moves = format_moves(self.get_legal_moves())
    return self.remove_symmetric_splits(all_valid_moves)
#
def get_truncated_moves(self, player, isMax):
    """Get truncated and ordered set of classical(all), split (15) and merge(10) top scoring moves
    TODO
    More sophisticated move ordering based on quiscence search and improved heuristics"""
    #split_keep=15
    #merge_keep=10
    #C_moves = self.get_classical_moves()
    #O_C_moves = self.order_moves_N(C_moves, player, isMax)
    #Split_moves, Merge_moves = self.separate_split_merge_moves()
    #Split_c_moves = self.remove_symmetric_splits(Split_moves)
    #O_Split_moves = self.order_moves_N(Split_c_moves, player, isMax)
    #O_Merge_moves = self.order_moves_N(Merge_moves, player, isMax)
    #O_moves = O_C_moves[:] + O_Split_moves[:] +  O_Merge_moves[:]
    #print(f'concatenated moves: {O_moves}')
    compacted_moves = self.get_compacted_moves()
    keep_ratio=0.8
    truncated_moves = self.order_moves_N(compacted_moves, player, isMax)
    max_idx = int(keep_ratio*len(truncated_moves))
    #print(f'truncated moves: {truncated_moves}')
    return truncated_moves[0:max_idx]
#
def is_game_over(self,move_code):
    """Returns True if the game has a winner or results in a draw and False if game needs to contiue"""
    if move_code in [2, 3, 4, 5]:
        return True
    else:
        return False
#
def is_favorable_move(self, move, Player) -> bool:
    """
    Returns True if the move is promotion and not en_passant(not implemented) and if it is a capture move the move score needs to be favorable
    TODO: There is a concept of attackers and comparison between attackers
    """
    gamedata, move_code = self.do_move(move)
    move_to_check = self.get_history()[-1]
    self.undo_move()
    if move_to_check['promotion_piece'] == 1:
        return True
    if move_to_check['does_measurement'] and move_to_check['type'] != "EP":
        if get_move_value(self,move,Player)>0:
            return True 
    return False
#
def select_favorable_moves(self, moves_list, Player):
    """ 
    Carries out move filtering to carry out quiscence search.
    Move filtering is based on the isfavorable move that looks for capture moves.
    """
    favorable_moves = [move for move in moves_list if self.is_favorable_move(move, Player)]
    return favorable_moves


#
QuantumChessGame.print_board_and_probabilities = print_board_and_probabilities
QuantumChessGame.print_board = print_board
QuantumChessGame.get_board_score = get_board_score
QuantumChessGame.get_board_score_PB = get_board_score_PB
QuantumChessGame.get_fen_code = get_fen_code
QuantumChessGame.get_classical_moves = get_classical_moves
QuantumChessGame.get_quantum_moves = get_quantum_moves
QuantumChessGame.get_compacted_moves = get_compacted_moves
QuantumChessGame.get_truncated_moves = get_truncated_moves
QuantumChessGame.remove_symmetric_splits = remove_symmetric_splits
QuantumChessGame.is_game_over = is_game_over
QuantumChessGame.order_moves = order_moves
QuantumChessGame.order_moves_N = order_moves_N
QuantumChessGame.get_move_value = get_move_value
QuantumChessGame.str_to_square_number = str_to_square_number
QuantumChessGame.is_square_empty = is_square_empty
QuantumChessGame.are_split_targets_empty = are_split_targets_empty
QuantumChessGame.is_spit_move_symmetric = is_spit_move_symmetric
QuantumChessGame.separate_split_merge_moves = separate_split_merge_moves
QuantumChessGame.is_favorable_move = is_favorable_move
QuantumChessGame.select_favorable_moves = select_favorable_moves
#
# Minimax Class
#
class MiniMaxAI:
    def __init__(self, game):
        self.game = game
        
    def minimax(self, player, depth, isMax):

        bestmove = ""

        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimax(player, depth - 1, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimax(player, depth - 1, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
            return minval, bestmove

#
    def minimaxN(self, player, depth, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxN(player, depth - 1, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxN(player, depth - 1, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
            return minval, bestmove

    def minimaxNM(self, player, depth, isMax):

        bestmove = ""

        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            #print(f"Moves before ordering: {no_quantum_moves}")
            OrderedMoves = self.game.order_moves_N(no_quantum_moves, player, isMax)
            print(f"Moved after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxNM(player, depth - 1, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            #print(f"Moves before ordering: {no_quantum_moves}")
            OrderedMoves = self.game.order_moves_N(no_quantum_moves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxNM(player, depth - 1, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
            return minval, bestmove

#
    def minimaxAB(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxAB(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxAB(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
            
#
    def minimaxABN(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()

        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxABN(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            for move in no_quantum_moves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxABN(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
#
    def minimaxABNM(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()

        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            #print(f"Moves before ordering: {no_quantum_moves}")
            OrderedMoves = self.game.order_moves_N(no_quantum_moves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxABNM(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            #print(f"Moves before ordering: {no_quantum_moves}")
            OrderedMoves = self.game.order_moves_N(no_quantum_moves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxABNM(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
#
    def minimaxABM(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            no_quantum_moves = self.game.get_classical_moves()
            #print(f"Moves before ordering: {no_quantum_moves}")
            OrderedMoves = self.game.order_moves(no_quantum_moves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxABM(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            no_quantum_moves = self.game.get_classical_moves()
            #print(f"Moves before ordering: {no_quantum_moves}")
            OrderedMoves = self.game.order_moves(no_quantum_moves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.minimaxABM(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove

# MinimaxAI Quantum Moves
#
class QuantumMiniMaxAI:
    def __init__(self, game):
        self.game = game
        
    def QminimaxAB(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            quantumMoves = self.game.get_quantum_moves()
            for move in quantumMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"Q-{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QminimaxAB(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            quantumMoves = self.game.get_quantum_moves()
            for move in quantumMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"C-{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QminimaxAB(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove

    def QminimaxABN(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            quantumMoves = self.game.get_quantum_moves()
            for move in quantumMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"Q-{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QminimaxABN(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            quantumMoves = self.game.get_quantum_moves()
            for move in quantumMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"C-{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QminimaxABN(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove

    def QminimaxABNM(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            quantumMoves = self.game.get_quantum_moves()
            #print(f"Moves before ordering: {quantumMoves}")
            OrderedMoves = self.game.order_moves_N(quantumMoves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"Q-{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QminimaxABNM(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            quantumMoves = self.game.get_quantum_moves()
            #print(f"Moves before ordering: {quantumMoves}")
            OrderedMoves = self.game.order_moves_N(quantumMoves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"C-{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QminimaxABNM(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
    #
    def QCminimaxAB(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            allMoves = self.game.get_compacted_moves()
            for move in allMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxAB(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            allMoves = self.game.get_compacted_moves()
            for move in allMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxAB(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove


    def QCminimaxABN(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            allMoves = self.game.get_compacted_moves()
            for move in allMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABN(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            allMoves = self.game.get_compacted_moves()
            for move in allMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABN(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove


    def QCminimaxABNM(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            allMoves = self.game.get_compacted_moves()
            #print(f"Moves before ordering: {allMoves}")
            OrderedMoves = self.game.order_moves_N(allMoves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABNM(player, depth-1, alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            allMoves = self.game.get_compacted_moves()
            #print(f"Moves before ordering: {allMoves}")
            OrderedMoves = self.game.order_moves_N(allMoves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")

            for move in OrderedMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABNM(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
#
    def QCminimaxABM(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            allMoves = self.game.get_compacted_moves()
            #print(f"Moves before ordering: {allMoves}")
            OrderedMoves = self.game.order_moves(allMoves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")
            for move in OrderedMoves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABM(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            allMoves = self.game.get_compacted_moves()
            #print(f"Moves before ordering: {allMoves}")
            OrderedMoves = self.game.order_moves(allMoves, player, isMax)
            #print(f"Moves after ordering: {OrderedMoves}")

            for move in OrderedMoves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABM(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
    #@jit
    def QCminimaxABNM_optm(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""

        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        if isMax:
            maxval = -40000
            truncated_moves = self.game.get_truncated_moves(player, isMax)
            for move in truncated_moves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} inside minimax")
                    continue
                elif move_code == 6: #Too many quantum split moves
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABNM_optm(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove  
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            truncated_moves = self.game.get_truncated_moves(player, isMax)
            for move in truncated_moves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    print(f"{move} failed with code {move_code} inside Minimax")
                    continue
                elif move_code == 6: #Too many quantum split moves
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABNM_optm(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove
#
    def QCminimaxABM_optm(self, player, depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or depth == 0:
            score = self.game.get_board_score(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        keep_ratio=1
        if isMax:
            maxval = -40000
            C_moves = self.game.get_classical_moves()
            keepC=int(keep_ratio*len(C_moves))
            O_C_moves = self.game.order_moves(C_moves, player, isMax)[0:5] #ordering via simple scoring
            Q_moves = self.game.get_quantum_moves()
            Q_c_moves = self.game.remove_symmetric_splits(Q_moves)
            keepQ=int(keep_ratio*len(Q_moves))
            O_Q_moves = self.game.order_moves(Q_c_moves, player, isMax)[0:5] #ordering via simple scoring
            O_moves = O_C_moves + O_Q_moves
            #print(f"Moves before ordering: {C_moves+Q_moves}")
            #print(f"Moves after ordering: {O_moves}")
            for move in O_moves:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                elif move_code == 6: #Too many quantum split moves
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABM_optm(player, depth-1,alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                           
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            C_moves = self.game.get_classical_moves()
            keepC=int(keep_ratio*len(C_moves))
            O_C_moves = self.game.order_moves(C_moves, player, isMax)[0:5] #ordering via simple scoring
            Q_moves = self.game.get_quantum_moves()
            Q_c_moves = self.game.remove_symmetric_splits(Q_moves)
            keepQ=int(keep_ratio*len(Q_moves))
            O_Q_moves = self.game.order_moves(Q_c_moves, player, isMax)[0:5] #ordering via simple scoring
            O_moves = O_C_moves + O_Q_moves
            #print(f"Moves before ordering: {C_moves+Q_moves}")
            #print(f"Moves after ordering: {O_moves}")
            for move in O_moves:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                elif move_code == 6: #Too many quantum split moves
                    print(f"{move} failed with code {move_code} at depth: {depth}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.QCminimaxABM_optm(player, depth-1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove

    def timed_minimax(self, player, depth, alpha, beta, isMax, start_time, move_time, heuristic=1, quiescence=False, move_order=True):
        """ Not Fully Implemented yet. Can be an envelope for all Minimax versions and timed moves"""
        heuristic_dict = dict({0: [get_board_score, "piece value heuristic"], 
                               1: [get_board_score_PB, "position based heuristic"]}) 

        if time.time() - start_time > move_time:
            return None, None
        else:
            bestmove = ""
            
            fen=self.game.get_fen_code()
            #print(f'fen: {fen}')
            if 'k' not in fen or 'K' not in fen or depth == 0:
                #score = self.game.get_board_score(player)
                score = heuristic_dict[heuristic][0](player)
                return score, bestmove
            elif (self.game.get_legal_moves() == []): #Draw
                return 0, bestmove
            
            if isMax:
                maxval = -40000
                allMoves = self.game.get_compacted_moves()
                if move_order:
                    OrderedMoves = self.game.order_moves(allMoves, player, isMax)
                    allMoves = OrderedMoves
                for move in allMoves:
                    #print(f"working on move {move} in Max for {player}")
                    gamedata, move_code = self.game.do_move(move)
                    if move_code == 0: #FAIL
                        #print(f"{move} failed with code {move_code}")
                        continue
                    else:              #SUCCESS
                        eval, _ = self.timed_minimax(player, depth-1, alpha, beta, False, start_time, move_time, heuristic=1, quiescence=False, move_order=True)
                        if eval > maxval:
                            maxval = eval
                            bestmove = move

                        self.game.undo_move()
                            
                        if maxval >= beta:
                            #print('pruning a branch in Max')
                            return maxval, bestmove
                        
                        if maxval > alpha:
                            alpha = maxval
                            #print(f'Redefined alpha = {alpha}')
                            
                return maxval, bestmove
            else:
                minval = 40000

                allMoves = self.game.get_compacted_moves()
                if move_order:
                    OrderedMoves = self.game.order_moves(allMoves, player, isMax)
                    allMoves = OrderedMoves
                for move in allMoves:
                    #Make the move and check move code for success
                    gamedata, move_code = self.game.do_move(move)
                    if move_code == 0: #FAIL
                        #print(f"{move} failed with code {move_code}")
                        continue
                    else:              #SUCCESS
                        eval, _ = self.timed_minimax(player, depth-1, alpha, beta, True, start_time, move_time, heuristic=1, quiescence=False, move_order=True)

                        if eval < minval:
                            minval = eval
                            bestmove = move
                            
                        self.game.undo_move()
                            
                        if minval <= alpha:
                            #print('pruning a branch in Min')
                            return minval, bestmove
                            
                        if minval < beta:
                            beta = minval
                            #print(f'Redefined beta = {beta}')
                            
                return minval, bestmove
#Quiescence Search with AB Pruning
    def Quiescence_SearchAB(self, player, max_depth, current_depth, alpha, beta, isMax):
        
        bestmove = ""
        
        fen=self.game.get_fen_code()
        #print(f'fen: {fen}')
        if 'k' not in fen or 'K' not in fen or current_depth == 0:
            score = self.game.get_board_score_PB(player)
            return score, bestmove
        elif (self.game.get_legal_moves() == []): #Draw
            return 0, bestmove
        
        allMoves = self.game.get_compacted_moves()
        OrderedMoves = self.game.order_moves_N(allMoves, player, isMax)
        if max_depth-current_depth > 3: 
            moves_to_explore = self.game.select_favorable_moves(OrderedMoves, player)
        else:
            moves_to_explore = OrderedMoves
        
        if isMax:
            maxval = -40000
            for move in moves_to_explore:
                #print(f"working on move {move} in Max for {player}")
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.Quiescence_SearchAB(player, max_depth, current_depth - 1, alpha, beta, False)
                    if eval > maxval:
                        maxval = eval
                        bestmove = move

                    self.game.undo_move()
                        
                    if maxval >= beta:
                        #print('pruning a branch in Max')
                        return maxval, bestmove
                        
                    if maxval > alpha:
                        alpha = maxval
                        #print(f'Redefined alpha = {alpha}')
                           
            return maxval, bestmove
        else:
            minval = 40000
            for move in moves_to_explore:
                #print(f"working on move {move} in Min for {player}")
                #Make the move and check move code for success
                gamedata, move_code = self.game.do_move(move)
                if move_code == 0: #FAIL
                    #print(f"{move} failed with code {move_code}")
                    continue
                else:              #SUCCESS
                    eval, _ = self.Quiescence_SearchAB(player, max_depth, current_depth - 1, alpha, beta, True)
                    if eval < minval:
                        minval = eval
                        bestmove = move
                        
                    self.game.undo_move()
                         
                    if minval <= alpha:
                        #print('pruning a branch in Min')
                        return minval, bestmove
                        
                    if minval < beta:
                        beta = minval
                        #print(f'Redefined beta = {beta}')
                         
            return minval, bestmove