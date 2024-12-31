from QuantumChess import QuantumChessGame, MoveCode, format_moves
from QuantumChess import is_quantum_move, is_split_move
from QuantumChess import apply_move_corrections, square_number_to_str
from QuantumChess import number_to_piece, piece_to_number, MoveType
import functools
import time
#from numba import njit
#
#
# Create a new class that inherits the QuantumChessGame class for interactions with AI
#
class QuantumChessGameWithAI(QuantumChessGame):
    def __init__(self):
        super().__init__()
        
    def new_game_with_AI(self,initial_state="",force_turn = True, force_win = True, rest_url = "", max_split_moves = [10,10]):
        self.initial_state_fen = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves " if initial_state == "" else initial_state
        self.new_game(initial_state=self.initial_state_fen, force_turn = True, force_win = True, rest_url = "", max_split_moves = [10,10])

    def __copy__(self):
        copied_game = QuantumChessGameWithAI()
        copied_game.new_game_with_AI(initial_state=self.initial_state_fen, force_turn = True, force_win = True, rest_url = "", max_split_moves = [10,10])
        move_history = self.get_history()
        move_history = format_moves(move_history)
        for move in move_history:
            _ , move_code = copied_game.do_move(move)
            if move_code == 0:
                print("Error in copying: move failed")
        return copied_game

    def str_to_square_number(self,str) -> int:
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
        target1_square = self.str_to_square_number(target1_str)
        target2_square = self.str_to_square_number(target2_str)
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
            
        piece_value = { "P":   100,
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
            pre_move_score = self.get_board_score_PB(Player)
            gamedata, move_code = self.do_move(move)
            if move_code == 0 or move_code == 6:
                print(f'Encountered issue in getting score for move: {move} while ordering. Move Code: 0')
                print('Check if the move is possible on the board below:')
                self.print_board_and_probabilities()
                return -100000
            elif move_code == 6:
                print(f'Encountered issue in getting score for move: {move} while ordering. Move Code: 6')
                self.print_board_and_probabilities()
                self.undo_move()
                return -100000
            else:
                post_move_score = self.get_board_score_PB(Player)
                self.undo_move()
                move_score = post_move_score - pre_move_score
                #if "^" in move and move_score !=0:
                #    print(f"{Player}'s Score for {move} is {move_score}")
                return move_score
    #
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

        moves_in_order = sorted(moves_list, key = lambda x: self.get_move_value(x, order_moves_for), reverse=True)
        return moves_in_order
    #
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
                print(f'Encountered issue in getting score for move: {move} while ordering. Move Code: {move_code}')
                self.print_board_and_probabilities()
                move_history = format_moves(move_history)
                print("Move history so far:")
                print(" ".join(move_history[:]))
                print(f'Current Player: {self.current_player}')
                print(f'pre_move_score: {pre_move_score}')
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

    def get_board_score(self, Player):
        """Returns the board score using piece values and probabilities for the specified maximizer: White or Black"""
        piece_value = { "P":   100,
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
    def get_all_moves(self):
        """Get all the legal quantum moves"""
        all_moves = [x for x in format_moves(self.get_legal_moves())]
        return all_moves
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
        compacted_moves = self.get_compacted_moves()
        keep_ratio=0.8
        truncated_moves = self.order_moves_N(compacted_moves, player, isMax)
        max_idx = int(keep_ratio*len(truncated_moves))
        #print(f'truncated moves: {truncated_moves}')
        return truncated_moves[0:max_idx]
    #
    def is_game_over_1(self):
        """Returns True if the game has a winner or results in a draw and False if game needs to contiue"""
        move_code = self.movecode.value
        if move_code in [2, 3, 4, 5]:
            return True
        else:
            return False
    #
    def is_game_over_2(self): 
        """
        Returns True if the game has a winner or results in a draw and False if game needs to contiue.
        This routine works if the move_code is not available.
        """
        fen=self.get_fen_code()
        if 'k' not in fen or 'K' not in fen:
            return  True
        elif (self.get_legal_moves() == []): #Draw
            return True
        else:
            return False
    #
    def is_game_over(self):
        return self.is_game_over_1()
    #
    #
    def game_result(self,AI_player):
        move_code = self.movecode.value
        """Returns 1 if player wins. -1 if opponent wins and 0 is a draw"""
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
        if move_code in [4,5]:
            return 0
    #
    def get_game_state_copy(self, fen, **kwargs):
        history = self.get_history()
        game = QuantumChessGame()
        game.new_game(initial_state = fen, **kwargs)

        
        return new_game()
    #
# QuantumChessGame.print_board_and_probabilities = print_board_and_probabilities
# QuantumChessGame.print_board = print_board
# QuantumChessGame.get_board_score = get_board_score
# QuantumChessGame.get_board_score_PB = get_board_score_PB
# QuantumChessGame.get_fen_code = get_fen_code
# QuantumChessGame.get_classical_moves = get_classical_moves
# QuantumChessGame.get_quantum_moves = get_quantum_moves
# QuantumChessGame.get_all_moves = get_all_moves
# QuantumChessGame.get_compacted_moves = get_compacted_moves
# QuantumChessGame.get_truncated_moves = get_truncated_moves
# QuantumChessGame.remove_symmetric_splits = remove_symmetric_splits
# QuantumChessGame.is_game_over = is_game_over
# QuantumChessGame.is_game_over_1 = is_game_over_1 #with move code provided: Minimax needs this
# QuantumChessGame.is_game_over_2 = is_game_over_2 #without move code provided: MCTS needs this
# QuantumChessGame.order_moves = order_moves
# QuantumChessGame.order_moves_N = order_moves_N
# QuantumChessGame.get_move_value = get_move_value
# QuantumChessGame.str_to_square_number = str_to_square_number
# QuantumChessGame.is_square_empty = is_square_empty
# QuantumChessGame.are_split_targets_empty = are_split_targets_empty
# QuantumChessGame.is_spit_move_symmetric = is_spit_move_symmetric
# QuantumChessGame.separate_split_merge_moves = separate_split_merge_moves
# QuantumChessGame.game_result = game_result