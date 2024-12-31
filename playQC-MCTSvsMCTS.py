from QuantumChessGame import QuantumChessGame, MoveCode, format_moves
#from MinimaxAlphaBeta import MiniMaxAI, QuantumMiniMaxAI, order_moves, str_to_square_number
#from QuantumChessWithAI import QuantumChessGameWithAI
from QuantumChessUtils import *
from MCTS import MCTS_Node
from ChessPuzzles import chess_puzzles
import numpy as np
import time

_start_time = time.time()
seed=np.random.randint(1000)
#print(seed)
np.random.seed(seed)

def tic1():
    global _start_time1
    _start_time1 = time.time()
    return
    
def tic2():
    global _start_time2
    _start_time2 = time.time()
    return

def tic3():
    global _start_time3
    _start_time3 = time.time()
    return
    
def toc1():
    t_diff1 = time.time() - _start_time1
    t_sec1 = int(np.floor(t_diff1))
    t_frac1 = t_diff1 - t_sec1
    (t_min1, t_sec1) = divmod(t_sec1,60)
    (t_hour1,t_min1) = divmod(t_min1,60)
    print('Time elapsed for the current method: {} hour:{} min:{} sec and frac sec {}'.format(t_hour1,t_min1,t_sec1,t_frac1))

def toc2():
    t_diff2 = time.time() - _start_time2
    t_sec2 = int(np.floor(t_diff2))
    t_frac2 = t_diff2 - t_sec2
    (t_min2, t_sec2) = divmod(t_sec2,60)
    (t_hour2,t_min2) = divmod(t_min2,60)
    print('Time taken for the puzzle: {} hour:{} min:{} sec and frac sec {}'.format(t_hour2,t_min2,t_sec2,t_frac2))

def toc3():
    t_diff3 = time.time() - _start_time3
    t_sec3 = int(np.floor(t_diff3))
    t_frac3 = t_diff3 - t_sec3
    (t_min3, t_sec3) = divmod(t_sec3,60)
    (t_hour3,t_min3) = divmod(t_min3,60)
    print('Time taken for move optimization: {} hour:{} min:{} sec and frac sec {}'.format(t_hour3,t_min3,t_sec3,t_frac3))


class AI_QC_Game(QuantumChessGameWithAI):

    def __init__(self):
        super().__init__()
        self.player_dict = dict({' b ' : "Black", ' w ' : "White"})
        self.AI_Methods_dict = dict({
            1:  ["minimax",'AI.minimax'],
            2:  ["minimax-position-score",'AI.minimaxN'],
            3:  ["minimax-move-ordering",'AI.minimaxNM'],
            4:  ["minimax-alpha-beta-pruning",'AI.minimaxAB'],
            5:  ["minimax-AB-position-score",'AI.minimaxABN'],
            6:  ["minimax-AB-position-score-move-ordering",'AI.minimaxABNM'],
            7:  ["minimax-AB-old_score-move-ordering",'AI.minimaxABM'],
            8:  ["minimax-AB-qc-all-moves",'QAI.QCminimaxAB'],
            9:  ["minimax-AB-qc-all-position-score",'QAI.QCminimaxABN'],
            10: ["minimax-AB-qc-all-move-ordering",'QAI.QCminimaxABNM'],
            11: ["minimax-AB-qc-all-oldscore-moveordering",'QAI.QCminimaxABM'],
            12: ["minimax-AB-qc-all-newscore-moveordering-top-q25c10",'QAI.QCminimaxABNM_optm'],
            13: ["minimax-AB-qc-all-oldscore-moveordering--top-60-percent",'QAI.QCminimaxABM_optm'],
            14: ["minimax-AB-quantum-maximizer",'QAI.QminimaxAB'], #quantum only methods not very useful.
            15: ["minimax-AB-quantum-maximizer-position_score",'QAI.QminimaxABN'],
            16: ["minimax-AB-qM-move-ordering",'QAI.QminimaxABNM'],
            17: ["MCTS",'QAI.MCTS']
            })
        
    def play_random_move(self,player):
        """
        Play random moves from a set of classical moves for a given player.
        """
        valid_classical_moves = self.get_classical_moves()
        if (valid_classical_moves!=[]):
            cmove = np.random.choice(valid_classical_moves)
            gamedata, move_code = self.do_move(cmove)
            print(f"{player}'s move: {cmove}, resulted in movecode: {MoveCode[move_code]}")
            if self.is_game_over():
                print("Random move did not succeed",MoveCode[move_code])
            #self.print_board()
        return gamedata, move_code
#
    def play_random_quantum_move(self,player):
        """
        Play random moves from a set of classical moves for a given player.
        """
        valid_quantum_moves = self.get_quantum_moves()
        if (valid_quantum_moves!=[]):
            qmove = np.random.choice(valid_quantum_moves)
            gamedata, move_code = self.do_move(qmove)
            print(f"{player}'s move: {qmove}, resulted in movecode: {MoveCode[move_code]}")
            if self.is_game_over():
                print("Random move did not succeed",self.MoveCode[move_code])
            #self.print_board()
        return gamedata, move_code
#
    def check_if_duplicates(self,movehistory):
        ''' Check if given list contains any duplicates '''
        if len(movehistory) == len(set(movehistory)):
            return False
        else:
            return True
#
    def AI_Play_MM(self,game,Player,depth,method):
        """
        Carry out the AI Player's move with timing for each move using Minimax
        """
        AI = MiniMaxAI(game)
        QAI = QuantumMiniMaxAI(game)
        possible_moves=game.get_truncated_moves(Player, True)
        c_moves=game.get_classical_moves()
        s_moves, m_moves = game.separate_split_merge_moves()
        print(f"Optimizing {Player}'s Move from {len(possible_moves)} possible moves.")
        
        tic3()
        if method == 1:
            value, move_minimax = AI.minimax(Player, depth, True)

        elif method == 2:
            value, move_minimax = AI.minimaxN(Player, depth, True)

        elif method == 3:
            value, move_minimax = AI.minimaxNM(Player, depth, True)

        elif method == 4:
            value, move_minimax = AI.minimaxAB(Player, depth, float('-inf'), float('inf'), True)

        elif method == 5:
            value, move_minimax = AI.minimaxABN(Player, depth, float('-inf'), float('inf'), True)

        elif method == 6:
            value, move_minimax = AI.minimaxABNM(Player, depth, float('-inf'), float('inf'), True)
         
        elif method == 7:
            value, move_minimax = AI.minimaxABM(Player, depth, float('-inf'), float('inf'), True)
            
        elif method == 8:
            value, move_minimax = QAI.QCminimaxAB(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 9:
            value, move_minimax = QAI.QCminimaxABN(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 10:
            value, move_minimax = QAI.QCminimaxABNM(Player, depth, float('-inf'), float('inf'), True)
            
        elif method == 11:
            value, move_minimax = QAI.QCminimaxABM(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 12:
            value, move_minimax = QAI.QCminimaxABNM_optm(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 13:
            value, move_minimax = QAI.QCminimaxABM_optm(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 14:
            value, move_minimax = QAI.QminimaxAB(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 15:
            value, move_minimax = QAI.QminimaxABN(Player, depth, float('-inf'), float('inf'), True)
        
        elif method == 16:
            value, move_minimax = QAI.QminimaxABNM(Player, depth, float('-inf'), float('inf'), True)

        elif method == 17:
            print("Uses MCTS. Please check code")
        toc3()
        if move_minimax == "":
            print("Minimax failed to reach optimum move")
            print(f"{Player} will choose a random classical move")
            game_data, move_code = AIQCGame.play_random_move(Player)
        else:
            print("Minimax found optimum move")
            print(f'value ={value}, move ={move_minimax}')
            total_idx=possible_moves.index(move_minimax)
            print(f'This is {total_idx} out of {len(possible_moves)} total ordered moves.')
            if move_minimax in c_moves:
                c_idx = c_moves.index(move_minimax)
                print(f'This is {c_idx} out of {len(c_moves)} classical moves.')
            elif move_minimax in s_moves:
                s_idx = s_moves.index(move_minimax)
                print(f'This is {s_idx} out of {len(s_moves)} split moves.')
            elif move_minimax in m_moves:
                m_idx = m_moves.index(move_minimax)
                print(f'This is {m_idx} out of {len(m_moves)} merge moves.')
            
            print(f"Board before attempting {Player}'s move:")
            AIQCGame.print_board_and_probabilities()
            game_data, move_code = AIQCGame.do_move(move_minimax)
            if move_code == 6:
                print(f"Too many quantum split moves")
            elif move_code == 0:
                print(f"Move {move_minimax} failed for some reason. Trying another move!")
            else:
                print(f"Executed move : {move_minimax} and the move code is {MoveCode[move_code]}")
        #game_data = AIQCGame.get_game_data() # testing the effect of this
        print(f"Board after {Player}'s move:")
        AIQCGame.print_board_and_probabilities()
        
        return move_code
#
    def print_move_history(self):
        """
        Formats and prints the move history so far.
        """
        move_history = self.get_history()
        move_history = format_moves(move_history)
        print("Move history so far:")
        print(" ".join(move_history[:]))
        return move_history
#   
    def AI_Play_MCTS(self,game,Player):
        """
        Carry out the AI Player's move with timing for each move using MCTS
        """
        root = MCTS_Node(game)
        bestMove_MCTS = root.best_action(Player)
        game_data, move_code = AIQCGame.do_move(bestMove_MCTS)

        print(f"Board after {Player}'s move:")
        AIQCGame.print_board_and_probabilities()
        return move_code

    
#AIQCGame = AI_QC_Game(max_split_moves=[0,1])
AIQCGame = AI_QC_Game()
for sample in range(1):
    print(f"sample no.: {sample}")
    for method in range(17,18):
        puzzle_count = 0
        print(f"Using Method {method}: {AIQCGame.AI_Methods_dict[method]}")
        tic1()
        for puzzle in chess_puzzles[0:1]:
            tic2()
            puzzle_count += 1
            fen = puzzle['FEN']
            
            AIQCGame.new_game_with_AI(initial_state = fen, max_split_moves=[0,1])
            print(f"Started New Game")
            game_data = AIQCGame.get_game_data()
            print("Welcome to Quantum Chess")
            print("Use CTRL+C (or CMD+C) to quit")
            print(f"working on puzzle no. {puzzle_count} for {fen}")

            if ' b ' in fen :
                starting_player = AIQCGame.player_dict[' b ']
                opponent = AIQCGame.player_dict[' w ']
            else:
                starting_player = AIQCGame.player_dict[' w ']
                opponent = AIQCGame.player_dict[' b ']

            print(f"{starting_player}'s turn:")
            print("Starting Board:")
            AIQCGame.print_board()
            if method<16:
                max_depth = 4
                print(f"Playing AI vs AI with tree-search depth = {max_depth}")
            else:
                print(f"Playing AI vs AI with MCTS algorithm")
            current_player = starting_player #set current player as starting_player
            player = 0
            while True:
                # print the move history and Game Score
                move_history=AIQCGame.print_move_history()
                if AIQCGame.check_if_duplicates(move_history) and len(move_history)>100:
                    print("Duplicates are appearing in the move history abandoning this puzzle")
                    break
                #Starting Player's turn being optimized
                value = 0
                if method<16:
                    depth = max_depth
                else:
                    depth = None
                #carry out AI move:
                move_code = AIQCGame.AI_Play_MCTS(AIQCGame,current_player)
                #check if game over
                if AIQCGame.is_game_over():
                    print(f"Game Over!:{MoveCode[move_code]}")
                    break
                elif move_code == 6:
                    print("Too many quatum split moves. Abondoning puzzle.")
                    break   #if move failed the current player tries again.
                elif move_code == 0:
                    print("Move failed trying another move.")
                    continue   #move failed. continue with the same player
                else:
                    player = (player + 1) % 2 #move successful. Switch player
                    if player == 0:
                        current_player = starting_player  
                    else: current_player = opponent #if move is a success opponent's turn
            else:
                print("Game Over")
            # print the move history
            AIQCGame.print_move_history()
            print(f"Current Game Score: {AIQCGame.get_board_score_PB(starting_player)}")
            toc2()
            #c = input("Press any key to continue:")
        toc1()

