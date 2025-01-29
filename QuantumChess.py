from sys import platform
from ctypes import (c_char, c_char_p, c_uint8, c_uint64, c_int,
                    c_float, c_double, c_long, c_size_t, c_bool,
                    Structure, CDLL, byref, sizeof, POINTER,
                    cast)

# GameData struct
class GameData(Structure):
    _fields_ = [
    ("pieces", c_char * 64),
    ("probabilities", c_float * 64),
    ("ply", c_int),
    ("fifty_count", c_int),
    ("castle_flags", c_int),
    ("ep_square", c_int),
    ]

class Move(Structure):
    _fields_ = [
    ("piece", c_uint8),
    ("square1", c_uint8),
    ("square2", c_uint8),
    ("square3", c_uint8),
    ("type", c_uint8),
    ("variant", c_uint8),
    ("does_measurement", c_bool),
    ("measurement_outcome", c_uint8),
    ("promotion_piece", c_uint8)
    ]

class SampleData(Structure):
    _fields_ = [
    ("bitboard", c_uint64),
    ("probability", c_double)
    ]

MoveCode = dict({0 : "FAIL", 1 : "SUCCESS", 2 : "WHITE_WIN",
                 3 : "BLACK_WIN", 4 : "MUTUAL_WIN", 5 : "DRAW",
                 6 : "TOO_MANY_QUANTUM_MOVES"})

MoveType = dict({
            0 : "NULL_TYPE",
            1 : "UNSPECIFIED_STANDARD",
            2 : "JUMP",
            3 : "SLIDE",
            4 : "SPLIT_JUMP",
            5 : "SPLIT_SLIDE",
            6 : "MERGE_JUMP",
            7 : "MERGE_SLIDE",
            8 : "OBSOLETE_PAWN_STEP", # Use JUMP instead
            9 : "OBSOLETE_iPAWN_TWO_STEP", # Use SLIDE instead
            10 : "PAWN_CAPTURE",
            11 : "PAWN_EP",
            12 : "KS_CASTLE",
            13 : "QS_CASTLE",
        })

MoveVariant = dict({
             0: "UNSPECIFIED",
             1: "BASIC",
             2: "EXCLUDED",
             3: "CAPTURE"
            })

number_to_piece = dict({ 1 : "P", 9 : "p",
                         2 : "N", 10 : "n",
                         3 : "B", 11 : "b",
                         4 : "R", 12 : "r",
                         5 : "Q", 13 : "q",
                         6 : "K", 14 : "k"})

piece_to_number = dict({ "P" : 1, "p" : 9,
                      "N" : 2, "n" : 10,
                      "B" : 3, "b" : 11,
                      "R" : 4, "r" : 12,
                      "Q" : 5, "q" : 13,
                      "K" : 6, "k" : 14 })

def is_quantum_move(move_str):
    return (move_str.find("^") != -1)

def is_split_move(move_str):
    if is_quantum_move(move_str):
        if len(move_str.split("^")[1]) == 4:
            return True
    return False

def square_number_to_str(square_number) -> str:
    column = int(square_number % 8)
    row = int(square_number / 8)
    return chr(column+97) + str(row+1)

def chess_notation_to_indices(move: str):
    
    def square_to_index(square: str) -> int:
        file = ord(square[0]) - ord('a')  # Convert 'a'-'h' to 0-7
        rank = int(square[1]) - 1         # Convert '1'-'8' to 0-7
        return rank * 8 + file            # Convert (rank, file) to 0-63 index

    source = square_to_index(move[:2])
    destination = square_to_index(move[2:])

    return source, destination

# Example usage:
move = "e2e4"
indices = chess_notation_to_indices(move)
print(indices)  # Output: (12, 28)


def format_move(move) -> str:
    if( move["type"] == 4 or move["type"] == 5 ):
        movestr = square_number_to_str(move["square1"]) + "^" \
                + square_number_to_str(move["square2"]) \
                + square_number_to_str(move["square3"])
    # Merge move
    elif( move["type"] == 6 or move["type"] == 7 ):
        movestr = square_number_to_str(move["square1"]) \
                + square_number_to_str(move["square2"]) + "^" \
                + square_number_to_str(move["square3"])
    # Standard
    else:
        movestr = square_number_to_str(move["square1"]) + square_number_to_str(move["square2"])

    # Was this a promotion move?
    if( move["promotion_piece"] != 0 ):
        movestr += str(number_to_piece[move["promotion_piece"]])

    return movestr

def format_moves(moves) -> str:
    allmoves = []
    for move in moves:
        movestr = format_move(move)
        allmoves.append(movestr)

    return allmoves

def apply_move_corrections(move_str):
    # converts a move like e2d1.m1q to e2d1q.m1
    pattern = ".m[01][qQrRbBnN]$"
    if re.search(pattern, move_str):
        move, measurement = move_str.split(".")
        move_str = "{}{}.{}".format(move, measurement[-1], measurement[:-1])
    return move_str

class QuantumChessGame:
    def __init__(self):
        """
            Loads the QuantumChessAPI library
        """
        shared_lib_path = ""

        # Linux
        if platform.startswith("linux"):
            shared_lib_path = "./QuantumChessLibraries/QuantumChessAPI.so"
        # Windows
        if platform.startswith('win32'):
            shared_lib_path = "./QuantumChessLibraries/QuantumChessAPI.dll"
        # Mac OSX
        if platform.startswith('darwin'):
            shared_lib_path = "./QuantumChessLibraries/QuantumChessAPI.dylib"

        if( shared_lib_path == "" ):
            print("This platform is not supported")
            exit(0)

        try:
            self.QChess_lib = CDLL(shared_lib_path)
        except Exception as e:
            print("Could not load the Quantum Chess library")
            print(e)
            exit(0)

        # Set up all the API signatures
        #QUANTUM_CHESS_API GameVariant* new_game(const char * position, bool force_turn, bool force_win, const char * rest_url);
        self.QChess_lib.new_game.argtypes = [c_char_p, c_bool, c_bool, c_char_p]
        self.QChess_lib.new_game.restype = POINTER(c_int)

        #QUANTUM_CHESS_API long delete_game(GameVariant* game);
        self.QChess_lib.delete_game.argtypes = [POINTER(c_int)]
        self.QChess_lib.delete_game.restype = c_long

        #QUANTUM_CHESS_API long do_move(GameVariant* game, const char * move, GameData* out_buffer, int* move_code);
        self.QChess_lib.do_move.argtypes = [POINTER(c_int), c_char_p, POINTER(GameData), POINTER(c_int)]
        self.QChess_lib.do_move.restype = c_long

        #QUANTUM_CHESS_API long undo_move(GameVariant* game, GameData* out_data);
        self.QChess_lib.undo_move.argtypes = [POINTER(c_int), POINTER(GameData)]
        self.QChess_lib.undo_move.restype = c_long

        #QUANTUM_CHESS_API long get_game_data(GameVariant* game, GameData* out_buffer);
        self.QChess_lib.get_game_data.argtypes = [POINTER(c_int), POINTER(GameData)]
        self.QChess_lib.get_game_data.restype = c_long

        #QUANTUM_CHESS_API long get_history(GameVariant* game, QC::Move* out_buffer, size_t buffer_size, size_t* out_size);
        self.QChess_lib.get_history.argtypes = [POINTER(c_int), POINTER(Move), c_size_t, POINTER(c_size_t)]
        self.QChess_lib.get_history.restype = c_long

        #QUANTUM_CHESS_API long get_legal_moves(GameVariant* game, QC::Move* out_buffer, size_t buffer_size, size_t* out_size);
        self.QChess_lib.get_legal_moves.argtypes = [POINTER(c_int), POINTER(Move), c_size_t, POINTER(c_size_t)]
        self.QChess_lib.get_legal_moves.restype = c_long

        #QUANTUM_CHESS_API long get_pairwise_bell_measures(GameVariant* game, int square1, int square2, float* out_buffer, int buffer_size);
        self.QChess_lib.get_pairwise_bell_measures.argtypes = [POINTER(c_int), c_int, c_int, POINTER(c_float), c_int]
        self.QChess_lib.get_pairwise_bell_measures.restype = c_long

        #QUANTUM_CHESS_API long get_samples(GameVariant* game, SampleData* out_buffer, size_t num_samples, size_t* out_size);
        self.QChess_lib.get_samples.argtypes = [POINTER(c_int), POINTER(SampleData), c_int, POINTER(c_size_t)]
        self.QChess_lib.get_samples.restype = c_long

        # Current gamedata
        self.gamedata = GameData()
        # Result of the last move
        self.movecode = c_int(0)
        # Current game pointer
        self.game_pointer = None

    def new_game(self, initial_state="", force_turn = True, force_win = True, rest_url = "", max_split_moves = [10,10]):
        initial_state_fen = "position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 moves " if initial_state == "" else initial_state

        self.max_split_moves = max_split_moves
        if type(max_split_moves) == int or type(max_split_moves) == float:
            self.max_split_moves = [max_split_moves, max_split_moves]

        # First index is split moves for white, second for black
        self.current_num_split_moves = [0,0]
        self.current_player = 0 if initial_state_fen.split(" ")[3] == 'w' else 1

        # Release previous game pointer
        if( self.game_pointer != None ):
            self.delete_game()

        # Create a new game pointer
        self.game_pointer = self.QChess_lib.new_game(initial_state_fen.encode('utf-8'), c_bool(force_turn), c_bool(force_win), rest_url.encode('utf-8'));

    def delete_game(self):
        # Destroy previous game pointer
        return self.QChess_lib.delete_game(self.game_pointer);

    def do_move(self, move_str):
        # See if this is an attempt at a split move
        is_split = is_split_move(move_str)
        if is_split:
            if self.current_num_split_moves[self.current_player] >= self.max_split_moves[self.current_player]:
                self.movecode = c_int(6)
                return self.gamedata, self.movecode

        self.QChess_lib.do_move(self.game_pointer, move_str.encode('utf-8'), byref(self.gamedata), byref(self.movecode))

        # If the move was successful and it was a split, increment the counter
        if self.movecode.value != 0 and is_split:
            self.current_num_split_moves[self.current_player] += 1

        self.current_player = self.gamedata.ply % 2
        return self.gamedata, self.movecode.value

    def undo_move(self):
        move_history = self.get_history()
        last_move = None
        if len(move_history) >= 1:
            last_move = move_history[-1]

        result = self.QChess_lib.undo_move(self.game_pointer, byref(self.gamedata))

        # Switch player
        self.current_player = self.gamedata.ply % 2

        # Decrement the split move counter if the last move was a split
        if( last_move != None ):
            if( last_move["type"] == 4 or last_move["type"] == 5):
                self.current_num_split_moves[self.current_player] -= 1

        return result

    def get_game_data(self):
        self.QChess_lib.get_game_data(self.game_pointer, byref(self.gamedata))
        return self.gamedata

    def get_history(self, max_history_size = 512):
        moves = (Move*max_history_size)()
        out_size = c_size_t()
        self.QChess_lib.get_history(self.game_pointer, moves, sizeof(Move)*max_history_size, byref(out_size) )

        allMoves = []
        for move in moves[:out_size.value]:
            thisMove = {}
            for field in move._fields_:
                thisMove[field[0]] = getattr(move, field[0])

            allMoves.append(thisMove)
        return allMoves

    def get_legal_moves(self, max_num_moves = 8192):
        moves = (Move*max_num_moves)()
        out_size = c_size_t()
        self.QChess_lib.get_legal_moves(self.game_pointer, moves, sizeof(Move)*max_num_moves, byref(out_size) )

        allMoves = []
        for move in moves[:out_size.value]:
            thisMove = {}
            for field in move._fields_:
                thisMove[field[0]] = getattr(move, field[0])

            # Move is not valid if we are over the split_move limit
            if( thisMove["type"] == 4 or thisMove["type"] == 5 ):
                if self.current_num_split_moves[self.current_player] >= self.max_split_moves[self.current_player]:
                    continue

            allMoves.append(thisMove)
        return allMoves

    def get_pairwise_bell_measures(self, square1: int, square2: int):
        '''
        Returns a list of 8 floats, each representing the measurement of a Bell operator.
        These bell operators are:
        TODO
        '''
        out_buffer = (c_float*8)()
        self.QChess_lib.get_pairwise_bell_measures(self.game_pointer, square1, square2, cast(out_buffer, POINTER(c_float)), 8 )
        return [i for i in out_buffer]

    def get_samples(self, max_num_samples = 4096):
        """
            Sample the quantum state.
            If max_num_samples is less than the number of states in superposition, this will return
            an exact sampling with the true probabilities of each state.
            If max_num_samples is larger than the number of states in superposition, a sample of size
            max_num_samples is drawn, and the assigned probabilities are simply the counts (i.e. how
            often the sample was drawn).
        """
        samples = (SampleData*max_num_samples)()
        out_size = c_size_t()
        self.QChess_lib.get_samples(self.game_pointer, samples, max_num_samples, byref(out_size))

        all_samples = []
        for sample in samples[:out_size.value]:
            this_sample = {}
            for field in sample._fields_:
                this_sample[field[0]] = getattr(sample, field[0])

            all_samples.append(this_sample)
        return all_samples

    def print_probability_board(self):
        """Renders a ASCII diagram showing the board probabilities."""
        s = ''
        s += ' +' + '-' * 8*9 + '--+\n'
        for y in reversed(range(8)):
            s += str(y + 1) + '| '
            for x in range(8):
                bit = y * 8 + x
                piece = chr(self.gamedata.pieces[bit])
                prob = str(int(100 * self.gamedata.probabilities[bit]))

                if piece != 'x':
                    displaystring = piece + "("+ prob + ") "
                    s += displaystring.center(8)
                else:
                    s += ".".center(8) #'   .   '

                s += ' '
            s += ' |\n'
        s += ' +' + '-' * 8*9 + '--+\n'
        s += '  '
        for x in range(8):
           s += chr(ord('a') + x).center(9)

        print(s)

if( __name__ == "__main__"):
    game = QuantumChessGame()
    game.new_game(max_split_moves=[2,2])#initial_state = "position fen 5qk1/3R4/6pP/6K1/8/8/1B6/8 w - - 0 1 moves", max_split_moves=4)

    print("Welcome to Quantum Chess")
    print("Use CTRL+C (or CMD+C) to quit")
    while True:

        valid_moves = game.get_legal_moves()
        valid_moves = format_moves(valid_moves)
        print(valid_moves[:10])

        # Get input string
        move = input("Enter your next move: ")

        if( move == "undo" ):
            game.undo_move()
            continue

        gamedata, move_code = game.do_move(move)
        print("Resulted in movecode: ", MoveCode[move_code])
        game.print_probability_board()

        print(game.movecode.value)

        move_history = game.get_history()
        move_history = format_moves(move_history)
        print(" ".join(move_history[:3]))

        game_data = game.get_game_data()
        print("Current ply: ", game_data.ply)
        print("Current number of split moves: ", game.current_num_split_moves)

        bell_measures = game.get_pairwise_bell_measures(0, 4)
        print(bell_measures)

        print("Asking for 4 state samples")
        samples = game.get_samples(4)

        for s,sample in enumerate(samples):
            bitstring = '{0:064b}'.format(sample['bitboard'])[::-1]
            board = "".join([chr(game_data.pieces[i]) if bitstring[i] == '1' else 'x' for i in range(64)])
            print("Sample {}: {} with probability {}".format(s, board, sample["probability"]))
            #game_data.pieces