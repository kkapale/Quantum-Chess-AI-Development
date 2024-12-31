from sys import platform
from ctypes import (c_char, c_char_p, c_uint8, c_uint64, c_int,
                    c_float, c_double, c_long, c_size_t, c_bool,
                    Structure, CDLL, byref, sizeof, POINTER,
                    cast)

class AISearchConfig(Structure):
    _fields_ = [
      ("max_search_depth", c_int),
      ("max_search_time", c_int),
      ("capture_move_repetitions", c_int),
      ("use_iterative_deepening", c_bool),
      ("use_transposition_table", c_bool),
      ("search_quantum_moves", c_bool),
    ]

class AISearchResult(Structure):
    _fields_ = [
    ("numPrunings", c_int),
    ("numNodesSearched", c_int),
    ("numEvaluations", c_int),
    ("numQuiescenceNodesSearched", c_int),
    ("maxDepthSearched", c_int),
    ("bestMove", c_char*10),
    ]

class QuantumChessAI:
    def __init__(self):
        """
            Loads the QuantumChessAI library
        """
        shared_lib_path = ""

        # # Linux
        if platform.startswith("linux"):
            shared_lib_path = "./QuantumChessLibraries/AI/QuantumChessAIAPI.so"
        # # Windows
        if platform.startswith('win32'):
            shared_lib_path = "./QuantumChessLibraries/AI/QuantumChessAIAPI.dll"
        # Mac OSX
        if platform.startswith('darwin'):
            shared_lib_path = "./QuantumChessLibraries/AI/QuantumChessAI.bundle/Contents/MacOS/QuantumChessAI"

        try:
            self.QChessAI_lib = CDLL(shared_lib_path)
        except Exception as e:
            print("Could not load the Quantum Chess AI library")
            print(e)
            exit(0)

        # Set up all the API signatures
        #MINIMAX_AI_API QC::AI::MinimaxAI* new_AI();
        self.QChessAI_lib.new_AI.argtypes = []
        self.QChessAI_lib.new_AI.restype = POINTER(c_int)

        #MINIMAX_AI_API long delete_AI(QC::AI::MinimaxAI* AI);
        self.QChessAI_lib.delete_AI.argtypes = [POINTER(c_int)]
        self.QChessAI_lib.delete_AI.restype = c_long

        #MINIMAX_AI_API long set_configuration(QC::AI::MinimaxAI *AI, AISearchConfig config);
        self.QChessAI_lib.set_configuration.argtypes = [POINTER(c_int), AISearchConfig]
        self.QChessAI_lib.set_configuration.restype = c_long
        
        #MINIMAX_AI_API long get_configuration(QC::AI::MinimaxAI *AI, AISearchConfig *config);
        self.QChessAI_lib.get_configuration.argtypes = [POINTER(c_int), POINTER(AISearchConfig)]
        self.QChessAI_lib.get_configuration.restype = c_long

        # Set up all the API signatures
        # MINIMAX_AI_API long find_best_move(AISearchResult *out_buffer, QC::Move* history, int history_length);
        self.QChessAI_lib.find_best_move.argtypes = [POINTER(c_int), POINTER(AISearchResult), POINTER(c_char_p), c_int]
        self.QChessAI_lib.find_best_move.restype = POINTER(c_int)

        # Current gamedata
        self.searchresult = AISearchResult()

        # Create the AI
        self.AI = self.QChessAI_lib.new_AI();

    def destroy(self):
        self.QChessAI_lib.delete_AI(self.AI)

    def set_configuration(self, config):
        self.QChessAI_lib.set_configuration(self.AI, config)
        
    def get_configuration(self):
        config = AISearchConfig()
        self.QChessAI_lib.get_configuration(self.AI, byref(config))
        return config

    def find_best_move(self, game):
        data = game.get_game_data()
        history = game.get_history()
        history_size = len(history)

        moves = (c_char_p * history_size)()
        for i in range(history_size):
            moves[i] = history[i].encode('utf-8')

        self.QChessAI_lib.find_best_move(self.AI, byref(self.searchresult), moves, history_size)
        return self.searchresult