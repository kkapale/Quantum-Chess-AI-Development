import numpy as np
from QuantumChessGame import QuantumChessGame

def format_piece_boards(pieces):
    white_piece_board = np.zeros((8,8,6))
    black_piece_board = np.zeros((8,8,6))

    white_pawn_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'p']
    white_piece_board[white_pawn_indices, 0] = 1
    white_rook_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'r']
    white_piece_board[white_rook_indices, 1] = 1
    white_knight_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'n']
    white_piece_board[white_knight_indices, 2] = 1
    white_bishop_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'b']
    white_piece_board[white_bishop_indices, 3] = 1
    white_queen_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'q']
    white_piece_board[white_queen_indices, 4] = 1
    white_king_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'k']
    white_piece_board[white_king_indices, 5] = 1

    black_pawn_indices = [(i / 8,i % 8) for i, c in enumerate(pieces) if c == 'P']
    black_piece_board[black_pawn_indices, 0] = 1
    black_rook_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'R']
    black_piece_board[black_rook_indices, 1] = 1
    black_knight_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'N']
    black_piece_board[black_knight_indices, 2] = 1
    black_bishop_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'B']
    black_piece_board[black_bishop_indices, 3] = 1
    black_queen_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'Q']
    black_piece_board[black_queen_indices, 4] = 1
    black_king_indices = [(i / 8, i % 8) for i, c in enumerate(pieces) if c == 'K']
    black_piece_board[black_king_indices, 5] = 1

    return white_piece_board, black_piece_board

def compute_same_and_opposite_entanglement(bell_measures):
    o = bell_measures[:4]
    s = bell_measures[4:]
    oo = np.dot(o,o)
    ss = np.dot(s,s)
    norm = oo+ss
    return ss/norm, oo/norm

def compute_entanglement_bitboards(game):
    entanglement_boards = np.zeros((8,8,2*64)) # dim 2 = (sameness, oppositeness)

    for source in range(64):
        for target in range(64):
            if( target == source ):
                continue

            bell_measures = game.get_pairwise_bell_measures(source, target)
            sameness, oppositeness = compute_same_and_opposite_entanglement(bell_measures)

            entanglement_boards[target//8, target%8, source*2 + 0] = sameness
            entanglement_boards[target//8, target%8, source*2 + 1] = oppositeness

    return entanglement_boards

def to_observation(game):
    # Needs 2 repetitions and 1 move count
    gamedata = game.get_game_data()
    halfmove = np.full((8, 8, 1), gamedata.fifty_count / 50)
    if gamedata.castle_flags == 15:
        white_castle = np.full((8, 8, 1), 1.0)
        black_castle = np.full((8, 8, 1), 1.0)
    elif gamedata.castle_flags == 10:
        white_castle = np.full((8, 8, 1), 0.0)
        black_castle = np.full((8, 8, 1), 1.0)
    elif gamedata.castle_flags == 5:
        white_castle = np.full((8, 8, 1), 1.0)
        black_castle = np.full((8, 8, 1), 0.0)
    else:
        white_castle = np.full((8, 8, 1), 0.0)
        black_castle = np.full((8, 8, 1), 0.0)

    color = np.full((8, 8, 1), gamedata.ply - 1.)
    white_piece_boards, black_piece_boards = format_piece_boards(gamedata.pieces)
    probability_board = np.reshape(gamedata.probabilities, (8,8,1))
    bell_measures = compute_entanglement_bitboards(game)
    reprs = np.concatenate([white_piece_boards, black_piece_boards, color, halfmove, white_castle, black_castle, probability_board, bell_measures], axis=2)
    return reprs

map_move1 = ['a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1', 'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2', 'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', \
 'h3', 'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4', 'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5', 'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6', \
 'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7', 'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8']

# Adapted from https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/classic/chess/chess_utils.py
# with changes made only for the quantum part of things

def square_to_coord(s):
    col = s % 8
    row = s // 8
    return (col, row)

def diff(c1, c2):
    x1, y1 = c1
    x2, y2 = c2
    return (x2 - x1, y2 - y1)

def sign(v):
    return -1 if v < 0 else (1 if v > 0 else 0)

def get_queen_dir(diff):
    dx, dy = diff
    assert dx == 0 or dy == 0 or abs(dx) == abs(dy)
    magnitude = max(abs(dx), abs(dy)) - 1

    assert magnitude < 8 and magnitude >= 0
    counter = 0
    for x in range(-1, 1 + 1):
        for y in range(-1, 1 + 1):
            if x == 0 and y == 0:
                continue
            if x == sign(dx) and y == sign(dy):
                return magnitude, counter
            counter += 1
    assert False, "bad queen move inputted"

def is_knight_move(diff):
    dx, dy = diff
    return abs(dx) + abs(dy) == 3 and 1 <= abs(dx) <= 2

def get_knight_dir(diff):
    dx, dy = diff
    counter = 0
    for x in range(-2, 2 + 1):
        for y in range(-2, 2 + 1):
            if abs(x) + abs(y) == 3:
                if dx == x and dy == y:
                    return counter
                counter += 1
    assert False, "bad knight move inputted"

def get_pawn_promotion_move(diff):
    dx, dy = diff
    assert dy == 1
    assert -1 <= dx <= 1
    return dx + 1

def get_queen_plane(diff):
    NUM_COUNTERS = 8
    mag, counter = get_queen_dir(diff)
    return mag * NUM_COUNTERS + counter

# -1 = no promote, 0 = knight, 1 = bishop, 2 = rook, 3 = queen
def get_move_plane(source, dest, promotion=-1):
    difference = diff(square_to_coord(source), square_to_coord(dest))

    QUEEN_MOVES = 56
    KNIGHT_MOVES = 8
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES

    if is_knight_move(difference):
        return KNIGHT_OFFSET + get_knight_dir(difference)
    else:
        if promotion != -1 and promotion != 3:
            return UNDER_OFFSET + 3 * get_pawn_promotion_move(difference) + promotion
        else:
            return QUEEN_OFFSET + get_queen_plane(difference)

# -1 = no promote, 0 = knight, 1 = bishop, 2 = rook, 3 = queen
def get_move_plane_split(source, dest1, dest2, promotion=-1):
    difference1 = diff(square_to_coord(source), square_to_coord(dest1))
    difference2 = diff(square_to_coord(source), square_to_coord(dest2))

    QUEEN_MOVES = 56 * 2
    KNIGHT_MOVES = 8 * 2
    QUEEN_OFFSET = 0
    KNIGHT_OFFSET = QUEEN_MOVES
    UNDER_OFFSET = KNIGHT_OFFSET + KNIGHT_MOVES

    if is_knight_move(difference1):
        return KNIGHT_OFFSET + get_knight_dir(difference1) + 8 * get_knight_dir(difference2) 
    else:
        if promotion != -1 and promotion != 3:
            return UNDER_OFFSET + 3 * get_pawn_promotion_move(difference1) + 6 * get_pawn_promotion_move(difference2) + promotion
        else:
            return QUEEN_OFFSET + get_queen_plane(difference1) + 56 * get_queen_plane(difference2)


moves_to_actions = {}
actions_to_moves = {}

def move_generation(legal_moves):
    TOTAL_normal = 73 
    TOTAL_quantum = TOTAL_normal * 2
    SPLIT_offset = 64 * TOTAL_normal
    MERGE_offset = SPLIT_offset + 64 * TOTAL_quantum

    moves = []
    for move in legal_moves:
        if move.find("^") == -1:
            source = map_move1.index(move[:2])
            dest  = map_move1.index(move[2:4])

            coord = square_to_coord(source)
            if len(move) == 5:
                promotion = 0 if move[-1] == "N" else (1 if move[-1] == "B" else 2)
            else:
                promotion = -1
            panel = get_move_plane(source, dest, promotion)
            cur_action = (coord[0] * 8 + coord[1]) * TOTAL_normal + panel
            
            moves_to_actions[move] = cur_action
            actions_to_moves[cur_action] = move

        elif move.index("^") == 2:
            # SPLIT MOVE
            source = map_move1.index(move[:2])
            dest1 = map_move1.index(move[3:5])
            dest2 = map_move1.index(move[5:7])

            coord = square_to_coord(source)
            if len(move) == 8:
                promotion = 0 if move[-1] == "N" else (1 if move[-1] == "B" else 2)
            else:
                promotion = -1
            panel = get_move_plane_split(source, dest1, dest2, promotion)
            cur_action = (coord[0] * 8 + coord[1]) * TOTAL_quantum + panel + SPLIT_offset
            
            moves_to_actions[move] = cur_action
            actions_to_moves[cur_action] = move
        else:
            # MERGE MOVE
            source1 = map_move1.index(move[:2])
            source2 = map_move1.index(move[2:4])
            dest = map_move1.index(move[5:7])

            coord1 = square_to_coord(source1)
            coord2 = square_to_coord(source1)

            if len(move) == 8:
                promotion = 0 if move[-1] == "N" else (1 if move[-1] == "B" else 2)
            else:
                promotion = -1

            panel = get_move_plane_split(dest, source1, source2, promotion)
            cur_action = (coord[0] * 8 + coord[1]) * TOTAL_quantum + panel + MERGE_offset
            
            moves_to_actions[move] = cur_action
            actions_to_moves[cur_action] = move


def index_to_move(i):
    return actions_to_moves[i]

def move_to_index(move):
    if move not in moves_to_actions:
        move_generation([move])
    return moves_to_actions[move]

if __name__ == "__main__":
    game = QuantumChessGame()
    #con = {"initial_state_fen" : "position fen 8/5P2/8/8/8/8/k7/7K w - - 0 1 moves"}
    con = {}
    game.new_game(con)

    while True:
        valid_moves = game.get_legal_moves()
        print(valid_moves)
        move_generation(valid_moves)
        # Get input string
        move = input("Enter your next move: ")
        gamedata, move_code = game.do_move(move)

        game.print_probability_board()
        reprs = to_observation(game)
        print(reprs.shape)
        print(moves_to_actions)
