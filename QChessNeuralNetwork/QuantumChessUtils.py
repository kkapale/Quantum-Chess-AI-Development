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

def str_to_square_number(str) -> int:
    col=ord(str[0])-97
    row=int(str[1])-1
    square_number = row * 8 + col 
    return square_number
#
def is_square_empty(game,square_number):
    pce = chr(game.gamedata.pieces[square_number])
    if pce == 'x':
        return True
    else:
        return False
#
def are_split_targets_empty(game,target1_str,target2_str):
    target1_square = str_to_square_number(target1_str)
    target2_square = str_to_square_number(target2_str)
    if is_square_empty(game,target1_square) and is_square_empty(game,target2_square):
        return True
    else:
        return False
#
def is_spit_move_symmetric(game,move):
    source, target = move.split("^")
    if len(target)==4 and are_split_targets_empty(game,target[0:2],target[2:4]):
        return True
    else:
        return False

def remove_symmetric_splits(game,moves_list):
        for move in moves_list:
            if "^" in move:
                if is_spit_move_symmetric(game,move):
                    source, target = move.split("^")
                    symmetric_move = source + "^" + target[2:4] + target[0:2]
                    moves_list.remove(symmetric_move)
        return moves_list

def compacted_moves(game, moves):
        """Remove symmetric quantum moves from a set of moves"""
        #formatted_moves = format_moves(moves)
        return remove_symmetric_splits(game,moves)

def format_moves(moves) -> str:
    allmoves = []
    for move in moves:
        # Split move
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

        # Was this a measurement?
        if( move["does_measurement"] ):
            movestr += ".m" + str(move["measurement_outcome"])

        allmoves.append(movestr)

    return allmoves

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