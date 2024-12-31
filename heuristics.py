piece_values = { "P":   100,
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

##
# The following are all seen from White's perspective (because the entries
# start in the upper left corner, i.e. square = 8 has value 5)
##
pawn_value_list = [ 
    0,   0,   0,   0,   0,   0,   0,   0,
    5,  10,  10, -20, -20,  10,  10,   5,
    5,  -5, -10,   0,   0, -10,  -5,   5,
    0,   0,   0,  20,  20,   0,   0,   0,
    5,   5,  10,  25,  25,  10,   5,   5,
    10,  10,  20,  30,  30,  20,  10,  10,
    50,  50,  50,  50,  50,  50,  50,  50,
    0,   0,   0,   0,   0,   0,   0,   0]

knight_value_list = [
      -50, -40, -30, -30, -30, -30, -40, -50,
      -40, -20,   0,   0,   0,   0, -20, -40,
      -30,  0,   10,  15,  15,  10,   0, -30,
      -30,  5,   15,  20,  20,  15,   5, -30,
      -30,  0,   15,  20,  20,  15,   0, -30,
      -30,  5,   10,  15,  15,  10,   5, -30,
      -40, -20,   0,   5,   5,   0, -20, -40,
      -50, -40, -30,- 30, -30, -30, -40, -50]

bishop_value_list = [
            -20, -10, -10, -10, -10, -10, -10, -20,
            -10,   5,   0,   0,   0,   0,   5, -10,
            -10,  10,  10,  10,  10,  10,  10, -10,
            -10,   0,  10,  10,  10,  10,   0, -10,
            -10,   5,   5,  10,  10,   5,   5, -10,
            -10,   0,   5,  10,  10,   5,   0, -10,
            -10,   0,   0,   0,   0,   0,   0, -10,
            -20, -10, -10, -10, -10, -10, -10, -20]

rook_value_list = [ 0,  0,  0,  5,  5,  0,  0,  0,
                     -5,  0,  0,  0,  0,  0,  0, -5,
                     -5,  0,  0,  0,  0,  0,  0, -5,
                     -5,  0,  0,  0,  0,  0,  0, -5,
                     -5,  0,  0,  0,  0,  0,  0, -5,
                     -5,  0,  0,  0,  0,  0,  0, -5,
                      5, 10, 10, 10, 10, 10, 10,  5,
                      0,  0,  0,  0,  0,  0,  0,  0]

queen_value_list = [
        -20, -10, -10, -5, -5, -10, -10, -20,
        -10,   0,   0,  0,  0,   0,   0, -10,
        -10,   0,   5,  5,  5,   5,   0, -10,
         -5,   0,   5,  5,  5,   5,   0,  -5,
          0,   0,   5,  5,  5,   5,   0,  -5,
        -10,   5,   5,  5,  5,   5,   0, -10,
        -10,   0,   5,  0,  0,   0,   0, -10,
        -20, -10, -10, -5, -5, -10, -10, -20]

king_middle_game_list = [
        20,  30,  10,   0,   0,  10,  30,  20,
        20,  20,   0,   0,   0,   0,  20,  20,
       -10, -20, -20, -20, -20, -20, -20, -10,
        20, -30, -30, -40, -40, -30, -30, -20,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30,
       -30, -40, -40, -50, -50, -40, -40, -30]

king_end_game_list = [
        50, -30, -30, -30, -30, -30, -30, -50,
       -30, -30,   0,   0,   0,   0, -30, -30,
       -30, -10,  20,  30,  30,  20, -10, -30,
       -30, -10,  30,  40,  40,  30, -10, -30,
       -30, -10,  30,  40,  40,  30, -10, -30,
       -30, -10,  20,  30,  30,  20, -10, -30,
       -30, -20, -10,   0,   0, -10, -20, -30,
       -50, -40, -30, -20, -20, -30, -40, -50]

def get_piece_position_value(piece, position, endgame=False):
    piece_value_position_dict = {
        "P" :  pawn_value_list,
        "N" :  knight_value_list,
        "B" :  bishop_value_list,
        "R" :  rook_value_list,
        "Q" :  queen_value_list,
        "Km" : king_middle_game_list,
        "Ke" : king_end_game_list,
    }
    
    # If empty, nothing to score
    if( piece == "x" ):
        return 0
    
    # Determine if this is a white or a black piece
    isWhite = False if (piece in ["p","n","b","r","q","k"]) else True
    
    # Convert to uppercase
    piece = piece.upper()
    
    # Now determine position dependent material score
    if piece == "K":
        piece += "e" if endgame else "m"
    
    if not isWhite:
        file = position & 7
        rank = position >> 3
        rank = 7 - rank
        position = rank * 8 + file
    
    # Select the right array
    piece_value_list = piece_value_position_dict[piece]
    # Return the value
    return piece_value_list[position]

def neutral(game):
    return 0

def material(game, player):
    gamedata = game.get_game_data()
    """
    Returns the board score using piece values and probabilities
    Adapted from the code developed by Arefur Rahman and Kishor T. Kapale, Western Illinois University.
    """
    # If this score is positive, white has a material advantage
    evaluation = sum([piece_values[chr(gamedata.pieces[i])] * gamedata.probabilities[i] for i in range(64)])
    # Look at this from the right perspective
    perspective = 1 if (player == 0) else -1
    return evaluation*perspective

def material_and_position(game):
    """
    Reference: https://www.chessprogramming.org/Simplified_Evaluation_Function
    This function implements improved board scoring using the piece values
    based on their positions. Layout of the position
    scores for each piece is as per the positional array: [a1b1...a8b8.h8]
    Adapted from the code developed by Kishor T. Kapale, Western Illinois University.
    """

    # Get latest gamedata
    gamedata = game.get_game_data()
    
    # Already from the right perspective
    material_score = material(game)

#     position_value = 0
#     for i,p in enumerate(gamedata.pieces):
#             key = str(chr(p))
#             # Get the value of this piece being in its location
#             val = get_piece_position_value(key, i, True)
#             position_value += val * gamedata.probabilities[i]
            
    position_value = sum([get_piece_position_value(str(chr(p)), i, True)* gamedata.probabilities[i] for i,p in enumerate(gamedata.pieces)])

    perspective = 1 if (game.current_player == 0) else -1
    return position_value*perspective + material_score
