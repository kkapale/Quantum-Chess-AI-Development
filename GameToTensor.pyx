import torch

def gameToTensor(game_state, AI_Player):
    cdef list whiteArr = ['K', 'Q', 'R', 'B', 'N', 'P']
    cdef list blackArr = ['k', 'q', 'r', 'b', 'n', 'p']
    
    if AI_Player%2 == 0:
        playerArr = whiteArr
        opponentArr = blackArr
    else:
        playerArr = blackArr
        opponentArr = whiteArr
    tensor = torch.zeros(12,8,8)

    cdef int x, y
    cdef str piece_char
    cdef int piece

    for y in reversed(range(8)):
        for x in range(8):
            piece = game_state.pieces[y * 8 + x]
            piece_char = chr(piece)
            if piece_char == playerArr[0]:
                tensor[0,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == playerArr[1]:
                tensor[1,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == playerArr[2]:
                tensor[2,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == playerArr[3]:
                tensor[3,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == playerArr[4]:
                tensor[4,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == playerArr[5]:
                tensor[5,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == opponentArr[0]:
                tensor[6,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == opponentArr[1]:
                tensor[7,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == opponentArr[2]:
                tensor[8,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == opponentArr[3]:
                tensor[9,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == opponentArr[4]:
                tensor[10,y,x] = game_state.probabilities[y * 8 + x]
            elif piece_char == opponentArr[5]:
                tensor[11,y,x] = game_state.probabilities[y * 8 + x]
            #elif piece_char == 'x':
                #continue
            else:
                continue           
    if AI_Player%2 == 1:
        tensor = torch.flip(tensor, [1])    
    return tensor