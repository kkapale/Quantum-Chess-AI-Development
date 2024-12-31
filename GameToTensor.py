
import numpy as np
import torch

def gameToTensor(game_state, AI_Player):

    tensor = torch.zeros(12,8,8)
    for y in reversed(range(8)):
        for x in range(8):
            piece = game_state.pieces[y * 8 + x]
        
            match chr(piece):
                case 'K':
                    tensor[0,y,x] = game_state.probabilities[y * 8 + x]
                    
                case 'Q':
                    tensor[1,y,x] = game_state.probabilities[y * 8 + x]
                
                case 'R':
                    tensor[2,y,x] = game_state.probabilities[y * 8 + x]
                    
                case 'B':
                    tensor[3,y,x] = game_state.probabilities[y * 8 + x]

                case 'N':
                    tensor[4,y,x] = game_state.probabilities[y * 8 + x]

                case 'P':
                    tensor[5,y,x] = game_state.probabilities[y * 8 + x]

                case 'k':
                    tensor[6,y,x] = game_state.probabilities[y * 8 + x]
                    
                case 'q':
                    tensor[7,y,x] = game_state.probabilities[y * 8 + x]
                
                case 'r':
                    tensor[8,y,x] = game_state.probabilities[y * 8 + x]

                case 'b':
                    tensor[9,y,x] = game_state.probabilities[y * 8 + x]

                case 'n':
                    tensor[10,y,x] = game_state.probabilities[y * 8 + x]

                case 'p':
                    tensor[11,y,x] = game_state.probabilities[y * 8 + x]

                case 'x':
                    tensor[11,y,x] = 0

                case _:
                    print("something went wrong")
            
        
    return tensor