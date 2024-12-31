from QuantumChess import format_moves
from heuristics import *
from ctypes import c_int
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp

#import sys
#sys.setrecursionlimit(NUMBER)

tnp.experimental_enable_numpy_behavior()


def QuantumZobristHash(game):
    return str(game.get_game_data().pieces)

class TranspositionTable:
    def __init__(self):
        self.reset()

    def reset(self):
        self.tt = {}

    def store(self, game, depth, plyFromRoot, val, type, move ):
        hash = QuantumZobristHash(game)
        probabilities = str(game.get_game_data().probabilities)
        self.tt[(hash,probabilities)] = {'depth':depth, 'plyFromRoot':plyFromRoot, 'value':val, 'type':type, 'move':move}

    def find(self, game, depth, plyFromRoot, alpha, beta):
        hash = QuantumZobristHash(game)
        probabilities = str(game.get_game_data().probabilities)

        # If we've seen this state before
        key = (hash, probabilities)
        if key in self.tt:

            # Have not yet seen this state at at least 'depth'
            if( self.tt[key]['depth'] < depth ):
                return -tnp.inf, [0]

            # Check if entanglement is the same
            if True:
                # Now we can return the value we want
                if self.tt[key]['type'] == "EXACT":
                    return self.tt[key]['value'], self.tt[key]['move']

                if (self.tt[key]['type'] == "UPPERBOUND") and self.tt[key]['value'] <= alpha:
                    return self.tt[key]['value'], self.tt[key]['move']

                if (self.tt[key]['type'] == "LOWERBOUND") and self.tt[key]['value'] >= beta:
                    return self.tt[key]['value'], self.tt[key]['move']

        return -tnp.inf, [0]


class ExpectiMiniMaxAI:
    def __init__(self, use_alpha_beta_pruning=True,
                       use_iterative_deepening=True,
                       use_transposition_table=False,
                       capture_move_repetitions=25,
                       heuristic=neutral):

        self.use_alpha_beta_pruning = use_alpha_beta_pruning
        self.heuristic = heuristic
        self.immediateMateValue = 10000
        self.use_iterative_deepening = use_iterative_deepening
        self.capture_move_repetitions = capture_move_repetitions
        self.use_transposition_table = use_transposition_table

        # Make a new transposition table
        self.transposition_table = TranspositionTable()

    def resetStatistics(self):
        self.numNodesSearched = 0
        self.numNodesPruned = 0
        self.numQuiescenceNodes = 0
        self.numTTLookups = 0
        self.maxDepthSearched = 0

    def findmove(self, game, maxdepth, plyFromRoot):
        self.resetStatistics()

        if not self.use_iterative_deepening:
            self.maxDepthSearched = maxdepth
            return self.negamax(game, maxdepth, plyFromRoot)

        bestmove = None
        for d in range(1, maxdepth+1):
            # TODO: RESET GAME
            game.movecode = c_int(1)
            
            bestvalue, bestmove = self.negamax(game, d, plyFromRoot)
            print("Bestval and move at %d: %d %s"%(d, bestvalue, bestmove))
            self.maxDepthSearched = d

             # Break iterative deepening if we found a mate sequence
            if bestvalue > self.immediateMateValue - 1000:
                print("Breaking from ID, bestvalue %d > %d"%(bestvalue, self.immediateMateValue-1000))
                break;

        return bestvalue, bestmove

    def negamax(self, game, maxdepth, plyFromRoot, alpha = -tnp.inf, beta = tnp.inf):
        """
            Variant of minimax (with perspective) and alpha-beta-pruning that is
            a little easier to read than minimax.
        """
        self.numNodesSearched += 1
        gamedata = game.get_game_data()

        # Get current perspective
        perspective = 1 if (gamedata.ply % 2 == 0) else -1;

        if (plyFromRoot > 0):
  			# TODO: Detect draw by repetition.

  			# If this position were a mate, its value would be immediateMateScore - plyFromRoot
            # So if alpha is larger than this value, we can skip this position because we
            # must have already seen a better move sequence to mate.
            alpha = tnp.max([alpha, -self.immediateMateValue + plyFromRoot])
            beta = tnp.min([beta, self.immediateMateValue - plyFromRoot])
            if (alpha >= beta):
                self.numNodesPruned += 1
                return alpha, [0]

        # If both won (4), the game ended in a draw (5) or the last move was an invalid quantum move (6; should not arise, currently)
        if game.movecode.value == 4 or game.movecode.value == 5 or game.movecode.value == 6:
            return 0, [0]
        # White won because of the last move
        if game.movecode.value == 2:
            return (self.immediateMateValue - plyFromRoot)*perspective, [0]
        # Black won because of the last move
        if game.movecode.value == 3:
            return -(self.immediateMateValue - plyFromRoot)*perspective, [0]

        # Look up the current game state in the transposition table
  		# We only use the value if the game state has already been searched to
        # at least the same depth that we are currently at (otherwise, it's value
        # might still change!).
        if( self.use_transposition_table ):
          ttval, ttmove = self.transposition_table.find(game, maxdepth, plyFromRoot, alpha, beta);
          if( ttval != -tnp.inf ):
              self.numTTLookups += 1
              return ttval, ttmove

        if maxdepth == 0:
            val = self.quiescenceSearch(game, alpha, beta)
            return val, [0]

        # Get all legal moves
        valid_moves = game.get_legal_moves()
        # Order the moves
        valid_moves = self.order_moves(gamedata, valid_moves)
        valid_move_str = format_moves(valid_moves)

        bestmove = None
        bestvalue = -tnp.inf
        evaltype = "UPPERBOUND"
        for i, move in enumerate(valid_moves):
            average_value = 0

            # If this is a measurement move
            num_repetitions = 1
            if valid_moves[i]['variant'] == 3:
                # Two possible outcomes of the measurement
                results = []
                for measurement in [0,1]:
                    # Move with forced measurement outcome .m0 or .m1 appended
                    capturemove = valid_move_str[i] + ".m%d"%measurement

                    # Perform the capture
                    _, movecode = game.do_move(capturemove)

                    # If the move was succesful (this should always be true!)
                    if movecode != 0 and movecode != 6:
                        # Recursive call to find value of resulting state
                        val, mv = self.negamax(game, maxdepth-1, plyFromRoot + 1, -beta, -alpha)
                        # Invert score, because it was evaluated as the opponent's turn
                        results.append(-val)
                        # Undo move
                        game.undo_move()

                # We now have the values for each measurement outcome.
                # But should weight them by the probabilities for each respective
                # outcome to occur. So we need to estimate those probabilities
                num_repetitions = self.capture_move_repetitions

                # Take into account the probability of the capturing piece being there
                source_square_probability = gamedata.probabilities[move['square1']]
                # Also take into account the probability of the captured piece being there
                target_square_probability = gamedata.probabilities[move['square2']]
                # If both source and target are 100% occupied, no need to sample
                # Note, these are now bool tensors, but they work in python conditionals
                if (tnp.ceil(1/source_square_probability) == 1) and (tnp.ceil(1/target_square_probability) == 1):
                    num_repetitions = 1

                occurences = {0:0, 1:0}
                for r in range(num_repetitions):
                    _, movecode = game.do_move(valid_move_str[i])
                    if movecode != 0 and movecode != 6:
                        # Check outcome
                        lastmove = game.get_history(max_history_size = 512)[-1]
                        occurences[lastmove["measurement_outcome"]] += 1
                        # Undo move
                        game.undo_move()

                # Now a tensor
                probability_estimates = tnp.array([occurences[0], occurences[1]])/num_repetitions
                if( tnp.abs(results[0]) != tnp.inf ):
                    average_value += probability_estimates[0]*results[0]
                if( tnp.abs(results[1]) != tnp.inf ):
                    average_value += probability_estimates[1]*results[1]

            else:
                _, movecode = game.do_move(valid_move_str[i])
                if movecode != 0 and movecode != 6:
                    # Recursive call to find value of resulting state
                    val, mv = self.negamax(game, maxdepth-1, plyFromRoot + 1, -beta, -alpha)
                    # Invert score, because it was evaluated as the opponent's turn
                    average_value += -val
                    # Undo move
                    game.undo_move()

            # Prune, because the opponent will never go down this path
            if average_value >= beta:
                self.numNodesPruned += 1

                if( self.use_transposition_table ):
                    self.transposition_table.store(game, maxdepth, plyFromRoot, beta, "LOWERBOUND", mv);
                return beta, mv

            # Found a new best move
            if average_value > alpha:
                alpha = average_value
                bestmove = valid_move_str[i]
                bestvalue = alpha
                evalType = "EXACT"

        if( self.use_transposition_table ):
            self.transposition_table.store(game, maxdepth, plyFromRoot, bestvalue, evaltype, bestmove);

        return bestvalue, bestmove

    def order_moves(self, gamedata, moves):
        """
            Order the `moves` according to some simple heuristics
        """
        scores = []
        for i, move in enumerate(moves):
            score = 0
            # If this is a capture move
            if move['variant'] == 3:
                capturing_piece = chr(gamedata.pieces[move['square1']])
                captured_piece = chr(gamedata.pieces[move['square2']])
                score += piece_values[captured_piece] - piece_values[capturing_piece]

            # Promotion moves are good
            if move['promotion_piece'] != 0:
                score += piece_values[chr(move['promotion_piece'])]

            # If we are moving into an attacked position by opponent pawns, that's bad
            # TODO
            #  1) Get all squares that are attacked by opponent pawns
            #  2) If move['square2'] or move['square3'] is in this set, penalty

            scores.append(score)

        # Sort
        indices = tnp.argsort(scores)[::-1]
        moves = tnp.array(moves)[indices]
        return moves

    def quiescenceSearch(self, game, alpha, beta):
        """
            Search only the capture moves, until we reach a 'quiet' state in
            which no more capture moves are available.
        """
        self.numQuiescenceNodes += 1
        gamedata = game.get_game_data()

        # Evaluate using our current heuristic
        val = self.heuristic(game)

        # See if we can prune
        if val >= beta:
            return beta
        alpha = tnp.max([alpha, val])

        # Get all legal moves
        valid_moves = game.get_legal_moves()
        # Keep only the capture moves
        capture_moves = []
        for move in valid_moves:
            if move['variant'] == 3:
                capture_moves.append(move)

        # Order the moves
        capture_moves = self.order_moves(gamedata, capture_moves)
        capture_move_str = format_moves(capture_moves)

        bestmove = None
        bestvalue = -tnp.inf
        for i, move in enumerate(capture_moves):
            _, movecode = game.do_move(capture_move_str[i])

            if movecode != 0 and movecode != 6:
                # Recursive call to find value of resulting state
                # Invert score, because it was evaluated as the opponent's turn
                val = -self.quiescenceSearch(game, -beta, -alpha)
                # Undo move
                game.undo_move()

            # Prune, because the opponent will never go down this path
            if val >= beta:
                return beta

            # Found a new best move
            alpha = tnp.max([alpha, val])

        return alpha

    def printStatistics(self):
        print("#nodes searched: ", self.numNodesSearched)
        print("#nodes pruned: ", self.numNodesPruned)
        print("#quiescence nodes searched: ", self.numQuiescenceNodes)
        print("#TT lookups: ", self.numTTLookups)
        print("#max depth: ", self.maxDepthSearched)

    def minimax(self, game, maximizing, maxdepth, plyFromRoot, alpha = -tnp.inf, beta = tnp.inf, white_is_maximizing=True, move_ordering=False):
        # Get source square probability
        gamedata = game.get_game_data()

        # Check endgame conditions
        # !! IMPORTANT:
        #    This checks the movecode of the LAST move that was made. So e.g. if that was a move by WHITE, we
        #    are now considering this state from the point of view of BLACK!

        if (plyFromRoot > 0):
  			# TODO: Detect draw by repetition.

  			# If this position were a mate, its value would be immediateMateScore - plyFromRoot
            # So if alpha is larger than this value, we can skip this position because we
            # must have already seen a better move sequence to mate.
            alpha = tnp.max([alpha, -self.immediateMateValue + plyFromRoot])
            beta = tnp.min([beta, self.immediateMateValue - plyFromRoot])
            if (alpha >= beta):
                return alpha, [0]

        # If both won (4), the game ended in a draw (5) or the last move was an invalid quantum move (6; should not arise, currently)
        if game.movecode.value == 4 or game.movecode.value == 5 or game.movecode.value == 6:
            return 0, [0]

        # White won because of the last move
        if game.movecode.value == 2:
            # If white made the last move, and white is maximizing, this is positive
            if white_is_maximizing and (not maximizing):
                return self.immediateMateValue - plyFromRoot, [0]
            return -(self.immediateMateValue - plyFromRoot), [0]

        # Black won because of the last move
        if game.movecode.value == 3:
            # If black made the last move, and black is maximizing, this is positive
            if (not white_is_maximizing) and maximizing:
                return self.immediateMateValue - plyFromRoot, [0]
            return -(self.immediateMateValue - plyFromRoot), [0]


        if maxdepth == 0:
            # heuristic evaluates the current board from the perspective
            # of the current player. So we invert the score
            val = -1 * self.heuristic(game, white_is_maximizing)
            return val, [0]

        # Get all legal moves
        valid_moves = game.get_legal_moves()
        valid_move_str = format_moves(valid_moves)

        maxv = -tnp.inf
        minv = tnp.inf

        # # Perform each move, evaluate the heuristic, and sort
        # scores = []
        # if move_ordering:
        #     for i, move in enumerate(valid_moves):
        #         _, movecode = game.do_move(valid_move_str[i])
        #         scores.append( self.heuristic(game, white_is_maximizing) )
        #
        #         # Undo move if we actually performed one
        #         if movecode != 0 and movecode != 6:
        #             game.undo_move()
        #
        #     indices = tnp.argsort(scores)[::-1]
        #     valid_moves = tnp.array(valid_moves)[indices]
        #     valid_move_str = format_moves(valid_moves)

        bestmove = None
        for i, move in enumerate(valid_moves):
            average_value = 0

            # If this is a measurement move
            if valid_moves[i]['variant'] == 3:
                # Take into account the probability of the capturing piece being there
                source_square_probability = gamedata.probabilities[move['square1']]
                # Also take into account the probability of the captured piece being there
                target_square_probability = gamedata.probabilities[move['square2']]

                # Number of repetitions
                num_repetitions = tnp.ceil(1/source_square_probability) * tnp.ceil(1/target_square_probability)
                num_repetitions = int(num_repetitions*self.move_repetition_factor)

                for r in range(num_repetitions):
                    _, movecode = game.do_move(valid_move_str[i])

                    if movecode != 0 and movecode != 6:
                        val, mv = self.findmove(game, not maximizing, maxdepth-1, plyFromRoot + 1, alpha, beta, white_is_maximizing)
                        average_value += 1/num_repetitions * val
                        game.undo_move()

            else:
                _, movecode = game.do_move(valid_move_str[i])

                if movecode != 0 and movecode != 6:
                    val, mv = self.findmove(game, not maximizing, maxdepth-1, plyFromRoot + 1, alpha, beta, white_is_maximizing)
                    average_value = val
                    game.undo_move()

            if maximizing:
                if average_value > maxv:
                    maxv = average_value
                    bestmove = valid_move_str[i]

                if self.use_alpha_beta_pruning:
                    if maxv >= beta:
                        break
                    alpha = tnp.max([alpha, maxv])

            else:
                if average_value < minv:
                    minv = average_value
                    bestmove = valid_move_str[i]

                if self.use_alpha_beta_pruning:
                    if minv <= alpha:
                        break
                    beta = tnp.min([beta, minv])

        if maximizing:
            return maxv, bestmove
        else: return minv, bestmove
