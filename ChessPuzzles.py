# FEN codes and solutions by Atharva Kapale
# Source: chesspuzzles.com

#{'FEN':'...', 'classical-outcome':'black'/'white'/'draw', 'starting-turn':'black', 'solution':['a2a4','...'], ...}
chess_puzzles = []

def get_puzzle_fen(index):
    return chess_puzzles[index].get('FEN')

def get_puzzle_solution(index):
    return chess_puzzles[index].get('solution')

def get_puzzle_fen_opp(index):
    fenlist = get_puzzle_fen(index).split(" ")
    player = fenlist.pop(3)
    if player == 'w':
        fenlist.insert(3,"b")
        return ' '.join(fenlist)
    if player == 'b':
        fenlist.insert(3,"w")
        return ' '.join(fenlist)
    
# Mate in 1
#0
chess_puzzles.append({'FEN':"position fen r1b1qb1r/1p1n2pp/p2P4/4N2k/7n/1Q4P1/PP2NP1P/R1B2RK1 w - - 0 1",  'solution':["g3g4"]})
#1
chess_puzzles.append({'FEN':"position fen 5rk1/1pp2p1p/r5p1/p7/1P4N1/P4BP1/1B3P1P/6K1 w - - 0 1", 'solution':["g4h6"]})
#2
chess_puzzles.append({'FEN':"position fen r6k/7p/8/3B4/8/p5K1/5B2/8 w - - 0 1",
                    'solution':["f2d4"]})
#3
chess_puzzles.append({'FEN':"position fen 5qk1/3R4/6pP/6K1/8/8/1B6/8 w - - 0 1",
                    'solution':["h6h7"]})
#4
chess_puzzles.append({'FEN':"position fen 4r1rk/5p2/3p4/R7/6R1/7P/7b/2BK4 w - - 0 1",
                    'solution':["a5h5"]})
#5
chess_puzzles.append({'FEN':"position fen r3rqkb/pp1b1pnp/2p1p1p1/4P1B1/2B1N1P1/5N1P/PPP2P2/2KR3R w - - 0 1",         'solution':["e4f6"]})
#6
chess_puzzles.append({'FEN':"position fen r1b2r1k/1p1n2b1/1np1pN1p/p7/3P4/qP2B3/2Q2PPP/2RR2K1 w - - 0 1",
                    'solution':["c2h7"]})
#7
chess_puzzles.append({'FEN':"position fen r2q1rk1/p1p2ppp/bpnbpn2/8/P5P1/1PN1PP1P/1BPP2B1/R2QK1NR b - - 0 1", 'solution':["d6g3"]})
#8
chess_puzzles.append({'FEN':"position fen 1kr5/ppp5/8/8/4Q3/8/3K2B1/8 w - - 0 1",
                    'solution':["e4b7"]})
#9
chess_puzzles.append({'FEN':"position fen r2q1rk1/1p2bp2/ppn5/2pP4/2P5/P2B4/1BQ2PPP/R4RK1 w - - 0 1",
                    'solution':["d3h7"]})
#10
chess_puzzles.append({'FEN':"r1bk3r/p1q1b1p1/7p/nB1pp1N1/8/3PB3/PPP2PPP/R3K2R w - - 0 1",
                    'solution':["g5f7"]})
#11
chess_puzzles.append({'FEN':"position fen 6rk/6pp/3N4/8/8/8/7P/7K w - - 0 1",
                    'solution':["d6f7"]})
#12
chess_puzzles.append({'FEN':"position fen r2q1bnr/ppp1kBpp/2np4/4N3/8/2N4P/PPPP1PP1/R1BbK2R w - - 0 1",
                    'solution':["c3d5"]})
#13
chess_puzzles.append({'FEN':"position fen 5k2/1r3p2/8/5N2/8/8/1r3PPP/3R2K1 w - - 0 1",
                    'solution':["d1d8"]})


# Mate in 2
#14
chess_puzzles.append({'FEN':"position fen r2q1r1k/7p/4pp2/3pP3/p4P2/3R3Q/2PK3P/6R1 w - - 0 1",
                    'solution':["h3h7","h8h7","d3h3"]})
#15
chess_puzzles.append({'FEN':"position fen 3r2rk/p4p1p/3p1Pp1/3R4/2p1B2Q/8/1q4PP/4R1K1 w - - 0 1",
                    'solution':["h4h7","h8h7","d5h5"]})
#16
chess_puzzles.append({'FEN':"position fen 1n4k1/r5np/1p4PB/p1p5/2q3P1/2P4P/8/4QRK1 w - - 0 1",
                    'solution':["e1e8","g7e8","f1f8"]})
#17
chess_puzzles.append({'FEN':"position fen r2q1bnr/pp2k1pp/3p1p2/1Bp1N1B1/8/2Pp4/PP3PPP/RN1bR1K1 w - - 0 1",
                    'solution':["e5g6","e7f7","g6h8"]})
#18
chess_puzzles.append({'FEN':"position fen 4b1k1/3n3R/n3p1p1/pp1pP1N1/2pP1rN1/2P5/PPB4P/7K w - - 0 1",
                    'solution':["g4h6","g8f8","g5e6"]})
#19
chess_puzzles.append({'FEN':"position fen 3q4/4b3/2p4p/pk1p1B2/N4P2/P1Q3P1/1P5P/7K w - - 0 1",
                    'solution':["f5d3","b5a4","c3c2"]})
#20
chess_puzzles.append({'FEN':"position fen 6k1/2P3p1/4p3/3b4/p7/6qP/3Q2P1/1rR3K1 b - - 0 1",
                    'solution':["b1c1","d2c1","g3g2"]})
#21
chess_puzzles.append({'FEN':"position fen rn1q1rk1/ppp3pp/4b3/3pP3/8/2bB3Q/P1P2P1P/R2K2R1 w - - 0 1",
                    'solution':["g1g7","g8g7","h3h7"]})
#22
chess_puzzles.append({'FEN':"position fen r1bqrk2/pp4pQ/2n1p3/2bpP1N1/8/2N5/PPP2PPP/R3K2R w - - 0 1",
                    'solution':["h7h8","f8e7","h8g7"]})
#23
chess_puzzles.append({'FEN':"position fen 6k1/2B1bppp/p7/1p1r4/8/2Pn1pPq/PPQ2P1P/R5RK b - - 0 1",
                    'solution':["h3h2","h1h2","d5h5"]})


# Mate in 3
#24
chess_puzzles.append({'FEN':"position fen 6nr/pQ4pp/2Bk4/2b5/3r1pbq/2N5/PPP3PP/R1B2K1R b - - 0 1",
                    'solution':["h4f2","f1f2","d4d1","c1e3","c5e3"]})
#25
chess_puzzles.append({'FEN':"position fen r1bq4/n1p2QB1/p5Nn/7k/1pP2P2/1P4P1/P7/2K2B2 w - - 0 1",
                    'solution':["g6e5","h6f7","f1e2","c8g4","e2g4"]})
#26
chess_puzzles.append({'FEN':"position fen 5rk1/p2p2bp/1p2p3/6p1/2P2Bn1/1PNQ1PPK/P4q1P/3R4 b - - 0 1",
                    'solution':["f2h2","h3g4","f8f4","g4f4","h2h4"]})
#27
chess_puzzles.append({'FEN':"position fen r1b1r1kq/pppnpp1p/1n4pB/8/4N2P/1BP5/PP2QPP1/R3K2R w - - 0 1",
                    'solution':["b3f7","g8f7","e4g5","f7f6","e2e6"]})
#28
chess_puzzles.append({'FEN':"position fen r1bq3k/ppp1nQp1/4pN1p/6N1/2BP4/2P1n3/PP4PP/R5K1 w - - 0 1",
                    'solution':["f7g6","d8g8","g6h7","g8h7","g5f7"]})
#29
chess_puzzles.append({'FEN':"position fen r5rk/pp1b1p1p/1qn1pPpQ/5nPP/5P2/1PP5/2B5/R1B1K2R w - - 0 1",
                    'solution':["h6h7","h8h7","h5g6","h7g6","h1h6"]})
#30
chess_puzzles.append({'FEN':"position fen rnbq1knr/ppb4p/3NpPp1/3p4/1P1p1Q2/8/P1PB1PPP/R3KBNR w - - 0",
                    'solution':["f4h6","g8h6","d2h6","f8g8","f6f7"]})
#31
chess_puzzles.append({'FEN':"position fen rn1q1rk1/ppp2pp1/3p4/2b1p1nQ/2b1P2N/2NP4/PPP2PP1/R3K2R w - - 0 1",
                    'solution':["h5h8","g8h8","h4g6","h8h7","h1h8"]})
#32
chess_puzzles.append({'FEN':"position fen 2r2r1k/pbq3pB/1p2p2p/3nN3/3P4/2P5/P1Q3PP/2R2RK1 w - - 0 1",
                    'solution':["e5g6","h8h7","g6f8","h7g8","c2h7"]})
#Quantum Puzzles
#33
chess_puzzles.append({'FEN':"position fen 8/k1K5/8/8/8/8/8/1R6 w - - 0 1", 'solution':["a7^b6b7", "a7^b7b6", "a7^b6b8", "a7^b8b7", "a7^b8b6", "a7^b7b8"]})

#34
chess_puzzles.append({'FEN':"position fen k7/1P6/P1N5/8/8/8/7K/8 b - - 0 1", 'solution':["a8^b8a7", "a8^a7b8"]})

#35
chess_puzzles.append({'FEN':"position fen 8/8/8/8/8/5k2/5p2/5K2 w - - 0 1", 'solution':["f1^g1g2", "f1^g2g1", "f1^e1e2", "f1^e1e2"]})

#36
chess_puzzles.append({'FEN':"position fen rr4kR/5pp1/8/8/7R/6K1/5PP1/8 b - - 0 1", 'solution':["g8^h7f8", "g8^f8h7"]})

#37
chess_puzzles.append({'FEN':"position fen rr4k1/5pp1/8/8/8/6K1/5PP1/7R w - - 0 1", 'solution':["h1^h7h8", "h1^h8h7"]})
