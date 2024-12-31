# Quantum Chess AI
Development repo for all things AI for Quantum Chess

## How-To
Please take a look at the Issues, and feel free to contribute to any of them if you're interested.

#### The Quantum Chess API libraries
The libraries are taken from [the main quantum chess repo](https://github.com/quantum-realm-games/QuantumChessEngine), and are synchronized *manually*. 
7/18/2024 -Torin Schroeder 
that repo doesen't seem to exist anymore. 

7/18/2024 -Torin Schroeder 
#### Conclusion notes of the SERLO SIR Summer program and open problems of the current MCTS algorithm:
The kernel restarts when too much memory is used, which prevents simulation at a higher level being recorded at a statistically significant level

The model seems top out around 50% accuracy, so there seems to be a problem with the algorithm propperly reaching through both paths of the quantum superstates, and understanding that it will lose if any other path is taken

Improvements in algorithm operations such as open loop type tree development and memory optimization would allow for data collection at higher simulation values and thus more complex puzzles and faster computation times

Look into changing "C", aka the Exploration paramater of the MCTS algorithm because that may help with the evaluation of quantum super states