# Assignment2

This assignment requires you using evolutionary strategy to find the best agent to play tic tak toe.

The agent contains a small MLP which has 16 input features and 9 output node (NO HIDDEN LAYER!)

The detail instruction is in the files.

# File structure
Here is file structure. you should only change content of files with (**).

- CI
- iris
- script
    - Agents.py **
    - Game.py
    - Model.py **
    - Player.py
    - Util.py 
- .gitlab.yml
- readme.md
- requirements.txt


# Requirements
There are few things you should do in this assignment.

Create an Agent class that contains a MLP network. This Agent should able to accept the 16 features (converted by feature_construction) and pick a empty position for next move according to the game state.

Apply a Evolutionary algorithm that create a bunch of agents, select the best of them and evolve them in order to reinforce the performance of agent.


# mark scheme

You should return the best agent of your model. Your mark will depend on the performance of best agent playing in the game.