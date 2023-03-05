import random

from game import Agent
from game import Directions
import time
import search
import problems

class GoWestAgent(Agent):
    def getAction(self, state):
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP


class RandomAgent(Agent):
    def getAction(self, state):
        actions = state.getLegalPacmanActions()
        random.shuffle(actions)
        return actions[0]


class SearchAgent(Agent):
    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        # TODO 11
        
        if self.searchFunction == None: 
            raise Exception("No search function provided for SearchAgent")
        
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print(f"The total cost of the path is {totalCost}")
        


    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        # TODO 12
        
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP


class BFSFoodSearchAgent(SearchAgent):
    # TODO 13
    
    def __init__(self):
        self.searchFunction = search.breadthFirstSearch
        self.searchType = problems.SingleFoodSearchProblem
        #self.searchType = problems.MultiFoodSearchProblem
    pass


class DFSFoodSearchAgent(SearchAgent):
    # TODO 14
    
    def __init__(self):
        self.searchFunction = search.depthFirstSearch
        #self.searchType = problems.SingleFoodSearchProblem
        self.searchType = problems.MultiFoodSearchProblem
    pass


class UCSFoodSearchAgent(SearchAgent):
    # TODO 15
    
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        self.searchType = problems.SingleFoodSearchProblem
        #self.searchType = problems.MultiFoodSearchProblem
    pass


class AStarFoodSearchAgent(SearchAgent):
    # TODO 16
    
    def __init__(self):
        self.searchFunction = search.aStarSearch
        self.searchType = problems.MultiFoodSearchProblem
    pass
