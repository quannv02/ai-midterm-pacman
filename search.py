"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from game import Directions
import util

n = Directions.NORTH
s = Directions.SOUTH
e = Directions.EAST
w = Directions.WEST


def depthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 17


def breadthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 18


def uniformCostSearch(problem):
    frontier = util.PriorityQueue()
    visited = []
    
    frontier.push((problem.getStartState(),[],0), 0)
    (state, direction, cost) = frontier.pop()
    visited.append((state,cost))

    while not problem.isGoalState(state):
        successors = problem.getSuccessors(state)
        for child in successors:
            visitedExist = False
            totalCost = cost + child[2]   
            for (visitedState,visitedCost) in visited:
                if (child[0] == visitedState) and (totalCost >= visitedCost):
                    visitedExist = True
                    break
            if not visitedExist:
                frontier.push((child[0], direction + [child[1]], cost + child[2]), cost + child[2])
                visited.append((child[0], cost + child[2]))  
        (state, direction, cost) = frontier.pop() 
    return direction

    # TODO 19


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def singleFoodSearchHeuristic(state, problem=None):
    """
    A heuristic function for the problem of single food search
    """
    # TODO 20
    pass


def multiFoodSearchHeuristic(state, problem=None):
    """
    A heuristic function for the problem of multi-food search
    """
    # TODO 21
    pass


def aStarSearch(problem, heuristic=nullHeuristic):
    '''
    return a path to the goal
    '''
    # TODO 22


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
