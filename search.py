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
    
    # Creates an empty Stack
    frontier = util.Stack()
    explored = set()

    path = []
    action_cost = 0  # Cost of each movement
    start_node = problem.getStartState()

    # Pushes the start node to the stack
    frontier.push((start_node, path, action_cost))

    while not frontier.isEmpty():

        current_node, path, cost = frontier.pop()
        
        # Returns the final path if the current position is goal.
        if problem.isGoalState(current_node):
            return path

        # Pushes the current position to the visited list if it is not visited.
        if current_node not in explored:
            explored.add(current_node)

        # Gets successors of the current node.
        successors = problem.getSuccessors(current_node)

        # Pushes the current node's successors to the stack if they are not visited.
        for successor, action, step_cost in successors:
            if successor not in explored:
                next_state = successor
                new_path = path + [action]
                frontier.append((next_state, new_path, step_cost))
    

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 18
    
    # Creates an empty Queue.
    frontier = util.Queue()
    explored = set()

    path = []
    action_cost = 0  # Cost of each movement.
    start_node = problem.getStartState()

    # Pushes the start position to the queue.
    frontier.push((start_node, path, action_cost))

    while not frontier.isEmpty():

        current_node, path, cost = frontier.pop()
        
        # Returns the final path if the current position is goal.
        if problem.isGoalState(current_node):
            return path

        # Add the current node to the explored list if it is not explored.
        if current_node not in explored:
            explored.add(current_node)

        # Gets successors of the current node.
        successors = problem.getSuccessors(current_node)

        # Pushes the current node's successors to the queue if they are not explored
        for successor, action, step_cost in successors:
            if successor not in explored:
                next_state = successor
                new_path = path + [action]
                frontier.append((next_state, new_path, step_cost))
    
    util.raiseNotDefined()


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
    pacman_position, food_grid = state

    # Find the closest food using Manhattan distance
    distances = []
    for i in range(food_grid.width):
        for j in range(food_grid.height):
            if food_grid[i][j]:
                distance = abs(pacman_position[0] - i) + abs(pacman_position[1] - j)
                distances.append(distance)

    if distances:
        return min(distances)
    else:
        return 0
    


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
