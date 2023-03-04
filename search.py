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
    pass


def multiFoodSearchHeuristic(state, problem=None):
    """
    A heuristic function for the problem of multi-food search
    """
    # TODO 21
    position, foodGrid = state
    H = 0
    maxDistance = 0
    # find the farthest distance by Astar search using mazeDistance() function.
    for y in range(foodGrid.height):
        for x in range(foodGrid.width):
            if (foodGrid[x][y] == 1) and (getMazeDistance(x,y) > maxDistance):
                maxDistance = getMazeDistance(x,y)
    H = maxDistance     
    return H

def getMazeDistance(self, start, end):
        """
        Returns the maze distance between two positions in the maze.
        """
        from util import PriorityQueue
        visited = set()
        frontier = PriorityQueue()
        frontier.push((start, []), 0)

        while not frontier.isEmpty():
            state, actions = frontier.pop()
            if state == end:
                return len(actions)
            if state in visited:
                continue
            visited.add(state)
            for successor, cost in self.getSuccessors(state):
                new_actions = actions + [successor]
                priority = self.getCostOfActions(new_actions) + self.heuristic(successor)
                frontier.push((successor, new_actions), priority)

        return float("inf")

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
