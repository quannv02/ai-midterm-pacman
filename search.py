"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

from game import Directions
import util, problems

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
    
    # Pushes the start node to the stack
    start_node = problem.getStartState()
    frontier.push((start_node, []))  

    while not frontier.isEmpty():

        current_node, path = frontier.pop()
        
        # Return the final path if the current node is goal state
        if problem.isGoalState(current_node):
            return path

        # If it is not explored, add the current node to the list 
        if current_node not in explored:
            explored.add(current_node)
        
        # Gets successors of the current node
        successors = problem.getSuccessors(current_node)
        for successor, action, action_cost in successors:
            if successor not in explored:
                new_path = path + [action]
                frontier.push((successor, new_path))
        
    return "No path found"


def breadthFirstSearch(problem):
    '''
    return a path to the goal
    '''
    # TODO 18
    
    # Creates an empty Queue
    frontier = util.Queue() 
    explored = set()

    # Pushes the start node to the Queue
    start_node = problem.getStartState()
    frontier.push((start_node, []))  

    while not frontier.isEmpty():

        current_node, path = frontier.pop()
        
        # Return the final path if the current node is goal state
        if problem.isGoalState(current_node):
            return path

        # If it is not explored, add the current node to the list 
        if current_node not in explored:
            explored.add(current_node)
        
        # Gets successors of the current node
        successors = problem.getSuccessors(current_node)
        for successor, action, action_cost in successors:
            if successor not in explored:
                new_path = path + [action]
                frontier.push((successor, new_path))
        
    return "No path found"


def uniformCostSearch(problem):
    visited = {} 
    frontier = util.PriorityQueue()  
    start_state = problem.getStartState()
    frontier.push((start_state, [], 0), 0)

    while not frontier.isEmpty():
        current_state, path, cost = frontier.pop()

        if problem.isGoalState(current_state):
            return path

        if (current_state not in visited) or (cost < visited[current_state]):
            visited[current_state] = cost
            for successor_state, successor_action, successor_cost in problem.getSuccessors(current_state):
                successor_path = path + [successor_action]
                successor_total_cost = cost + successor_cost
                frontier.push((successor_state, successor_path, successor_total_cost), successor_total_cost)

    return "No path found"


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
    pacmanPos, foodGrid = state
    foodList = foodGrid.asList()  # get a list of food coordinates
    heuristic = 0

    # calculate the distance from current pacmanPos to food-containing pos
    if len(foodList) > 0:
        currentState = problem.startingGameState

        # find the closest food
        closestFood = findClosestPoint(pacmanPos, foodList)
        closestFoodIndex = closestFood[0]
        closestFoodPos = foodList[closestFoodIndex]

        # find the farthest food
        farthestFood = findFarthestPoint(pacmanPos, foodList)
        farthestFoodIndex = farthestFood[0]
        farthestFoodPos = foodList[farthestFoodIndex]

        # distance between current location and closest food state
        currentToClosest = mazeDistance(pacmanPos, closestFoodPos, currentState)

        # distance between the closest food state and farthest food state
        closestToFarthest = mazeDistance(closestFoodPos, farthestFoodPos, currentState)

        heuristic = currentToClosest + closestToFarthest

    return heuristic

def mazeDistance(point1, point2, gameState):
    """
    Returns the maze distance between any two points.
    support for multiFoodSearchHeuristic
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()

    if walls[x1][y1]:
        raise 'point1 is a wall: ' + str(point1)
    if walls[x2][y2]:
        raise 'point2 is a wall: ' + str(point2)

    prob = problems.SingleFoodSearchProblem(gameState)
    return len(breadthFirstSearch(prob))

def findClosestPoint(location, goalArray):
    """
    support for calculate mazeDistance
    """

    closestPoint = 0
    closestPointCost = util.manhattanDistance(location, goalArray[0])

    for j in range(len(goalArray)):
        # calculate distance between current state to corner
        cornerLocation = goalArray[j]
        lengthToCorner = util.manhattanDistance(location, cornerLocation)

        if lengthToCorner < closestPointCost:
            closestPoint = j
            closestPointCost = lengthToCorner

    return closestPoint, closestPointCost


def findFarthestPoint(location, goalArray):
    """
    support for calculate mazeDistance
    """

    farthestPoint = 0
    farthestPointCost = util.manhattanDistance(location, goalArray[0])

    for j in range(len(goalArray)):
        # calculate distance between current state to corner
        cornerLocation = goalArray[j]
        lengthToCorner = util.manhattanDistance(location, cornerLocation)

        if lengthToCorner > farthestPointCost:
            farthestPoint = j
            farthestPointCost = lengthToCorner

    return farthestPoint, farthestPointCost

def aStarSearch(problem, heuristic=nullHeuristic):
    '''
    return a path to the goal
    '''
    # TODO 22
    frontier = util.PriorityQueue()

    def frontierAdd(frontier, state, cost):  # state is a tuple with format like : (state, cost, path)
        cost += heuristic(state[0], problem)  # f(n) = g(n) + h(n), heuristic(state, problem=None)
        frontier.push(state, cost)

    # initialize the frontier using the initial state of problem
    startState = (problem.getStartState(), 0, [])  # state is a tuple with format like : (state, cost, path)
    frontierAdd(frontier, startState, 0)  # frontierAdd(frontier, state, cost)

    # initialize the visited set to be empty
    visited = set()  # use set to keep distinct

    while not frontier.isEmpty():
        # choose a child state and remove it from the frontier
        (currentState, cost, path) = frontier.pop()

        # if it is a goal state then return the corresponding solution
        if problem.isGoalState(currentState):
            return path

        # add the state to the visited set
        if currentState not in visited:
            visited.add(currentState)

            # expand the chosen state, adding the resulting states to the frontier
            # ??? only if not in the frontier or visited set
            for childState, childAction, childCost in problem.getSuccessors(currentState):
                newCost = cost + childCost  
                newPath = path + [childAction]  
                newState = (childState, newCost, newPath)
                frontierAdd(frontier, newState, newCost)

    return "No path found"

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
