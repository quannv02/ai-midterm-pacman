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
    fringe = util.PriorityQueue() 
    visitedList = []

    #push the starting point into queue
    fringe.push((problem.getStartState(),[],0),0 + heuristic(problem.getStartState(),problem)) # push starting point with priority num of 0
    #pop out the point
    (state,toDirection,toCost) = fringe.pop()
    #add the point to visited list
    visitedList.append((state,toCost + heuristic(problem.getStartState(),problem)))

    while not problem.isGoalState(state): #while we do not find the goal point
        successors = problem.getSuccessors(state) #get the point's succesors
        for son in successors:
            visitedExist = False
            total_cost = toCost + son[2]
            for (visitedState,visitedToCost) in visitedList:
                # if the successor has not been visited, or has a lower cost than the previous one
                if (son[0] == visitedState) and (total_cost >= visitedToCost): 
                    visitedExist = True
                    break

            if not visitedExist:        
                # push the point with priority num of its total cost
                fringe.push((son[0],toDirection + [son[1]],toCost + son[2]),toCost + son[2] + heuristic(son[0],problem)) 
                visitedList.append((son[0],toCost + son[2])) # add this point to visited list

        (state,toDirection,toCost) = fringe.pop()

    return toDirection

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
