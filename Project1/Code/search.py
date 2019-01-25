# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    # -----------------------------------------------------------------------------
    from util import Stack
    stack=Stack()
    is_goal = False
    explored_list = []
    #Recursive dfs:
    def _dfs(xy):
        is_goal = problem.isGoalState(xy)
        if is_goal:
            return(1)
        elif xy not in explored_list:
            explored_list.append(xy)
            successors =  problem.getSuccessors(xy) # get successors of state
            # for aSuccessor in successors:                                           #Cost for each 3 mazes: 8,246,210
            for i,aSuccessor in reversed(list(enumerate(successors))):            #Cost for each 3 mazes: 10,130,210   <-- Seems to be better on average. Use this.
                stack.push(aSuccessor)
                xy_s = aSuccessor[0]
                val = _dfs(xy_s)
                if val==None:   #This node didn't take us to the exit. Let's remove it from the "stack".
                    stack.pop()
                else:
                    return(1)   #We found the goal. Let's return(1) to indicate that. Our path is already saved in the "stack" which is a local variable in the scope of "depthFirstSearch" function
        else:
            return(None)
        return

    xy=problem.getStartState() # get the starting position.
    _dfs(xy)

    result=[]           #list of directions that we return
    while False == stack.isEmpty():
        poppedVal = stack.pop()
        result.append(poppedVal[1])

    result.reverse()
    return(result)
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    queue = util.Queue() #Instantiate the queue
    path = [] #Collection of the path from start to finish
    visited = set() #Collection of the visited nodes
    startNode = [problem.getStartState(), []] #Starting node of the pacman

    util.Queue.push(queue, startNode) #Add starting node to queue to begin looping from
    is_goal = False #Set goal to false

    #Main loop, run while goal is not found
    while not is_goal:
        #Check if Queue is empty
        if not util.Queue.isEmpty(queue):
            (node, path) = util.Queue.pop(queue) #Extract node and path from the next node in queue

        #Check if the current node is the goal node, if it is, break out of the loop
        is_goal = problem.isGoalState(node)
        if is_goal:
            break

        #Check if the current node is already visited
        if node not in visited:
            visited.add(node) #Add the current node to visited nodes list
            successors = problem.getSuccessors(node) #Get successors of the current node

            #Iterate through the child nodes of the current node
            for successor in successors:
                #Ignore successors that are already visited
                if successor[0] not in visited:
                    new_node_path = path + [successor[1]] #Create a new path and add the current successor's action to it
                    new_node = (successor[0], new_node_path) #Create new node with the path information
                    util.Queue.push(queue, new_node) #Add the new node with the path information to queue

    #Return the guide to the goal (e.g. ["West", "West", "South", ...])
    return path

def uniformCostSearch(problem):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue=util.PriorityQueue() #Instantiate the priorityqueue
    path = [] #Collection of the path from start to finish
    visited = set() #Collection of the visited nodes
    startNode = [problem.getStartState(), [], 0] #Starting node of the pacman, this time also add cost (0)

    cost = startNode[2] #+ heuristic(startNode[0], problem) #Set the initial cost
    util.PriorityQueue.push(queue, startNode, cost) #Add starting node to queue to begin looping from

    is_goal = False #Set goal to false

    #Main loop, run while goal is not found
    while not is_goal:
        #Check if PriorityQueue is empty
        if not util.PriorityQueue.isEmpty(queue):
            (node, path, cost) = util.PriorityQueue.pop(queue) #Extract node, path and cost from the next node in queue

        #Check if the current node is the goal node, if it is, break out of the loop
        is_goal = problem.isGoalState(node)
        if is_goal:
            break

        #Check if the current node is already visited
        if node not in visited:
            visited.add(node) #Add the current node to visited nodes list
            successors = problem.getSuccessors(node) #Get successors of the current node

            #Iterate through the child nodes of the current node
            for successor in successors:
                #Ignore successors that are already visited
                if successor[0] not in visited:
                    new_node_path = path + [successor[1]] #Create a new path and add the current succesor's action to it
                    new_node_cost = cost + successor[2] #Create new cost
                    new_node = (successor[0], new_node_path, new_node_cost) #Create new node with the path information and cost
                    heuristic_cost = new_node_cost #+ heuristic(successor[0], problem) #Calculate heuristic cost
                    util.PriorityQueue.push(queue, new_node, heuristic_cost) #Add the new node with the path information to queue and the heuristic cost

    #Return the guide to the goal (e.g. ["West", "West", "South", ...])
    return path

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    queue=util.PriorityQueue() #Instantiate the priorityqueue
    path = [] #Collection of the path from start to finish
    visited = set() #Collection of the visited nodes
    startNode = [problem.getStartState(), [], 0] #Starting node of the pacman, this time also add cost (0)

    cost = startNode[2] + heuristic(startNode[0], problem) #Set the initial cost
    queue.push(startNode, cost) #Add starting node to queue to begin looping from

    is_goal = False #Set goal to false

    #Main loop, run while goal is not found
    while not is_goal:
        #Check if PriorityQueue is empty
        if not queue.isEmpty():
            (node, path, cost) = queue.pop() #Extract node, path and cost from the next node in queue

        #Check if the current node is the goal node, if it is, break out of the loop
        is_goal = problem.isGoalState(node)
        if is_goal:
            break

        #Check if the current node is already visited
        if node not in visited:
            visited.add(node) #Add the current node to visited nodes list
            successors = problem.getSuccessors(node) #Get successors of the current node

            #Iterate through the child nodes of the current node
            for successor in successors:
                #Ignore successors that are already visited
                if successor[0] not in visited:
                    new_node_path = path + [successor[1]] #Create a new path and add the current succesor's action to it
                    new_node_cost = cost + successor[2] #Create new cost
                    new_node = (successor[0], new_node_path, new_node_cost) #Create new node with the path information and cost
                    heuristic_cost = new_node_cost + heuristic(successor[0], problem) #Calculate heuristic cost
                    queue.push(new_node, heuristic_cost) #Add the new node with the path information to queue and the heuristic cost

    #Return the guide to the goal (e.g. ["West", "West", "South", ...])
    #print(path)
    return path


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
