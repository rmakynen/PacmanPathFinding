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

    #Recursive dfs.
    def _dfs(xy):
        print("------------------------------------------------------------------------")
        print(xy)
        is_goal = problem.isGoalState(xy)
        if is_goal:
            return(1)
        elif xy not in explored_list:
            explored_list.append(xy)
            successors =  problem.getSuccessors(xy) # get successors of state
            for aSuccessor in successors:
                stack.push(aSuccessor)
                xy_s = aSuccessor[0]
                val = _dfs(xy_s)
                if val==None:   #This path didn't take us to the exit. Let's remove it from the "stack" aka path.
                    stack.pop()
                else:
                    return(1)   #We found the goal. Let's return(1) to indicate that. Our path is already saved in the "stack" which is a local variable in the scope of "depthFirstSearch" function
        else:
            return(None)
        return
    
    xy=problem.getStartState() # get the starting position.
    _dfs(xy)
    
    result=[]           #list of directions that we return
    print("POP STACK: --------")
    while False == stack.isEmpty():
        poppedVal = stack.pop()
        print(poppedVal)
        result.append(poppedVal[1])

    result.reverse()
    print("Route: --------")
    for aRouteDir in result:
        print(aRouteDir)

    return(result)


	
	
	
	
	
	
	
	# -----------------------------------------------------------------------------
	
    '''
        Task 1:
        python pacman.py -l tinyMaze -p SearchAgent         #NOTE CURRRENTLY NOT IMPLEMENTED
        python pacman.py -l mediumMaze -p SearchAgent
        python pacman.py -l bigMaze -z .5 -p SearchAgent 
    '''

    print("CODE START - depthFirstSearch")
    state=problem.getStartState() # get the starting position.
    is_goal = False
    def getIndex(aState):
        return(aState[0])
    def getXY(aState):
        return(aState[1][0])
    def getDirection(aState):
        return(aState[1][1])
    def getStepCost(aState):
        return(aState[1][2])

    explored_list = []
    #explored_list.append(state) #Append (x,y) to the list of explored coordinates
    print("Start location: "+str(state))
    
    from util import Stack
    stack=Stack()

    times=0
    xy = state
    import sys
    explored_list=[]
    first_run=True
    stack=util.Stack()# Define a   Stack structure stack.
    route = []
    real_route=[]
    xy = state
    while False == is_goal:
        if xy not in explored_list:
            successors=problem.getSuccessors(xy) # get successors of state
            explored_list.append(xy)
            for aSuccessor in successors:
                util.Stack.push(stack, aSuccessor)# push  the state state into the Stack (stack) or Queue (queue)
                is_goal = problem.isGoalState(aSuccessor[0])
                if is_goal:
                    #print("GOOOAL")
                    break 
        if not stack.isEmpty():
            aSucc = util.Stack.pop(stack)
            xy = aSucc[0]
            if xy not in explored_list:
                route.append(aSucc)
    print("route------------")
    real_route=[]
    prev = None
    for i,aThing in reversed(list(enumerate(route))):
        print("--------------------------------------------")
        if i==len(route)-1:
            print("PREV: "+str(aThing))
            print("PREV_dir: "+str(aThing[1]))
            prev = aThing
            real_route.append(aThing[1])
        else:
            dist = util.manhattanDistance( prev[0], aThing[0] )
            print("distance: "+str(dist))
            if dist == 1:
                real_route.append(aThing[1])
                prev = aThing
  
    real_route.reverse()
    print("real route: ")
    print(real_route)
    print("---------")
    
    return(real_route)    
    
    '''
    #STACK
    stack=util.Stack()# Define a   Stack structure stack.
    util.Stack.push(stack, state)# push  the state state into the Stack (satck) or Queue (queue)
    state=util.Stack.pop(stack)# pop  the  last element from a stack stack
    #Queue
    queue=util.Queue()# Define a    Queue queue. 
    Pqueue=util.PriorityQueue ()# Define a    PriorityQueue Pqueue. 
    util.Queue.push(queue, state)# push  the state state into the Stack (satck) or Queue (queue)
    state=util.Queue.pop(queue)#  pop the last element from the Queue  queue.
    #STATE
    problem.isGoalState(state)# check if the state  state is a final state
    '''

    print("CODE END - depthFirstSearch")
    return  [s, s, w, s, w, w, s, w]
	
    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
