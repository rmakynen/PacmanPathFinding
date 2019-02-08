# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util
import sys
from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newGhostPositions = currentGameState.getGhostPositions()
        oldPos = currentGameState.getPacmanPosition()
        oldFood = currentGameState.getFood()
        agentCount = currentGameState.getNumAgents()

        #If Ghost is close, then give small score to that node.
        for i in range(1,agentCount):
           current_ghost = successorGameState.getGhostPosition(i)
           dist_to_ghost = manhattanDistance(current_ghost,newPos)

           if newScaredTimes[i-1] > 2:
               continue  #we don't want to break since the other ghost could still be dangerous.
           elif dist_to_ghost < 2:
               return -10000
        #Stop actions doesn't provide positive bonuses. Therefore return 0.
        if action == "Stop":
            return(0)
            
        #The less food we have left the better score for food should be:
        food_score=(oldFood.count()-newFood.count())*120
        
        #food_score should be higher if we will be close to other new food.
        min_dist = 99999
        for food in oldFood.asList():
            dist_to_food = manhattanDistance(food,newPos)
            if dist_to_food < min_dist:
                min_dist = dist_to_food

        if min_dist == 0:
            food_score+=100
        else:
            food_score+=(100)/min_dist
        return (food_score)

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    print("Inside \"scoreEvaluationFunction\" -----------------------------------------")
    print("Eval function for task 2")
    
    
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

        
class RootNode:
    def __init__(self):
        self.children = []
    def getChildren(self):
        return(self.children)
    def addChild(self,anObject):
        self.children.append(anObject)
        
class ANode(RootNode):
    def __init__(self, xy, score):
        RootNode.__init__(self)
        self.xy = xy
        self.score = score
        #self.children = []
    def __str__(self):
        xy = str(self.xy)
        score = str(self.score)
        children = str(self.children)
        return (xy + "; " + score+ "; " + children)
    def getXY(self):
        return(self.xy)
    def getScore(self):
        return(self.score)

class MaxNode(ANode):
    def __init__(self, xy, score):
        ANode.__init__(self, xy, score)
    def getMax(self):
        all_children = self.getChildren()
        max = -sys.maxint
        maxChild=None
        # for i,aChild in enumerate(all_children):
        for aChild in all_children:
            child_score = aChild.getScore()
            # print(child_score)
            if child_score > max:
                maxChild=aChild
                max = child_score
        return(maxChild)
        
class MinNode(ANode):
    def __init__(self, xy, score):
        ANode.__init__(self, xy, score)
    def getMin(self):
        all_children = self.getChildren()
        min = sys.maxint
        minChild=None
        # for i,aChild in enumerate(all_children):
        for aChild in all_children:
            child_score = aChild.getScore()
            # print(child_score)
            if child_score < min:
                minChild=aChild
                min = child_score
        return(minChild)


        
		
class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        
        #testing MaxNode
        # aMax = MaxNode((10,5),5)
        # aMax2 = MaxNode((3,3),3)
        # aMax3 = MaxNode((4,4),4)
        # aMax.addChild(aMax2)
        # aMax.addChild(aMax3)
        # maxChild = aMax.getMax()
        # if maxChild != None:
            # print("maxChild: "+str(maxChild))

        #testing MinNode
        # aMin = MinNode((10,5),5)
        # aMin.addChild(aMax2)
        # aMin.addChild(aMax3)
        
        # minChild = aMin.getMin()
        # if minChild != None:
            # print("minChild: "+str(minChild))
            
            
        root=MaxNode((0,0),0)
        
        # root=RootNode()
        
        for i in range (0,self.depth):
            print("Current depth:"+str(i)+" ------------------------------------------------")
            for i in range (0,3):
                newChild = MinNode((i,i),i)
                root.addChild(newChild)
        
        maxChild = root.getMax()
        if maxChild != None:
            print("maxChild: "+str(maxChild))
        

        
        print("Inside \"getAction\" -------------------------------------------------------")
        # successorGameState = currentGameState.generatePacmanSuccessor(action)
        # newPos = successorGameState.getPacmanPosition()
        # newFood = successorGameState.getFood()
        # newGhostStates = successorGameState.getGhostStates()
        # newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        # newGhostPositions = currentGameState.getGhostPositions()
        # oldPos = currentGameState.getPacmanPosition()
        # oldFood = currentGameState.getFood()

        
        eval = self.evaluationFunction(gameState)
        agents = gameState.getNumAgents()
        ghost_count = 0
        if agents > 0:
            ghost_count = agents-1

            

        #Loop through all agents and print their possible actions.
        for i in range (0,agents):
            actions = gameState.getLegalActions(i)
            print("Possible actions for this actor: "+str(actions))
        
        #action = "e"
        print("Eval:    "+str(eval))
        print("Agents:  "+str(agents))
        print("Depth:   "+str(self.depth))
        
        
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
