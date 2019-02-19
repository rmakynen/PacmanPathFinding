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

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    """
    Returns list that contains SCORE and ACTION
    """
    def miniMax(self, gameState, depth, agent):
        # Check if we have reached the end of the tree or game is already won/lost
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        # Get the current legal actions of given agent
        actions = gameState.getLegalActions(agent)
        # Get last agent
        lastAgent = gameState.getNumAgents() - 1
        # List of best actions
        bestActions = []

        # MAX NODE
        if agent == 0:
            bestNodeScore = -sys.maxint         # Set to negative "infinite"
            nextDepth = depth                   # Stay in the same depth
            nextAgent = 1                       # Set next agent to first ghost

            # Iterate through valid actions of the current agent
            for action in actions:
                successorState = gameState.generateSuccessor(agent, action)
                nodeScore = self.miniMax(successorState, nextDepth, nextAgent)[0]

                # Check if we have found new best MAX node with current action
                if nodeScore > bestNodeScore:
                    bestNodeScore = nodeScore   # Our new best score
                    bestActions = [action]      # Set action as a single best action

                elif nodeScore == bestNodeScore:
                    bestActions.append(action)  # We have found another equally valid action

        # MIN NODE
        else:
            bestNodeScore = sys.maxint          # Set to positive "infinite"

            # Check if we have not reached the last agent
            if agent != lastAgent:
                nextDepth = depth               # Stay in the same depth
                nextAgent = agent + 1           # Move on to next agent (ghost)
            else:
                nextDepth = depth + 1           # Move to next depth level
                nextAgent = 0                   # Move back to pacman agent

            # Iterate through valid actions of the current agent
            for action in actions:
                successorState = gameState.generateSuccessor(agent, action)
                nodeScore = self.miniMax(successorState, nextDepth, nextAgent)[0]

                # Check if we have found new best MIN node with current action
                if nodeScore < bestNodeScore:
                    bestNodeScore = nodeScore   # Our new best score
                    bestActions = [action]      # Set action as a single best action

                if nodeScore == bestNodeScore:
                    bestActions.append(action)  # We have found another equally valid action

        # Return SCORE and random ACTION from bestActions-list
        return bestNodeScore, random.choice(bestActions)

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

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0               # Starting depth
        agent = 0               # Start with Pacman agent

        # Return minimax action
        minimaxAction = self.miniMax(gameState, depth, agent)[1]
        return minimaxAction

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    """
    Returns list that contains SCORE and ACTION
    """
    def miniMax(self, gameState, depth, alpha, beta, agent, nodeType):
        # Check if we have reached the end of the tree or game is already won/lost
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        # Get the current legal actions of given agent
        actions = gameState.getLegalActions(agent)
        # Get last agent
        lastAgent = gameState.getNumAgents() - 1
        # List of best actions
        bestActions = [Directions.STOP]

        # MAX NODE
        if nodeType == "max":
            bestNodeScore = -sys.maxint         # Set to negative "infinite"

            nextDepth = depth                   # Stay in the same depth
            nextNodeType = "min"                # Set next node type back to MIN
            nextAgent = 1                       # Set next agent to first ghost

            # Iterate through valid actions of the current agent
            for action in actions:
                successorState = gameState.generateSuccessor(agent, action)
                nodeScore = self.miniMax(successorState, nextDepth, alpha, beta, nextAgent, nextNodeType)[0]

                # Always use larger number between score and alpha
                if nodeScore > alpha:
                    alpha = nodeScore

                # Check if we have found new best MAX node with current action
                if nodeScore > bestNodeScore:
                    bestNodeScore = nodeScore   # Our new best score
                    bestActions = [action]      # Set action as a single best action

                elif nodeScore == bestNodeScore:
                    bestActions.append(action)  # We have found another equally valid action

                # If best node score is larger than beta break out (pruning)
                if bestNodeScore > beta:
                    break

        # MIN NODE
        elif nodeType == "min":
            bestNodeScore = sys.maxint          # Set to positive "infinite"

            # Check if we have not reached the last agent
            if agent != lastAgent:
                nextDepth = depth               # Stay in the same depth
                nextAgent = agent + 1           # Move on to next agent (ghost)
                nextNodeType = "min"            # Set next node type to MIN
            else:
                nextDepth = depth + 1           # Move to next depth level
                nextAgent = 0                   # Move back to pacman agent
                nextNodeType = "max"            # Set next node type back to MAX

            # Iterate through valid actions of the current agent
            for action in actions:
                successorState = gameState.generateSuccessor(agent, action)
                nodeScore = self.miniMax(successorState, nextDepth, alpha, beta, nextAgent, nextNodeType)[0]

                # Always use smaller number between score and beta
                if nodeScore < beta:
                    beta = nodeScore

                # Check if we have found new best MIN node with current action
                if nodeScore < bestNodeScore:
                    bestNodeScore = nodeScore   # Our new best score
                    bestActions = [action]      # Set action as a single best action

                if nodeScore == bestNodeScore:
                    bestActions.append(action)  # We have found another equally valid action

                # If best node score is smaller than alpha break out (pruning)
                if bestNodeScore < alpha:
                    break

        # Return SCORE and random ACTION from bestActions-list
        return bestNodeScore, random.choice(bestActions)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0               # Starting depth of the minimax tree
        alpha = -sys.maxint     # MAXs best option on path to root
        beta = sys.maxint       # MINs best option on path to root
        agent = 0               # Start with Pacman agent
        nodeType = "max"        # Start with MAX node

        # Return minimax actiony
        minimaxAction = self.miniMax(gameState, depth, alpha, beta, agent, nodeType)[1]
        return minimaxAction

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    """
    Returns list that contains SCORE and ACTION
    """
    def expectiMax(self, gameState, depth, agent):
        # Check if we have reached the end of the tree or game is already won/lost
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), Directions.STOP

        # Get the current legal actions of given agent
        actions = gameState.getLegalActions(agent)
        # Get last agent
        lastAgent = gameState.getNumAgents() - 1
        # List of best actions
        bestActions = [Directions.STOP]

        # PACMAN (MAX NODE)
        if agent == 0:
            bestNodeScore = -sys.maxint         # Set to negative "infinite"

            nextDepth = depth                   # Stay in the same depth
            nextAgent = 1                       # Set next agent to first ghost

            # Iterate through valid actions of the current agent
            for action in actions:
                successorState = gameState.generateSuccessor(agent, action)
                nodeScore = self.expectiMax(successorState, nextDepth, nextAgent)[0]

                # Check if we have found new best MAX node with current action
                if nodeScore > bestNodeScore:
                    bestNodeScore = nodeScore   # Our new best score
                    bestActions = [action]      # Set action as a single best action

                elif nodeScore == bestNodeScore:
                    bestActions.append(action)  # We have found another equally valid action

        # GHOST (EXPECTIMAX NODE)
        else:
            bestNodeScore = 0                   # Start with 0 score
            probability = 1.0/len(actions)      # The probability of ghost moves

            # Check if we have not reached the last agent
            if agent != lastAgent:
                nextDepth = depth               # Stay in the same depth
                nextAgent = agent + 1           # Move on to next agent (ghost)
            else:
                nextDepth = depth + 1           # Move to next depth level
                nextAgent = 0                   # Move back to pacman agent

            # Iterate through valid actions of the current agent
            for action in actions:
                successorState = gameState.generateSuccessor(agent, action)
                nodeScore = self.expectiMax(successorState, nextDepth, nextAgent)[0]

                # Expectimax, multiply score by probability
                bestNodeScore += nodeScore * probability
                bestActions = [action]          # Always return the ghost action

        # Return SCORE and random ACTION from bestActions-list
        return bestNodeScore, random.choice(bestActions)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = 0               # Starting depth of the expectimax tree
        agent = 0               # Start with Pacman

        # Return expectimax action
        expectiMaxAction = self.expectiMax(gameState, depth, agent)[1]
        return expectiMaxAction

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return(sys.maxint)
    elif currentGameState.isLose():
        return(-sys.maxint)

    # Get ghost states, scared timers and agent count
    ghostStates = currentGameState.getGhostStates()
    scaredTimers = [ghostState.scaredTimer for ghostState in ghostStates]
    agentCount = currentGameState.getNumAgents()

    # Score factors
    baseFood = 20       # Food score factor  (20)
    baseCapsule = 25    # Capsule score factor (25)
    baseGhost = 70      # Ghost score factor (70)

    # Set base score, give remaining food a weight factor as well based on score factors
    score = currentGameState.getScore() - (baseFood + baseCapsule + baseGhost) * currentGameState.getNumFood()

    # Get pacman position
    position = currentGameState.getPacmanPosition()

    walls = currentGameState.getWalls()
    # ------------------------------------------------------------------------------
    def bfs_(new_food_list):
        distance_map = walls.copy()
        distance_map[position[0]][position[1]] = 0
        queue = util.Queue()
        queue.push(position)
        closest_food_dist = sys.maxint
        cur_dist = 0
        while not queue.isEmpty():
            x , y = queue.pop()
            cur_dist = distance_map[x][y] + 1
            if (x,y) in new_food_list:
                closest_food_dist=cur_dist-1
                if closest_food_dist < 0:
                    closest_food_dist = 0
                return(closest_food_dist)
                break;
            directions = [(-1,0),(0,-1),(1,0),(0,1)]
            for aDirection in directions:
                x_ = x + aDirection[0]
                y_ = y + aDirection[1]
                if distance_map[x_][y_] == False and not (x_ == position[0] and y_ == position[1]) :
                    distance_map[x_][y_] = cur_dist
                    queue.push((x_, y_))
    # ------------------------------------------------------------------------------
    # Get foods and calculate their impact on score based on distance
    # foodList = currentGameState.getFood().asList()
    # for food in foodList:
        # dist = util.manhattanDistance(position, food)
        # if dist == 0:
            # score = score
        # else:
            # score += baseFood * 1.0/dist

    foodList = currentGameState.getFood().asList()
    dist = bfs_(foodList)
    if dist != 0:
        score += (baseFood * 1.0/dist)

    # Get capsules and calculate their impact on score based on distance
    capsuleList = currentGameState.getCapsules()
    for capsule in capsuleList:
        dist = util.manhattanDistance(position, capsule)
        if dist != 0:
            score += (baseCapsule * 1.0/dist)
    # Get ghosts and calculate their impact on score
    for i in range(1,agentCount):
        current_ghost = currentGameState.getGhostPosition(i)
        dist = manhattanDistance(current_ghost,position)
        # If scaredTimer is running ignore ghosts
        if scaredTimers[i-1] > 3:
            continue
        # If ghost too close return negative "infinite"
        elif dist < 2:
            score += -sys.maxint
        # Else calculate ghosts impact on score based on the distance
        else:
            score -= (baseGhost * 1.0/dist)
    return score

# Abbreviation
better = betterEvaluationFunction
