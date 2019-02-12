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


def scoreEvaluationFunction2(currentGameState,oldState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """

    food_score = currentGameState.getScore()
    ghost_dist_score=0
    return (ghost_dist_score, food_score)

    # print("Inside \"scoreEvaluationFunction\" -----------------------------------------")
    # print("Eval function for task 2")

    # successorGameState = currentGameState.generatePacmanSuccessor(action)
    # newPos = successorGameState.getPacmanPosition()

    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    # newGhostPositions = currentGameState.getGhostPositions()
    newPos = currentGameState.getPacmanPosition()
    oldFood = oldState.getFood()
    newFood = currentGameState.getFood()
    # oldFood


    agentCount = currentGameState.getNumAgents()

    #If Ghost is close, then give small score to that node.
    ghost_dist_score = 0
    for i in range(1,agentCount):
        # current_ghost = successorGameState.getGhostPosition(i)
        current_ghost = currentGameState.getGhostPosition(i)
        dist_to_ghost = manhattanDistance(current_ghost,newPos)

        if newScaredTimes[i-1] > 2:
           continue  #we don't want to break since the other ghost could still be dangerous.
        elif dist_to_ghost < 5:
            ghost_dist_score+=(-260/(1+dist_to_ghost*dist_to_ghost))
            #return (ghost_dist_score)

    # ghost_dist_score = 0
    #Stop actions doesn't provide positive bonuses. Therefore return.
    ##NOTE IMPLEMENT THE STOP ELSEWHERE. See line: ghost_score, food_score = self.evaluationFunction(nodeState)
    # if action == "Stop":
        # return(ghost_dist_score)

    food_score=0

    #The less food we have left the better score for food should be:
    food_score=(oldFood.count()-newFood.count())*120
    # food_score=0

    #food_score should be higher if we will be close to other new food.
    min_dist = 999999999
    for food in newFood.asList():
        dist_to_food = manhattanDistance(food,newPos)
        if dist_to_food < min_dist:
            min_dist = dist_to_food

    food_score+=100/(1+min_dist)
    # if min_dist == 0:
        # food_score+=100
    # else:
        # food_score+=(100)/min_dist
    return (ghost_dist_score,food_score)



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
        self.parent = None
        self.nodeType = "RootNode"
    def hasChildren(self):
        return(len(self.children)!=0)
    def hasParent(self):
        return(self.parent != None)
    def getChildren(self):
        return(self.children)
    def addChild(self,anObject):
        self.children.append(anObject)
    def setParent(self,anObject):
        self.parent = anObject
	def getParent(self):
		return(self.parent)
	def getNodeType(self):
		return(self.nodeType)

class GenericNode(RootNode):
    # def __init__(self, gameState, score, action):
    def __init__(self, gameState, action):
        RootNode.__init__(self)
        self.gameState = gameState
        # self.score = score
        self.score = None
        self.nodeType = "GenericNode"
        GenericNode.action = action
	def getNodeAction(self):
		return(self.action)
	def getNodeType(self):
		return(self.nodeType)
    def __str__(self):
        gameState = str(self.gameState)
        score = str(self.score)
        children = str(self.children)
        return (gameState + "; " + score+ "; " + children)
    def getState(self):
        return(self.gameState)
    def getScore(self):
        return(self.score)
    def setScore(self,score):
        self.score = score

class MaxNode(GenericNode):
    # def __init__(self, gameState, score, action):
        # GenericNode.__init__(self, gameState, score, action)
    def __init__(self, gameState, action):
        GenericNode.__init__(self, gameState, action)
        self.nodeType = "MaxNode"
        self.score = None
    def getScore(self):
        return(self.score)
    # def getScore(self):
        # return(self.getBestNode())
    def getNodeAction(self):
        return(GenericNode.action)
    def getNodeType(self):
        return(self.nodeType)
    #Get either Min or max depending on the node type for MaxNode class object its always max
    def getBestNode(self):
        return(self.getMax)
    def getBestNodeValue(self):
        return(self.getMax("value"))
    def getMax(self, *args, **kwargs):
        return_object=1
        if len(args) == 1 and args[0] == "value":
            return_object=0
        all_children = self.getChildren()
        max = -sys.maxint
        maxChild=None
        # for i,aChild in enumerate(all_children):
        for aChild in all_children:
            # child_score = aChild.getScore()
            child_score = aChild.getBestNodeValue()
            # print(child_score)
            if child_score > max:
                maxChild=aChild
                max = child_score
        self.score = max
        if return_object == 0:
            if max == -sys.maxint:
                return(None)
            return(max)

        return(maxChild)

class MinNode(GenericNode):
    # def __init__(self, gameState, score, action):
        # GenericNode.__init__(self, gameState, score, action)
    def __init__(self, gameState, action):
        GenericNode.__init__(self, gameState, action)
        self.nodeType = "MinNode"
        # self.score = None
    # def getScore(self):
        # return(self.score)
    # def getScore(self):
        # return(GenericNode.getScore())
    def getNodeAction(self):
        return(GenericNode.action)
    def getNodeType(self):
        return(self.nodeType)
    #Get either Min or max depending on the node type for MinNode class object its always min
    def getBestNode(self):
        return(self.getMin)
    def getBestNodeValue(self):
        return(self.getMin("value"))
    # def getMin(self):
        # all_children = self.getChildren()
        # min = sys.maxint
        # minChild=None
        # for aChild in all_children:
            # child_score = aChild.getScore()
            # if child_score < min:
                # minChild=aChild
                # min = child_score
        # return(minChild)
    def getMin(self, *args, **kwargs):
        return_object=1
        if len(args) == 1 and args[0] == "value":
            return_object=0

        all_children = self.getChildren()
        min = sys.maxint
        minChild=None
        # for i,aChild in enumerate(all_children):
        for aChild in all_children:
            # child_score = aChild.getScore()
            child_score = aChild.getBestNodeValue()
            # print(child_score)
            if child_score < min:
                minChild=aChild
                min = child_score



        if return_object == 0:
            if min == sys.maxint:
                return(None)
            return(min)

        return(minChild)

        # self.score = max

    '''
    def getMax(self, *args, **kwargs):
        return_object=1
        if len(args) == 1 and args[0] == "value":
            return_object=0
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
        self.score = max
        if return_object == 0:
            if max == -sys.maxint:
                return(None)
            return(max)

        return(maxChild)
    '''


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

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth * 2  # Depth of the minmax tree
        alpha = -sys.maxint     # MAXs best option on path to root
        beta = sys.maxint       # MINs best option on path to root
        agent = 0               # Start with Pacman

        """
        Returns list that contains SCORE and ACTION
        """
        def miniMax(gameState, depth, alpha, beta, agent):
            # Check if we have reached the end of the tree or game is already won/lost
            if depth <= 0 or gameState.isWin() or gameState.isLose():
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
                nextDepth = depth - 1               # Move down in depth level
                nextAgent = 1                       # Set next agent to first ghost

                # Iterate through valid actions of the current agent
                for action in actions:
                    successorState = gameState.generateSuccessor(agent, action)
                    nodeScore = miniMax(successorState, nextDepth, alpha, beta, nextAgent)[0]

                    # Always use larger number between score and alpha
                    # if nodeScore > alpha:
                        # alpha = nodeScore

                    # Check if we have found new best MAX node with current action
                    if nodeScore > bestNodeScore:
                        bestNodeScore = nodeScore   # Our new best score
                        bestActions = [action]      # Set action as a single best action

                    elif nodeScore == bestNodeScore:
                        bestActions.append(action)  # We have found another equally valid action

                    # If best node score is larger than beta break out (pruning)
                    # if bestNodeScore > beta:
                        # break
            # MIN NODE
            else:
                bestNodeScore = sys.maxint          # Set to positive "infinite"

                # Check if we have not reached the last agent
                if agent != lastAgent:
                    nextDepth = depth               # Stay in the same depth
                    nextAgent = agent + 1           # Move on to next agent (ghost)
                else:
                    nextDepth = depth - 1           # Move down in depth level
                    nextAgent = 0                   # Move back to pacman agent

                # Iterate through valid actions of the current agent
                for action in actions:
                    successorState = gameState.generateSuccessor(agent, action)
                    nodeScore = miniMax(successorState, nextDepth, alpha, beta, nextAgent)[0]

                    # Always use smaller number between score and beta
                    # if nodeScore < beta:
                        # beta = nodeScore

                    # Check if we have found new best MIN node with current action
                    if nodeScore < bestNodeScore:
                        bestNodeScore = nodeScore   # Our new best score
                        bestActions = [action]      # Set action as a single best action

                    if nodeScore == bestNodeScore:
                        bestActions.append(action)  # We have found another equally valid action

                    # If best node score is smaller than alpha break out (pruning)
                    # if bestNodeScore < alpha:
                        # break

            # Return SCORE and random ACTION from bestActions-list
            return bestNodeScore, random.choice(bestActions)

        # Return minimax action
        minimaxAction = miniMax(gameState, depth, alpha, beta, agent)[1]
        return minimaxAction
        '''
        # print("Inside \"getAction\" -------------------------------------------------------")

        #eval = self.evaluationFunction(gameState)
        
        global agent_count
        agent_count = gameState.getNumAgents()
        ghost_count = 0
        if agent_count > 0:
            ghost_count = agent_count-1
        # ---------------------------------------------------------------------------------
        def incrementAgentIndex(agentIndex):
            if ghost_count == 0:
                return(0)
            agentIndex +=1
            if agentIndex >= agent_count:
                agentIndex=0
            return(agentIndex)
        # ---------------------------------------------------------------------------------
        def getDepth(agentIndex):
            if agentIndex == 0:
                return(1)
            return(0)
        # ---------------------------------------------------------------------------------
        def giveScore(node,oldNode,agentIndex,current_depth,action):
            if node == None:
                return(-1)
            nodeState = node.getState()
            oldState = oldNode.getState()
            nodeAction = node.getNodeAction()
            # ghost_score, food_score = self.evaluationFunction2(nodeState,oldState)            a better evaluation function call:
            # if nodeAction == "Stop":
                # food_score=-1
                # return(ghost_dist_score)
            # score = ghost_score + food_score
        
            score = self.evaluationFunction(nodeState)
            # if agentIndex == 0:
                # print("-----------------------------------------------")
                # print("current_depth: -------> "+str(current_depth+1)+" <-------")
                # print("agentIndex:     "+str(agentIndex))
                # print("action:         "+str(action))
                # print("My given score: "+str(score))
                # print("My State      : "+str(node.getState()))
            return(score)
        # ---------------------------------------------------------------------------------  
        target_depth = self.depth
        def setScoreAction(best_action,best_score,new_action,new_score, agentIndex):
            if agentIndex==0:   #Max
                if new_score >  best_score:
                    return(new_action,new_score)
                else:
                    return(best_action,best_score)
            else:       #Min
                if new_score <  best_score:
                    return(new_action,new_score)
                else:
                    return(best_action,best_score)
        
        def miniMax(gameState,agentIndex,current_depth):
            global agent_count
            agentIndex=incrementAgentIndex(agentIndex)
            current_depth += getDepth(agentIndex)

            if current_depth <= 0 and (gameState.isWin() or gameState.isLose()):
                return(Directions.STOP, self.evaluationFunction(gameState))

            if current_depth == target_depth:
                score = scoreEvaluationFunction(gameState)
                action = None
                return(None,score)
            actions = gameState.getLegalActions(agentIndex)
            # print("Possible actions for agent["+str(agentIndex)+"] are: "+str(actions))
            best_action = None
            best_score = None
            for i,action in enumerate(actions):
                successorState = gameState.generateSuccessor(agentIndex, action)
                action_, score_ = miniMax(successorState,agentIndex,current_depth)
                if score_ != None:
                    # print("Score: "+str(score_))
                    # print("Score: "+str(successorState))
                    if best_action == None or best_score == None:
                        best_action=action
                        best_score=score_
                    else:
                        best_action,best_score = setScoreAction(best_action,best_score,action,score_, agentIndex)
            return(best_action,best_score)
        '''
        '''
        def miniMax(gameState,agentIndex,current_depth):
            global agent_count
            agentIndex=incrementAgentIndex(agentIndex)
            current_depth += getDepth(agentIndex)
            if current_depth == target_depth:
                score = scoreEvaluationFunction(gameState)
                return(score)
            actions = gameState.getLegalActions(agentIndex)
            print("Possible actions for agent["+str(agentIndex)+"] are: "+str(actions))
            if len(actions)==0:
                print("Out of moves!")
                sys.exit(1)
                actions.append("Stop")
            best_action = None
            best_score = None
            action_score=[]
            
            # if agentIndex != 0:
            # current_agent 
                
            for i,action in enumerate(actions):
                successorState = gameState.generateSuccessor(agentIndex, action)
                score = miniMax(gameState,agentIndex,current_depth)
                if score != None:
                    print("Score: "+str(score))
                    print("Score: "+str(successorState))
                    action_score.append([action,score])
                # should_continue_checking_scores=1
                # if should_continue_checking_scores == 0:
                    # break
                # score = None
                # if current_depth == target_depth-1 and agentIndex == agent_count-1:

                    # score = scoreEvaluationFunction(successorState)
            print("Done:------------------------------------------------------------------------.")
            
            def findSmallest(action_score):
                # smallest = None
                # smallest_val = 
                smallest_unit = ["Stop",sys.maxint]
                for anActionScore in action_score:
                    if anActionScore[1] < smallest_unit[1]:
                        smallest_unit=anActionScore
                    # print("current_depth: "+str(current_depth))
                    # print("agentIndex:    "+str(agentIndex))
                    # print("anActionScore: "+str(anActionScore))
                return(smallest_unit)
            def findBiggest(action_score):
                biggest_unit = ["Stop",-sys.maxint]
                for anActionScore in action_score:
                    if anActionScore[1] > biggest_unit[1]:
                        biggest_unit=anActionScore
                return(biggest_unit)
            if agentIndex != 0:
                return(findSmallest(action_score))
            else:
                return(findSmallest(action_score))
            '''
            # current_agent 
            
            # for anActionScore in action_score:
                # print("current_depth: "+str(current_depth))
                # print("agentIndex:    "+str(agentIndex))
                # print("anActionScore: "+str(anActionScore))
            # print("Done:------------------------------------------------------------------------.")
            
            # if agentIndex == 0:
            
            # if else:
        '''
        def miniMax(parent,agentIndex,current_depth):
            global agent_count
            agentIndex=incrementAgentIndex(agentIndex)
            current_depth += getDepth(agentIndex)
            if current_depth == target_depth:
                return
            parentState=parent.getState()
            actions = parentState.getLegalActions(agentIndex)
            print("Possible actions for agent["+str(agentIndex)+"] are: "+str(actions))
            testValie=0
            action = "Stop"
            best_action = "Stop"
            best_score=None
            for i,action in enumerate(actions):
                testValie+=1
                childState = parentState.generateSuccessor(agentIndex, action)
                if agentIndex == 0:
                    childNode = MaxNode(childState,action)
                else:
                    childNode = MinNode(childState,action)
                childNode.setParent(parent)
                # if current_depth < target_depth:   # if current_depth < self.depth:
                miniMax(childNode,agentIndex,current_depth)

                #Alpha-Beta pruning --------------------------------------------------------------------------
                should_continue_checking_scores=1
                # if i>0:
                    # if parent.getNodeType() == "MaxNode":
                        # if agentIndex != 0:         #This is true when we have a MinNode
                            # parent_best = parent.getBestNode()
                            # if parent_best != None:
                                # parent_best_val = parent.getBestNodeValue()
                                # if parent_best_val != None:
                                    # print("parent_best_val: "+str(parent_best_val))
                                    # should_continue_checking_scores=0
                if should_continue_checking_scores == 0:
                    break
                score = None
                if current_depth == target_depth-1 and agentIndex == agent_count-1:
                    # if target_depth_reached==True:
                    score = giveScore(childNode,parent,agentIndex,current_depth,action)
                    # print("scoreeeeeee:"+str(score))
                elif (childNode.hasChildren()):
                    score = childNode.getBestNodeValue()
                # else:
                    # print("No CHILDREN!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # sys.exit(1)
                if score != None:
                    childNode.setScore(score)
                parent.addChild(childNode)
                # break
            # if testValie == 0:
                # print("THINGYYYY:"+str(testValie))
                # childState = parent.getState()
                # print(childState)
                # print("Action:"+str(action))
                # sys.exit(1)
            # print("-----------------------------")
            # print(parent.getState())
            children = parent.getChildren()
            # for aChild in children:
                # print(aChild.getState())
            # print("-----------------------------")
            # sys.exit(1)
            # parent.getBestNodeValue()
        
        # ---------------------------------------------------------------------------------
        # action_ = None
        # root=MaxNode(gameState,action_)

        # best_node = root.getBestNode()
        
        # move = miniMax(gameState,-1,-1)
        action, score  = miniMax(gameState,-1,-1)
        if action == None:
            print("None ACTION!!!!!!!!!!!!!!!!!!!!")
            return("Stop")
            # sys.exit(1)
        return(action)
        print("--------------------------------------------")
        print("move[0]::" +str(move[0]))
        print("--------------------------------------------")
        return(move[0])
        # print("---------------------======================---------------------------------=======================================----")
        # miniMax(root,-1,-1)
        # print("---------------------======================---------------------------------=======================================----")
        print("Now print the entire TREEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEeeeEEEEEEEEEEe")
        # node = root
        # while True:
        # for i in range (0,1):
            # if node.hasChildren():
                # for a
                # gameState = node.getState()
                # if gameState != None:
                    # print(gameState)
                    
        global counter
        counter=0
        def printTree(node):
            if node.hasChildren():
                # print("Yes, the node has children: "+str(node.getState()))
                for aChild in node.getChildren():
                    # print(printTree(aChild))
                    printTree(aChild)
                    # print(aChild)
            node_action = node.getNodeAction()
            if node_action != None:
                score = node.getBestNodeValue()
                print("node_action:      "+str(node_action))
                print("getBestNodeValue: "+str(score))
                gameState = node.getState()
                if gameState != None:
                    global counter
                    counter+=1
                    print("----------------------------------------------------------")
                    print(gameState)
            else:
                print("No action for this node")
            return
                # nodeState = gameState.generateSuccessor(0, "Stop")
                # print(nodeState)
			# print(node)
            # else:
                # print(node)    
        printTree(root)
        if counter>0:
            print("----------------------------------------------------------")
            print("How many members does the tree have: "+str(counter))
        
        sys.exit(1)
        util.raiseNotDefined()
        return("Stop")      
        '''

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        depth = self.depth * 2  # Depth of the minmax tree
        alpha = -sys.maxint     # MAXs best option on path to root
        beta = sys.maxint       # MINs best option on path to root
        agent = 0               # Start with Pacman
        nodeType = "max"        # Start with MAX node

        """
        Returns list that contains SCORE and ACTION
        """
        def miniMax(gameState, depth, alpha, beta, agent, nodeType):
            # Check if we have reached the end of the tree or game is already won/lost
            if depth <= 0 or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState), Directions.STOP

            # Get the current legal actions of given agent
            actions = gameState.getLegalActions(agent)
            # Get last agent
            lastAgent = gameState.getNumAgents() - 1
            # List of best actions
            bestActions = []

            # MAX NODE
            if nodeType == "max":
                bestNodeScore = -sys.maxint         # Set to negative "infinite"

                nextDepth = depth - 1               # Move down in depth level
                nextNodeType = "min"                # Set next node type back to MIN
                nextAgent = 1                       # Set next agent to first ghost

                # Iterate through valid actions of the current agent
                for action in actions:
                    successorState = gameState.generateSuccessor(agent, action)
                    nodeScore = miniMax(successorState, nextDepth, alpha, beta, nextAgent, nextNodeType)[0]

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
                    nextDepth = depth - 1           # Move down in depth level
                    nextAgent = 0                   # Move back to pacman agent
                    nextNodeType = "max"            # Set next node type back to MAX

                # Iterate through valid actions of the current agent
                for action in actions:
                    successorState = gameState.generateSuccessor(agent, action)
                    nodeScore = miniMax(successorState, nextDepth, alpha, beta, nextAgent, nextNodeType)[0]

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

        # Return minimax action
        minimaxAction = miniMax(gameState, depth, alpha, beta, agent, nodeType)[1]
        return minimaxAction

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
