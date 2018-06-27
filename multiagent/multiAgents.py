#coding:utf8
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
        oldFood = currentGameState.getFood().asList()
        #set infinite values for best/worst moves  (初始化两个值：正无穷和负无穷)
        inf = float('inf')
        negInf = float('-inf')
        gPositions = []
        #iterate over ghost states
        for gState in newGhostStates:
          #get the ghost's coordinates
            gPos = gState.getPosition()
            gPositions.append(gPos)
            #if action causes you to safely eat a pellet, that's the best move（判断如果下一个动作能让你吃到豆子，且不被ghost吃掉，则返回正无穷;可用newFood和oldFood这两个列表的长度来判断是否有豆子被吃了，以及用采取吃豆子后的位置和gohost的新位置不一样，按照上面的两种判断，补全如下条件
            if newFood.count(True) != oldFood.count(True) and [newPos != x.getPosition() for x in newGhostStates]:
                return inf
            #if action causes you to die, that's the worst move（如果吃豆人的下一个动作会引起被幽灵吃掉，则返回一个负无穷；判断是否会被幽灵吃掉可利用吃豆人采取动作之后的位置newPos和幽灵的位置gPos是否一致
            elif [newPos != x.getPosition() for x in newGhostStates] and gState.scaredTimer == 0:
                return negInf
        values = []
        #iterate over new food coordinates
        for food in newFood:
            for gPos in gPositions:#get pacman's distance to a pellet and his distance to a ghost, weigh the pellets as more important，请自己定义状态的值：例如用pacman距离豆子距离的远近，越近值越大；以及pacman距离gohost的远近，越远值应该越大，可以综合考虑来定义这样的距离，补全下述函数；例如：manhattanDistance(newPos, food)表示pacman和豆子的距离
                v = manhattanDistance(newPos, food)*2 - manhattanDistance(newPos, gPos)
                values.append(v)
        return max(values)
        #return successorGameState.getScore()

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
        #util.raiseNotDefined()
        def maxValue(state, depth):
            depth=depth+1
            # 若当前状态已经赢了或输了 或者 已经到达了规定的深度
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            # 初始化v为负无穷
            v= float('-Inf')
            # 对每个min分支求max
            for pAction in state.getLegalActions(0):
                v=max(v, minValue(state.generateSuccessor(0, pAction), depth, 1))
            return v
        def minValue(state, depth, ghostNum):
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            v=float('Inf')
            # 对每个max分支求min 其中有多个Ghost 所有多个Ghost分支
            for pAction in state.getLegalActions(ghostNum):
                if ghostNum == gameState.getNumAgents()-1:
                    #所有Ghost的min找完了 开始找下一个max
                    v=min(v, maxValue(state.generateSuccessor(ghostNum, pAction), depth))
                else:
                    #继续下一个Ghost
                    v=min(v, minValue(state.generateSuccessor(ghostNum, pAction), depth, ghostNum+1))
            return v

        # pacman下一个状态可能的行动
        Pacman_Actions = gameState.getLegalActions(0)

        maximum = float('-Inf')
        result = ''

        for action in Pacman_Actions:
            if(action != "Stop"):
                depth = 0
                currentMax = minValue(gameState.generateSuccessor(0, action), depth , 1)
                if currentMax > maximum:
                    maximum=currentMax
                    result =action
        return result

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        def maxValue(state, alpha, beta, depth):
            # 当前深度加一
            depth=depth+1
            # 若当前状态已经赢了或输了 或者 已经到达了规定的深度
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)
            v=float('-Inf')
            # 对每个min分支求max
            for pAction in state.getLegalActions(0):
                if pAction!="Stop":
                    v=max(v, minValue(state.generateSuccessor(0, pAction), alpha, beta, depth, 1))
                    # 若已经比beta要大了 就没有搜索下去的必要了
                    if v >= beta:
                        return v
                    # 更新alpha的值
                    alpha=max(alpha, v)
            return v
        def minValue(state, alpha, beta, depth, ghostNum):
            # 若当前状态已经赢了或输了
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state)
            # 初始化v
            v=float('Inf')
            # 对每个max分支求min 其中有多个Ghost 所有多个Ghost分支
            for pAction in state.getLegalActions(ghostNum):
                if ghostNum == gameState.getNumAgents()-1:
                    # 所有Ghost的min找完了 开始找下一个max
                    v=min(v, maxValue(state.generateSuccessor(ghostNum, pAction), alpha, beta, depth))
                else:
                    # 继续下一个Ghost
                    v=min(v,
                          minValue(state.generateSuccessor(ghostNum, pAction), alpha, beta, depth, ghostNum+1))
                # 若比alpha还要小了 就没搜索的必要了
                if v <= alpha:
                    return v
                # 更新beta的值
                beta=min(beta, v)
            return v
        # pacman下一个状态可能的行动
        pacmanActions=gameState.getLegalActions(0)
        maximum=float('-Inf')
        # 初始化alpha bate
        alpha=float('-Inf')
        beta=float('Inf')
        maxAction=''

        # 针对下一个状态 寻找获胜概率最高的move
        for action in pacmanActions:
            if action!="Stop":
                depth=0
                # 而所有的Ghost希望胜利概率最低的选择
                currentMax=minValue(gameState.generateSuccessor(0, action), alpha, beta, depth, 1)
                if currentMax > maximum:
                    maximum=currentMax
                    maxAction=action
        print maximum
        return maxAction

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

