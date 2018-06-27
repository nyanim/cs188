#coding:utf8
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

    search_stack = util.Stack()
    successors = problem.getSuccessors(problem.getStartState())

    for ea in successors:
        search_stack.push(ea)
    find_goal = False

    path_actions = []

    visited_pos = set()
    visited_pos.add(problem.getStartState())

    #开始遍历：如果栈不空，且没有到达目标结点(请填充如下两个条件)：
    while not search_stack.isEmpty() and not find_goal:
        choice = search_stack.pop()
        print choice
        if not problem.isGoalState(choice[0]):
        # 如果该节点没被访问
            if choice[0] not in visited_pos:
                visited_pos.add(choice[0])
                path_actions.append(choice)
                #filter的意思是对sequence中的所有item依次执行 function(item)
            choice_successors = filter(lambda v: v[0] not in visited_pos, problem.getSuccessors(choice[0])) 

            if not len(choice_successors):
                path_actions.pop(-1)
                if path_actions:
                    search_stack.push(path_actions[-1])
            else:
                for ea in choice_successors:
                    search_stack.push(ea)
        else:
            path_actions.append(choice)
            visited_pos.add(choice[0])
            find_goal = True

    return [ea[1] for ea in path_actions]
    #util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"

    #初始化一个队列
    queue = util.PriorityQueueWithFunction(len)

    #使用一个list保存已探索过的位置
    explored = []
    queue.push([(problem.getStartState(), "Stop" , 0)])

    #当队列非空时
    while not queue.isEmpty():
        #取队头
        path = queue.pop()

        s = path[len(path)-1]
        s = s[0]
        #判断是否为目标地点
        if problem.isGoalState(s):
            return [x[1] for x in path][1:]
        #如果当前位置背被探索过，则入队
        if s not in explored:
            explored.append(s)
        #遍历后继节点
        for successor in problem.getSuccessors(s):
            if successor[0] not in explored:
                successorPath = path[:]
                successorPath.append(successor)
                queue.push(successorPath)
    return []
    #util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    #util.raiseNotDefined()
    # 新建一个队列
    queue = util.PriorityQueue()
    # 初始状态入队
    queue.push(problem.getStartState(),0)
    explored = []
    paths = {}
    # 代价
    g = {}
    paths[problem.getStartState()] = []
    g[problem.getStartState()] = 0

    # 当队列非空时
    while not queue.isEmpty():
        # 队头出队
        s = queue.pop()
        # s是表示当前位置的坐标的tuple,eg.(13, 5)
        # 判断是否为终点
        if problem.isGoalState(s):
            return paths[s]
        # 加入已探索队列
        explored.append(s)
        # 遍历后继节点
        for successor in problem.getSuccessors(s):
            successorState = successor[0]
            move = successor[1]
            cost = successor[2]
            # 检查当前节点未被探索过
            if successorState not in explored:
                paths[successorState] = list(paths[s]) + [move]
                g[successorState] = g[s] + cost
                # 使用g和h计算优先级并入队
                queue.push(successorState, g[successorState])
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def Heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    # 启发函数，计算当前与终点的曼哈顿距离
    return util.manhattanDistance(state, problem.goal)

def euclideanHeuristic(position, problem):
    xy1 = position
    xy2 = problem.goal
    return ((xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2) ** 0.5

def aStarSearch(problem, heuristic=euclideanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # 新建一个队列
    queue = util.PriorityQueue()
    # 初始状态入队
    queue.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    explored = []
    paths = {}
    # 代价
    g = {}
    paths[problem.getStartState()] = []
    g[problem.getStartState()] = 0

    # 当队列非空时
    while not queue.isEmpty():
        # 队头出队
        s = queue.pop()
        # s是表示当前位置的坐标的tuple,eg.(13, 5)
        # 判断是否为终点
        if problem.isGoalState(s):
            return paths[s]
        # 加入已探索队列
        explored.append(s)
        # 遍历后继节点
        for successor in problem.getSuccessors(s):
            successorState = successor[0]
            move = successor[1]
            cost = successor[2]
            # 检查当前节点未被探索过
            if successorState not in explored:# and is_best(queue, g, g[s]+cost, successorState):
                paths[successorState] = list(paths[s]) + [move]
                g[successorState] = g[s] + cost
                # 使用g和h计算优先级并入队
                queue.push(successorState, heuristic(successorState, problem) + g[successorState])
    return []

def greedySearch(problem, heuristic=euclideanHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    # 新建一个队列
    queue = util.PriorityQueue()
    # 初始状态入队
    queue.push(problem.getStartState(), heuristic(problem.getStartState(), problem))
    explored = []
    paths = {}
    paths[problem.getStartState()] = []

    # 当队列非空时
    while not queue.isEmpty():
        # 队头出队
        s = queue.pop()
        # s是表示当前位置的坐标的tuple,eg.(13, 5)
        # 判断是否为终点
        if problem.isGoalState(s):
            return paths[s]
        # 加入已探索队列
        explored.append(s)
        # 遍历后继节点
        for successor in problem.getSuccessors(s):
            successorState = successor[0]
            move = successor[1]
            if successorState not in explored:#and is_best(queue, g, g[s]+cost, successorState):
                paths[successorState] = list(paths[s]) + [move]
                queue.push(successorState, heuristic(successorState, problem))
    return []

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
greedy = greedySearch
