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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

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
        closest_ghost_position = float("inf")
        for ghostState in newGhostStates:
            ghost_x, ghost_y = ghostState.getPosition()
            ghost_coords = (int(ghost_x), int(ghost_y))

            if ghostState.scaredTimer == 0:
                closest_ghost_position = min(
                    closest_ghost_position,
                    manhattanDistance(ghost_coords, newPos),
                )

        food_list = newFood.asList()
        closest_food_position = 0 if len(food_list) == 0 else float("inf")

        for food in food_list:
            if closest_food_position > manhattanDistance(food, newPos):
                closest_food_position = manhattanDistance(food, newPos)

        return (
            successorGameState.getScore()
            - 10 / (closest_ghost_position + 1)
            - closest_food_position / 5
        )


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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        _, best_action = self.minimax(gameState, 0, self.depth)
        return best_action

    def getLegalActions(self, gameState, agentIndex, depth):
        legal_actions = []
        for action in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, action)

            legal_actions.append(
                (
                    self.minimax(
                        successorState,
                        agentIndex + 1,
                        depth,
                    )[0],
                    action,
                )
            )

        return legal_actions

    def minimax(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)

        agentIndex %= gameState.getNumAgents()
        if agentIndex == gameState.getNumAgents() - 1:
            depth -= 1

        legal_actions = self.getLegalActions(gameState, agentIndex, depth)

        if agentIndex == 0:
            return max(legal_actions)
        else:
            return min(legal_actions)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        _, best_action = self.alphaBetaPruning(
            gameState, 0, self.depth, float("-inf"), float("inf")
        )

        return best_action

    def alphaBetaPruning(self, gameState, agentIndex, depth, alpha, beta):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)
        else:
            legal_actions = gameState.getLegalActions(agentIndex)
            if agentIndex == gameState.getNumAgents() - 1:
                agent = 0
                depth -= 1
            else:
                agent = agentIndex + 1

            if agentIndex == 0:
                return self.getMaxValue(
                    gameState, legal_actions, agent, agentIndex, depth, alpha, beta
                )
            else:
                return self.getMinValue(
                    gameState, legal_actions, agent, agentIndex, depth, alpha, beta
                )

    def getMaxValue(
        self, gameState, legal_actions, agent, agentIndex, depth, alpha, beta
    ):
        max_score, max_action = float("-inf"), Directions.STOP

        for action in legal_actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            successor_score, _ = self.alphaBetaPruning(
                successorState, agent, depth, alpha, beta
            )

            if successor_score > max_score:
                max_action = action
            max_score = max(max_score, successor_score)

            if successor_score > beta:
                return successor_score, action

            alpha = max(alpha, max_score)

        return max_score, max_action

    def getMinValue(
        self, gameState, legal_actions, agent, agentIndex, depth, alpha, beta
    ):
        min_score, min_action = float("inf"), Directions.STOP

        for action in legal_actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            successor_score, _ = self.alphaBetaPruning(
                successorState, agent, depth, alpha, beta
            )

            if successor_score < min_score:
                min_action = action
            min_score = min(min_score, successor_score)

            if successor_score < alpha:
                return successor_score, action

            beta = min(beta, min_score)

        return min_score, min_action


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
        _, best_action = self.expectimaxSearch(gameState, 0, self.depth)
        return best_action

    def expectimaxSearch(self, gameState, agentIndex, depth):
        if gameState.isWin() or gameState.isLose() or depth == 0:
            return (self.evaluationFunction(gameState), Directions.STOP)
        else:
            legal_actions = gameState.getLegalActions(agentIndex)
            if agentIndex == gameState.getNumAgents() - 1:
                agent = 0
                depth -= 1
            else:
                agent = agentIndex + 1

            if agentIndex == 0:
                return self.playAsAgent(
                    gameState, legal_actions, agent, agentIndex, depth
                )
            else:
                return self.playAsGhosts(
                    gameState, legal_actions, agent, agentIndex, depth
                )

    def playAsAgent(self, gameState, legal_actions, agent, agentIndex, depth):
        max_score, max_action = float("-inf"), Directions.STOP

        for action in legal_actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            successor_score, _ = self.expectimaxSearch(successorState, agent, depth)

            if successor_score > max_score:
                max_action = action
            max_score = max(max_score, successor_score)

        return max_score, max_action

    def playAsGhosts(self, gameState, legal_actions, agent, agentIndex, depth):
        expectation_score = 0

        for action in legal_actions:
            successorState = gameState.generateSuccessor(agentIndex, action)
            successor_score, _ = self.expectimaxSearch(successorState, agent, depth)
            expectation_score += successor_score

        # Each action has an equal probability.
        expectation_score /= len(legal_actions)

        return expectation_score, Directions.STOP


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: The pacman is awarded relative to it's position with the position of the closest food pellet.
    The pacman is negatively awarded if it approaches a ghost, and thus taught to prioritize navigating to the nearest
    food pellet or the ghost scare pellet. Finally, the pacman is dynamically rewarded a value higher than it would have gained
    by consuming a food pellet to prioritize moving towards a ghost scare pellet when given the chance. The choice of using random.randint
    produced better results than a fixed value.
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()

    pellet_multiplier = random.randint(2, 10)

    score = currentGameState.getScore()

    food_list = newFood.asList()
    closest_food_position = 0 if len(food_list) == 0 else float("inf")

    for food in food_list:
        if closest_food_position > manhattanDistance(food, newPos):
            closest_food_position = manhattanDistance(food, newPos)

    if closest_food_position > 0:
        score += 1 / closest_food_position

    for ghostState in newGhostStates:
        ghost_x, ghost_y = ghostState.getPosition()
        ghost_coords = (int(ghost_x), int(ghost_y))

        distance_to_ghost = manhattanDistance(newPos, ghost_coords)

        if distance_to_ghost > 0:
            if ghostState.scaredTimer > 0:
                score += pellet_multiplier / distance_to_ghost
            else:
                score -= 1 / distance_to_ghost
        else:
            return 0

    return score


# Abbreviation
better = betterEvaluationFunction
