# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A ValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
        Your value iteration agent should take an mdp on
        construction, run the indicated number of iterations
        and then act according to the resulting policy.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state, action, nextState)
            mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        while self.iterations != 0:
            self.iterations -= 1
            q_values = util.Counter()  # store new value of a state
            is_state_visited = util.Counter()  # store whether a state has been updated

            for state in self.mdp.getStates():
                next_action = self.getAction(state)
                if next_action:
                    q_values[state] = self.getQValue(state, next_action)
                    is_state_visited[state] = True

            for state in self.mdp.getStates():
                if is_state_visited[state]:
                    self.values[state] = q_values[state]

    def getValue(self, state):
        """
        Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transition_states = self.mdp.getTransitionStatesAndProbs(state, action)
        q_value = 0.0

        for next_state, probability in transition_states:
            q_value += probability * (
                self.mdp.getReward(state, action, next_state)
                + self.discount * self.getValue(next_state)
            )

        return q_value

    def computeActionFromValues(self, state):
        """
        The policy is the best action in the given state
        according to the values currently stored in self.values.

        You may break ties any way you see fit.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions_for_state = self.mdp.getPossibleActions(state)
        if not actions_for_state:
            return None

        next_action, next_action_reward = "", float("-inf")

        for action in actions_for_state:
            reward = self.getQValue(state, action)

            if next_action_reward < reward:
                next_action_reward = reward
                next_action = action

        return next_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    An AsynchronousValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs cyclic value iteration
    for a given number of iterations using the supplied
    discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
        Your cyclic value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy. Each iteration
        updates the value of only one state, which cycles through
        the states list. If the chosen state is terminal, nothing
        happens in that iteration.

        Some useful mdp methods you will use:
            mdp.getStates()
            mdp.getPossibleActions(state)
            mdp.getTransitionStatesAndProbs(state, action)
            mdp.getReward(state)
            mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        mdp_states = self.mdp.getStates()
        state_index = 0

        while self.iterations != 0:
            self.iterations -= 1
            current_state = mdp_states[state_index % len(mdp_states)]
            state_index += 1

            next_action = self.getAction(current_state)
            if next_action:
                self.values[current_state] = self.getQValue(current_state, next_action)


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
    * Please read learningAgents.py before reading this.*

    A PrioritizedSweepingValueIterationAgent takes a Markov decision process
    (see mdp.py) on initialization and runs prioritized sweeping value iteration
    for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
        Your prioritized sweeping value iteration agent should take an mdp on
        construction, run the indicated number of iterations,
        and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        predecessor_states = self.mdp.getStates()
        pr_queue = util.PriorityQueue()

        adjacent_matrix = []
        state_indices = util.Counter()

        for index, state_x in enumerate(predecessor_states):
            adjacent_list = set()

            for state_y in predecessor_states:
                legal_actions = self.mdp.getPossibleActions(state_y)
                for action in legal_actions:
                    next_states = self.mdp.getTransitionStatesAndProbs(state_y, action)

                    for next_state, transition_prob in next_states:
                        if next_state == state_x and transition_prob > 0.0:
                            adjacent_list.add(state_y)

            adjacent_matrix.append(adjacent_list)
            state_indices[state_x] = index

        q_values = util.Counter()
        for state in predecessor_states:
            if self.mdp.isTerminal(state):
                continue

            if self.computeActionFromValues(state):
                updated_q_value = self.computeQValueFromValues(
                    state, self.computeActionFromValues(state)
                )
                q_values[state] = updated_q_value
                diff = abs(self.getValue(state) - updated_q_value)
                pr_queue.push(state, -diff)
            else:
                q_values[state] = self.getValue(state)

        for _ in range(self.iterations):
            if pr_queue.isEmpty():
                break

            next_state = pr_queue.pop()
            if not self.mdp.isTerminal(next_state):
                self.values[next_state] = q_values[next_state]

            for predecessor in adjacent_matrix[state_indices[next_state]]:
                if self.computeActionFromValues(predecessor):
                    updated_q_value = self.computeQValueFromValues(
                        predecessor, self.computeActionFromValues(predecessor)
                    )
                    diff = abs(self.getValue(predecessor) - updated_q_value)
                    q_values[predecessor] = updated_q_value

                    if diff > self.theta:
                        pr_queue.update(predecessor, -diff)
