import sys
import numpy as np


UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
INITIAL_STATE: int = 8
GOAL_STATE: tuple = (0, 3)
NOMINAL_REWARD: int = -1
GOAL_REWARD: int = 25
PIT_REWARD: int = -25


def cum_sample(prob_n, np_random: np.random.Generator):
    """
    Return a sample from a categorical distribution where
    each row specifies class probabilities.
    :param prob_n: float - the size of the distribution
    :param np_random: float - the random seed
    :return: float - the sample
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return np.argmax(csprob_n > np_random)


def build_pit(shape):
    """
    Define where the pit is for the robot.
    :param shape: tuple - the shape of the current map
    :return: np.array - a new map with the pit defined
    """
    _pit = np.zeros(shape, dtype=bool)
    _pit[1, -1:] = True
    return _pit


def build_block(shape):
    """
    Define where the block (non-occupying state) is for the robot.
    Note: when we hit the block, we go back to the previous state and incur cost of 0.
    :param shape: tuple - the shape of the current map
    :return: np.array - a new map with the block defined
    """
    _block = np.zeros(shape, dtype=bool)
    _block[1, -2:-1] = True
    return _block


class RobotGame():

    def __init__(self):
        self.shape = (3, 4)
        self.start_state_index = INITIAL_STATE
        self.nS = np.prod(self.shape)
        self.nA = 4

        # Define the pit
        self._pit = np.zeros(self.shape, dtype=bool)
        self._pit = build_pit(self.shape)

        # Define the block
        self._block = np.zeros(self.shape, dtype=bool)
        self._block = build_block(self.shape)

        # Compute transition probabilities
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._compute_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._compute_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._compute_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._compute_transition_prob(position, [0, -1])

        # Define initial state
        self.initial_state = np.zeros(self.nS)
        self.initial_state[self.start_state_index] = 1.0

        # Define state & action spaces
        self.observation_space = [int(i) for i in range(self.nS)]
        self.action_space = [int(i) for i in range(self.nA)]

    def _clip_boundaries(self, coord: np.ndarray) -> np.ndarray:
        """
        Prevent the agent from transitioning to states outside of the map..
        :param coord: np.ndarray - the state the agent wants to transition to.
        :return: np.array - the clipped state if outside of the map
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _compute_transition_prob(self, current, delta):
        """
        Compute the probability of transitioning from current state to next state.
        :param current: int - the current state occupied by the agent.
        :param delta: int - the state the agent wants to transition to
        :return: float - the probability from moving to delta from current
        """
        current_position = np.ravel_multi_index(tuple(current), self.shape)
        new_position = np.array(current) + np.array(delta)
        new_position = self._clip_boundaries(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)

        if self._pit[tuple(new_position)]:
            return [(1.0, self.start_state_index, PIT_REWARD, False)]

        if self._block[tuple(new_position)]:
            return [(1.0, current_position, 0, False)]

        is_done = tuple(new_position) == GOAL_STATE
        if is_done:
            return [(1.0, new_state, GOAL_REWARD, is_done)]
        return [(1.0, new_state, NOMINAL_REWARD, is_done)]


    def reset(self, return_info: bool = False):
        """
        Resets the environment after episode termination.
        :param return_info: bool - flag indicating whether or not to return probability
        :return: dict - the starting state and optional probability
        """
        self.s = cum_sample(self.initial_state, np.random.random())
        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def _render(self):
        """
        Renders the map showing the starting (S) and terminal states (T, X)
        :return: nil
        """
        outfile = sys.stdout
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if position == (0, 3):
                output = " T "
            elif s == self.start_state_index:
                output = " S "
            elif self._pit[position]:
                output = " X "
            else:
                output = " o "
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

    def get_number_of_actions(self):
        """
        Return the number of all possible actions that the agent can execute.
        :return: int - the number of all possible actions
        """
        return self.nA

    def get_number_of_states(self):
        """
        Return the number of all possible states that the agent can occupy.
        :return: int - the number of all possible states
        """
        return self.nS - 1

    def get_state(self):
        """
        Return the state that the agent currently occupies.
        :return: int - the currently occupied state
        """
        return self.s

    def execute_action(self, a):
        """
        Move from state s to next state s' with action a.
        :param a: int - the action that the agent should execute
        :return: tuple - a tuple of the next state s', the reward, and whether terminal state reached.
        """
        transitions = self.P[self.s][a]
        i = cum_sample([t[0] for t in transitions], np.random.random())
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return int(s), r, d
