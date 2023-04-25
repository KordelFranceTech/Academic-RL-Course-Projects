# sarsa_algorithm.py
# Kordel France
########################################################################################################################
# This file establishes the specification for the SARSA algorithm for reinforcement learning.
########################################################################################################################

from random import shuffle, random, randint
import numpy as np
import os
import time
from copy import deepcopy
from Lab5 import config
if config.DEMO_MODE and config.DEBUG_MODE:
    from Lab5.RL import demo_config_sarsa as model_config
else:
    from Lab5 import model_config as model_config

model_config = model_config

"""
State space:
    - all values of lateral position, x
    - all values of longitudinal position, y
    - all values of lateral velocity, xx
    - all values of longitudinal velocity, yy

Action space:
    - all values of lateral acceleration, xxx
    - all values of longitudinal acceleration, yyy

Model parameters:
    - s = initial state
    - a = initial action
    - s_prime = next state
    - a_prime = next action   
    - q = current q-value
    - q_prime = next q-value
"""

# configuration parameters
TRACK_NAME: str = model_config.TRACK_NAME
AGENT_SYMBOL: str = config.AGENT_SYMBOL
STATE_INITIAL: str = config.STATE_INITIAL
STATE_TERMINAL: str = config.STATE_TERMINAL
STATE_COLLISION: str = config.STATE_COLLISION
STATE_TRACK: str = config.STATE_TRACK
LAP_COUNT: int = model_config.LAP_COUNT
VELOCITY_MIN: int = model_config.VELOCITY_MIN      # velocity lower bound
VELOCITY_MAX: int = model_config.VELOCITY_MAX       # velocity upper bound
GAMMA: float = model_config.GAMMA          # discount rate
NU: float = model_config.NU       # learning rate
ACCELERATION_RATE: float = model_config.ACCELERATION_RATE  # probability that the acceleration control succeeds
ACCELERATION_MISFIRE_RATE: float = 1 - ACCELERATION_RATE  # probability that the acceleration control fails
EPOCHS: int = model_config.EPOCHS
RESET_AFTER_CRASH: bool = model_config.RESET_AFTER_CRASH
EPISODES: int = model_config.EPISODES
Q_STABILITY_ERROR: float = model_config.Q_STABILITY_ERROR
EPOCH_THRESHOLD: int = model_config.EPOCH_THRESHOLD
UPDATE_STEPS: int = model_config.UPDATE_STEPS
state_space_velocity = model_config.STATE_SPACE_VELOCITY
action_space = model_config.ACTION_SPACE


def reset_hyperparameters():
    """
    Resets configuration and hyperparameters so that a new configuration may be passed immediately after.
    """
    global TRACK_NAME
    global AGENT_SYMBOL
    global STATE_INITIAL
    global STATE_TERMINAL
    global STATE_COLLISION
    global STATE_TRACK
    global LAP_COUNT
    global VELOCITY_MIN
    global VELOCITY_MAX
    global GAMMA
    global NU
    global ACCELERATION_RATE
    global ACCELERATION_MISFIRE_RATE
    global EPOCHS
    global RESET_AFTER_CRASH
    global EPISODES
    global Q_STABILITY_ERROR
    global EPOCH_THRESHOLD
    global UPDATE_STEPS
    global state_space_velocity
    global action_space

    TRACK_NAME = model_config.TRACK_NAME
    AGENT_SYMBOL = config.AGENT_SYMBOL
    STATE_INITIAL = config.STATE_INITIAL
    STATE_TERMINAL = config.STATE_TERMINAL
    STATE_COLLISION = config.STATE_COLLISION
    STATE_TRACK = config.STATE_TRACK
    LAP_COUNT = model_config.LAP_COUNT
    VELOCITY_MIN = model_config.VELOCITY_MIN  # velocity lower bound
    VELOCITY_MAX = model_config.VELOCITY_MAX  # velocity upper bound
    GAMMA = model_config.GAMMA  # discount rate
    NU = model_config.NU  # learning rate
    ACCELERATION_RATE = model_config.ACCELERATION_RATE  # probability that the acceleration control succeeds
    ACCELERATION_MISFIRE_RATE = 1 - ACCELERATION_RATE  # probability that the acceleration control fails
    EPOCHS = model_config.EPOCHS
    RESET_AFTER_CRASH = model_config.RESET_AFTER_CRASH
    EPISODES = model_config.EPISODES
    Q_STABILITY_ERROR = model_config.Q_STABILITY_ERROR
    EPOCH_THRESHOLD = model_config.EPOCH_THRESHOLD
    UPDATE_STEPS = model_config.UPDATE_STEPS
    state_space_velocity = model_config.STATE_SPACE_VELOCITY
    action_space = model_config.ACTION_SPACE


def construct_environment(input_file):
    """
    Constructs the agent environment (the racetrack) from the input file.
    :param input_file: str - the name of the file to build env from
    """
    env_space = []

    # configured specifically for demonstration purposes
    if config.DEMO_MODE and config.DEBUG_MODE:
        with open(f'io_files/{input_file}', 'r') as track_file:
            env_space_data = track_file.readlines()
        track_file.close()

        for index, track_line in enumerate(env_space_data):
            if index > 0:
                track_line = track_line.strip()
                if track_line == '':
                    continue
                env_line = [x for x in track_line]
                # env_space.append(env_line)
                env_space.append([x for x in track_line])

    # for monte carlo simulation or other analysis
    else:
        with open(f'{config.IO_DIRECTORY}/{input_file}', 'r') as track_file:
            env_space_data = track_file.readlines()
        track_file.close()

        for index, track_line in enumerate(env_space_data):
            if index > 0:
                track_line = track_line.strip()
                if track_line == '':
                    continue
                env_line = [x for x in track_line]
                # env_space.append(env_line)
                env_space.append([x for x in track_line])

    return env_space


def display_current_state(env_data, state=[0, 0]):
    """
    Prints the current state of the agent to the console - basically the whole track with the car position.
    :param env_data: list - the racetrack
    :param state: list - an initializing state
    """
    # get current state
    current_state = env_data[state[0]][state[1]]
    env_data[state[0]][state[1]] = AGENT_SYMBOL

    # pause console printing for readability
    if config.DEMO_MODE:
        time.sleep(0.3)
    os.system('cls')

    # build the ractrack and place the car
    for index_i in range(0, len(env_data)):
        env_str: str = ''
        env_line = env_data[index_i]
        for index_j in range(0, len(env_line)):
            env_str += env_line[index_j]
        if config.DEMO_MODE:
            print(env_str)
    env_data[state[0]][state[1]] = current_state


def get_new_initial_state(env_data):
    """
    Gets a new starting position on the track.
    :param env_data: list - the racetrack
    """
    initial_states: list = []
    for y, row in enumerate(env_data):
        for x, col in enumerate(row):
            if col == STATE_INITIAL:
                initial_states += [(y, x)]

    # select a random position on the starting line
    shuffle(initial_states)
    return initial_states[0]


def update_state_position(position, velocity_vector):
    """
    Update x- and y- components of the agent position for new state given velocity vector.
    :param position: list - the x- and y- coords of position
    :param velocity_vector: list = the velocity vector to move position by
    """
    y, x = position[0], position[1]
    yy, xx = velocity_vector[0], velocity_vector[1]
    x_prime = x + xx
    y_prime = y + yy
    return y_prime, x_prime


def update_state_velocity(velocity_vector, acc_vector, velocity_min, velocity_max):
    """
    Update x- and y- components of the agent velocity for new state given acceleration vector.
    :param velocity_vector: list - the current velocity
    :param acc_vector: list - the acceleration vector to alter velocity by
    :param velocity_min: int - minimum allowed velocity
    :param velocity_max: int - maximum allowed velocity
    """
    yy = velocity_vector[0] + acc_vector[0]
    xx = velocity_vector[1] + acc_vector[1]
    if xx < velocity_min:
        xx = velocity_min
    if xx > velocity_max:
        xx = velocity_max
    if yy < velocity_min:
        yy = velocity_min
    if yy > velocity_max:
        yy = velocity_max

    return yy, xx


def check_collision_with_bresenham_algorithm(y, x, y_prime, x_prime, env_data):
    """
    Use Bresenham algorithm as suggested by project documentation to determine collision
    :param y: int -current y position
    :param x: int - current x position
    :param y_prime: int - y pos of next state
    :param x_prime: int - x pos of next state
    :env_data: list - the racetrack
    """
    # init a points list with a slope and slope error
    points_list: list = []
    m_prime = 2 * (y_prime - y)
    m_prime_error = m_prime - (x_prime - x)
    y_line = y_prime

    # step among the points and calculate slope
    for x_line in range(x, x_prime + 1):
        points_list.append((x_line, y_line))
        m_prime_error += m_prime
        if m_prime_error >= 0:
            y_line += 1
            m_prime_error = m_prime_error - 2 * (x_prime - x)

    # if slope determines a collision, return and update q-value
    if env_data[y][x] == STATE_COLLISION:
        if config.DEBUG_MODE and config.DEMO_MODE:
            print(f'COLLISION')
        return



def search_for_actionable_track_spaces(env_data, y_crash, x_crash, yy=0, xx=(0),
                                       open_positions=[STATE_TRACK, STATE_INITIAL, STATE_TERMINAL]):
    """
    Local search for positions that are nearby and not boundaries
    :param env_data: list - the racetrack
    :param y_crash: int - y pos of crash
    :param x_crash: int - x pos of crash
    :param yy: int - y velocity
    :param xx: int - x velocity
    :param open_positions: list - list of states that indicate open position types
    """
    # define boundaries for radius
    env_lat = len(env_data)
    env_lon = len(env_data[0])
    max_radius = max(env_lat, env_lon)

    # do local search over radius
    for radius in range(0, max_radius):
        if yy == 0:
            y_span = range(-radius, radius + 1)
        elif yy < 0:
            y_span = range(0, radius + 1)
        else:
            y_span = range(-radius, 1)

        # check each candidate within y-component radius and make sure close enough
        for dy in y_span:
            y = y_crash + dy
            x_span = radius - abs(dy)

            if xx == 0:
                x_span = range(x_crash - x_span, x_crash + x_span + 1)
            elif xx < 0:
                x_span = range(x_crash, x_crash + x_span + 1)
            else:
                x_span = range(x_crash - x_span, x_crash + 1)

            # check each candidate within x-component radius and make sure close enough
            # if so, go to that position
            for x in x_span:
                if y < 0 or y >= env_lat:
                    continue
                if x < 0 or x >= env_lon:
                    continue
                if env_data[y][x] in open_positions:
                    return (y, x)
    return


def act(y_prev, x_prev, yy_prev, xx_prev, acc_vector, env_data, is_deterministic=(False),
        reset_after_crash: bool = False):
    """
    Allows agent to act within its environment, showing some exploration ability as well as policy adherence.
    :param y_prev: int - previous y pos
    :param x_prev: int - prvious x pos
    :param yy_prev: int - previous y velocity
    :param xx_prev: int - previous x velocity
    :param acc_vector: list - x- and y- components of acceleration for action
    :param env_data: list - the racetrack
    :param is_deterministic: bool - flag that allows me to turn on and off explorability
    :param reset_after_crash: bool - flat that allows me to indicate what happens after the crash occurs
    """

    # this induces some explorability for the agent
    # explorability randomly occurs
    # get a random number and if greater than acceleration rate, randomly accelerate
    # otherwise, adhere to policy and select action from there
    # definitely exists opportunity to optimize
    if not is_deterministic:
        if random() > ACCELERATION_RATE:
            #### BREAKPOINT ###
            acc_vector = (0, 0)
            # random acceleration vector has issues at times
            # opportunity to optimize
            # xxx = randint(-1, 1)
            # yyy = randint(-1, 1)
            # acc_vector(xxx, yyy)

    # the previous velocity vector
    velocity_vector = (yy_prev, xx_prev)

    # position of the nex state
    yy_prime, xx_prime = update_state_velocity(velocity_vector, acc_vector, VELOCITY_MIN, VELOCITY_MAX)

    # position of previous state
    position = (y_prev, x_prev)
    y, x = update_state_position(position, velocity_vector)

    # find actionable position within proposed position
    y_prime, x_prime = search_for_actionable_track_spaces(env_data, y, x, yy_prime, xx_prime)

    # check to make sure we aren't selecting to travel to same point we are on
    if y_prime != y or x_prime != x:
        if reset_after_crash and env_data[y_prime][x_prime] != STATE_TERMINAL:

            # either we crashed and crash policy indicates reset, or we reached the finish line
            y_prime, x_prime = get_new_initial_state(env_data)
        yy_prime, xx_prime = 0, 0

    return y_prime, x_prime, yy_prime, xx_prime


def update_policy_given_q(cols, rows, state_space_velocity, q, action_space):
    """
    Perform policy update and adjust the console output to reflect the policy update visually.
    :param cols: int - column count for track file
    :param rows: int - row count for track file
    :param state_space_velocity: list - the list of possible state values for velocity
    :param q: float - the current q-value (reward)
    :param action_space: list - a list of possible actions to take
    """
    # init empty policy
    policy = {}
    for y in range(rows):
        for x in range(cols):
            for yy in state_space_velocity:
                for xx in state_space_velocity:
                    # set the policy value at this scalar location
                    policy[(y, x, yy, xx)] = action_space[np.argmax(q[y][x][yy][xx])]
                    # state_vector = (y, x, yy, xx)
                    # q_value = np.argmax(q[y][x][yy][xx])
                    # policy[state_vector] = action_space[q_value]

    # return the updated policy
    return (policy)


########################################################################################################################
########################################################################################################################
# MARK: -  MAIN LEARNING ALGORITHM BEGIN
########################################################################################################################
########################################################################################################################

def perform_sarsa(env_data, reset_after_crash: bool = False, reward: float = 0.0, epochs=(EPOCHS), episodes=EPISODES):
    """
    Implements the full SARSA learning algorithm.
    :param env_data: list - the racetrack
    :param reset_after_crash: bool - crash policy - whether the car should return to the start position after a crash
    :param reward: float - the reward or q-value
    :param epochs: int - number of epochs to train for
    :param episodes: int - number of iterations per epoch
    """
    # define the track file dimensionality
    rows = len(env_data)
    cols = len(env_data[0])

    # init the q-value
    q_value = [[[[[random() for _ in action_space] for _ in state_space_velocity] for _ in (state_space_velocity)] for _ in line] for line in env_data]

    # build and init state, action pairs for q-values
    for y in range(0, rows):
        for x in range(0, cols):
            if env_data[y][x] == STATE_TERMINAL:
                for yy in state_space_velocity:
                    for xx in state_space_velocity:
                        for action_i, action in enumerate(action_space):
                            q_value[y][x][yy][xx][action_i] = reward

    # begin training
    for epoch in range(0, epochs):
        for y in range(0, rows):
            for x in range(0, cols):
                # if finish line crossed, agent receives total cumulative reward
                if env_data[y][x] == STATE_TERMINAL:
                    q_value[y][x] = [[[reward for _ in action_space] for _ in (state_space_velocity)] for _ in state_space_velocity]

        # initialize all state positions
        y = np.random.choice(range(0, rows))
        x = np.random.choice(range(0, cols))

        # initialize all state velocities
        yy = np.random.choice(state_space_velocity)
        xx = np.random.choice(state_space_velocity)

        #### BREAKPOINT 1 ###
        # choose action a using policy given from greedy Q
        action = np.argmax(q_value[y][x][yy][xx])

        # perform updates to q-values for each episode
        for episode in range(0, episodes):
            if env_data[y][x] == STATE_TERMINAL:
                break
            if env_data[y][x] == STATE_COLLISION:
                break

            #### BREAKPOINT 2 ###
            # if there is a collision check collision policy and restart
            check_collision_with_bresenham_algorithm(y, x, 0, 0, env_data)

            #### BREAKPOINT 3 ###
            # take action and get new state position, velocity
            y_prime, x_prime, yy_prime, xx_prime = act(y, x, yy, xx, action_space[action], env_data, reset_after_crash=reset_after_crash)
            reward = -1

            # choose action a' using policy given from greedy Q
            action_prime = np.argmax(q_value[y_prime][x_prime][yy_prime][xx_prime])

            # update Q(s, a)
            # q-value from previous episode
            q_value_prev = q_value[y][x][yy][xx][action]
            # max cumulative reward
            q_value_max = max(q_value[y_prime][x_prime][yy_prime][xx_prime])

            #### BREAKPOINT 4 ###
            # generation of <state, action, reward, state, action> tuple
            q_value[y][x][yy][xx][action] = ((1 - NU) * q_value_prev + NU * (reward + GAMMA * q_value_max))

            # quality control
            if config.DEMO_MODE:
                print(f'\tprevious q-value: {q_value_prev}\n\tupdated q-value: {q_value}')

            #### BREAKPOINT 2 ###
            # set new states, actions as current states, actions
            y, x, yy, xx = y_prime, x_prime, yy_prime, xx_prime
            action = action_prime

    return (update_policy_given_q(cols, rows, state_space_velocity, q_value, action_space))

########################################################################################################################
########################################################################################################################
# MARK: -  MAIN LEARNING ALGORITHM END
########################################################################################################################
########################################################################################################################


def initialize_race(env_data, policy, reset_after_crash, update_steps=UPDATE_STEPS):
    """
    Sets up the race environment and configures the algorithm for learning based on hyperparameters and config file.
    :param env_data: list - the racetrack
    :param reset_after_crash: bool - the crash policy
    :param update_steps: bool - the number of update steps to allow
    :return update_steps: int - the number of update steps left to complete
    """
    # create a new copy of the console output for alteration and reprint
    env_view = deepcopy(env_data)

    # initialize car at starting line
    position_init = get_new_initial_state(env_view)

    # define kinematics
    y, x = position_init
    yy, xx = 0, 0
    clock_count: int = 0

    # build the UI and get current action to take from policy
    for index_i in range(0, update_steps):
        display_current_state(env_view, state=[y, x])
        action = policy[(y, x, yy, xx)]

        if env_data[y][x] == STATE_TERMINAL:
            return index_i

        y, x, yy, xx = act(y, x, yy, xx, action, env_data, reset_after_crash=reset_after_crash)

        if xx == 0 and yy == 0:
            clock_count += 1
        else:
            clock_count = 0

        if clock_count == 5:
            return update_steps

    return update_steps


def train_sarsa_algorithm(explainability:bool=False):
    """
    Driver function for the SARSA algorithm
    :param explainability: bool - a flag that allows all agent SARSA tuples to be saved for debugging and QC
    """
    # reset the spec file in case there was a model run before with a different config
    reset_hyperparameters()
    print('starting racetrack RL agent for SARSA algorithm')
    epochs = EPOCHS
    epoch_list: list = []
    step_list: list = []
    racetrack_name = f'{TRACK_NAME}'
    racetrack = construct_environment(racetrack_name)
    track_name = TRACK_NAME

    # begin training
    while (epochs < EPOCH_THRESHOLD):
        print(f'\n\n\nCURRENT EPOCH: {epochs}')
        step_count: int = 0

        policy = perform_sarsa(racetrack, reset_after_crash=RESET_AFTER_CRASH, epochs=epochs)
        for lap in range(0, LAP_COUNT):
            step_count += initialize_race(racetrack, policy, reset_after_crash=(RESET_AFTER_CRASH))

        # show how many epochs have passed
        print(f'number of training iterations: {epochs}')
        if RESET_AFTER_CRASH == 1:
            print(f'crash policy = reset after crash')
        else:
            print(f'crash policy = continue exploring after crash')

        # show number of update steps so user knows if algo is converging
        print(f'average # of steps car needs to take before finish line: {step_count / LAP_COUNT} steps\n')
        print(f'car is training...\n')
        epoch_list.append(epochs)
        step_list.append(step_count / LAP_COUNT)

        # useful for visualizing state transitions in the console
        if config.DEMO_MODE:
            time.sleep(5)

        # a file that keeps track of the model statistics during training
        stats_file = track_name
        stats_file += '_'
        stats_file += f'sarsa_epoch'
        stats_file += f'{epochs}_crash'
        stats_file += f'{RESET_AFTER_CRASH}_stats.txt'

        # write stats to an output file
        output_file = open(stats_file, 'w')
        output_file.write(f'_______________________________________________________________\n')
        output_file.write(f'sarsa summmary stats:\n')
        output_file.write(f'_______________________________________________________________\n')
        output_file.write(track_name)
        output_file.write(f'# epochs: {epochs}')
        if RESET_AFTER_CRASH:
            output_file.write(f'\ncrash policy = reset after crash\n')
        else:
            output_file.write(f'\ncrash policy = continue exploring after crash\n')
        output_file.write(f'cost & average number of steps vehicle took before finish line: {step_count / LAP_COUNT} steps')

        # for more QC, the explainability flag will allow the model to save all SARSA tuples
        if explainability:
            controls_log = f'output/{track_name}'
            controls_log += '_'
            controls_log += f'sarsa_epoch'
            controls_log += f'{epochs}_crash'
            controls_log += f'{RESET_AFTER_CRASH}_explainability.txt'

            if epochs <= 5:
                log_file = open(controls_log, 'w')
                log_file.write(str(policy))
                log_file.close()

        output_file.close()

        # increase epoch count if algo hasn't converged yet
        if epochs == 500000:
            epochs += 500000
        else:
            epochs += 500000

    # reset model config for next run
    reset_hyperparameters()

    # return a list of epochs and convergence steps for metrics logging
    # mostly used for monte carlo
    return epoch_list, step_list


if config.DEMO_MODE:
    train_sarsa_algorithm()





















