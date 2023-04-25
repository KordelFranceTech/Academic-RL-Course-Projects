# demo_config_td_lambda.py
# Kordel France
########################################################################################################################
# This file contains hyperparameters to establish a reinforcement learning agent
########################################################################################################################

import Lab5.config as config

TRACK_NAME: str = 'L-track.txt'
AGENT_SYMBOL: str = config.AGENT_SYMBOL
STATE_INITIAL: str = config.STATE_INITIAL
STATE_TERMINAL: str = config.STATE_TERMINAL
STATE_COLLISION: str = config.STATE_COLLISION
STATE_TRACK: str = config.STATE_TRACK
LAP_COUNT: int = 3
# LAP_COUNT: int = 1
VELOCITY_MIN: int = -5          # velocity lower bound
VELOCITY_MAX: int = 5           # velocity upper bound
GAMMA: float = 0.9              # discount rate
LAMBDA: float = 0.1             # for td-lambda
NU: float = 0.2                 # learning rate
THETA: float = 0.2              # exploration rate
ACCELERATION_RATE: float = 0.8  # probability that the acceleration control succeeds
ACCELERATION_MISFIRE_RATE: float = 1 - ACCELERATION_RATE  # probability that the acceleration control fails
EPOCHS: int = 500000
# EPOCHS: int = 50
RESET_AFTER_CRASH: bool = True
EPISODES: int = 500
Q_STABILITY_ERROR: float = 0.001
EPOCH_THRESHOLD: int = 3000000
# EPOCH_THRESHOLD: int = 1000
UPDATE_STEPS: int = 100
STATE_SPACE_VELOCITY = range(VELOCITY_MIN, VELOCITY_MAX + 1)
ACTION_SPACE = [(-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 0),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1)]
