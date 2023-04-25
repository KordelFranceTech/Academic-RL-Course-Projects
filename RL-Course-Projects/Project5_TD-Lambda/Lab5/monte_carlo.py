# monte_carlo.py
# Kordel France
########################################################################################################################
# This file establishes the driver for a monte-carlo simulation of a reinforcement learning model.
########################################################################################################################



from Lab5 import data_processing
from Lab5 import encoding
from Lab5 import config
from Lab5 import model_config
from Lab5.cross_validate import construct_data_folds, get_datasets_for_fold
from Lab5 import standardize
from Lab5.Model import Model
from Lab5.RL import q_learning_algorithm
from Lab5.RL import td_lambda_algorithm
from Lab5.RL import value_iteration_algorithm
from Lab5.RL import sarsa_algorithm
from Lab5.RL.q_learning_algorithm import train_q_learning_algorithm
from Lab5.RL.td_lambda_algorithm import train_td_lamda_algorithm
from Lab5.RL.value_iteration_algorithm import train_value_iteration_algorithm
from Lab5.RL.sarsa_algorithm import train_sarsa_algorithm
from Lab5.RL import demo_config_q_learning
from Lab5.RL import demo_config_td_lambda
from Lab5.RL import demo_config_value_iteration
from Lab5.RL import demo_config_sarsa


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random


def initialize_rl_agent(algorithm) -> Model:
    if algorithm == 'q-learning':
        function = train_q_learning_algorithm
    elif algorithm == 'td-lambda':
        function = train_td_lamda_algorithm
    elif algorithm == 'value iteration':
        function = train_value_iteration_algorithm
    else:
        function = train_sarsa_algorithm

    # agent: Model = Model(function=function,
    #                      algorithm=algorithm,
    #                      track_file=model_config.TRACK_NAME,
    #                      lap_count=model_config.LAP_COUNT,
    #                      velocity_min=model_config.VELOCITY_MIN,
    #                      velocity_max=model_config.VELOCITY_MAX,
    #                      gamma=model_config.GAMMA,
    #                      nu=model_config.NU,
    #                      reset_after_crash=model_config.RESET_AFTER_CRASH,
    #                      acceleration_rate=model_config.ACCELERATION_RATE,
    #                      acceleration_misfire_rate=model_config.ACCELERATION_MISFIRE_RATE,
    #                      epochs=model_config.EPOCHS,
    #                      episodes=model_config.EPISODES,
    #                      q_stability_error=model_config.Q_STABILITY_ERROR,
    #                      epoch_threshold=model_config.EPOCH_THRESHOLD,
    #                      update_steps=model_config.UPDATE_STEPS)

    agent: Model = Model(function=function,
                         algorithm=algorithm,
                         track_file=model_config.TRACK_NAME,
                         lap_count=model_config.LAP_COUNT,
                         velocity_min=model_config.VELOCITY_MIN,
                         velocity_max=model_config.VELOCITY_MAX,
                         gamma=model_config.GAMMA,
                         nu=model_config.NU,
                         reset_after_crash=model_config.RESET_AFTER_CRASH,
                         acceleration_rate=model_config.ACCELERATION_RATE,
                         acceleration_misfire_rate=model_config.ACCELERATION_MISFIRE_RATE,
                         epochs=model_config.EPOCHS,
                         episodes=model_config.EPISODES,
                         q_stability_error=model_config.Q_STABILITY_ERROR,
                         epoch_threshold=model_config.EPOCH_THRESHOLD,
                         update_steps=model_config.UPDATE_STEPS)
    return agent


def run_monte_carlo_situation():
    print('\n\n\nrunning monte carlo simulation')

    # check path i/o directory and move it to the correct place if necessary
    data_processing.rename_data_files(config.IO_DIRECTORY)

    # ALGORITHM - Q-LEARNING
    ##############################################################################################
    ##############################################################################################
    # NU_VALUES: list = [0.001, 0.1, 0.2, 0.3, 0.4]
    # GAMMA_VALUES: list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    NU_VALUES: list = [0.1]
    GAMMA_VALUES: list = [0.8]
    CRASH_TYPES: list = [0, 1]
    EPOCHS: list = []
    COSTS: list = []
    HYPERS: list = []
    ALGORITHM: str = 'q-learning'

    for CRASH_TYPE in CRASH_TYPES:
        # for NU in NU_VALUES:
        NU = 0.2
        for GAMMA in GAMMA_VALUES:
            print(f'\n\n_______________________________________________________________')
            print(f'evaluating model for crash type: {CRASH_TYPE}, gamma = {GAMMA}, nu = {NU}')
            model_config.TRACK_NAME = 'L-track.txt'
            model_config.LAP_COUNT = demo_config_q_learning.LAP_COUNT
            model_config.GAMMA = GAMMA
            model_config.NU = NU
            model_config.EPOCHS = demo_config_q_learning.EPOCHS
            model_config.EPISODES = demo_config_q_learning.EPISODES
            model_config.RESET_AFTER_CRASH = CRASH_TYPE
            model_config.Q_STABILITY_ERROR = demo_config_q_learning.Q_STABILITY_ERROR
            model_config.EPOCH_THRESHOLD = demo_config_q_learning.EPOCH_THRESHOLD
            model_config.UPDATE_STEPS = demo_config_q_learning.UPDATE_STEPS

            agent = initialize_rl_agent('q-learning')
            q_learning_algorithm.model_config = model_config
            epochs, costs = agent.train_agent()
            hypers = agent.print_hyperparameters()
            EPOCHS.append(epochs)
            COSTS.append(costs)
            HYPERS.append(hypers)

    output_file = open(f'{ALGORITHM}_{model_config.TRACK_NAME}_monte-carlo.txt', 'w')
    output_file.write(f'{ALGORITHM} monte carlo summary stats over track {model_config.TRACK_NAME}:\n')
    output_file.write(f'_______________________________________________________________\n')
    for index_i in range(0, len(EPOCHS)):
        output_file.write(f'e{index_i} = {EPOCHS[index_i]}\ns{index_i} = {COSTS[index_i]}\n')
    output_file.write(f'_______________________________________________________________\n')
    for index_j in range(0, len(HYPERS)):
        output_file.write(f'_______________________________________________________________\n')
        output_file.write(f'{HYPERS[index_j]}')
    output_file.close()

    # # ALGORITHM - VALUE ITERATION
    # ###############################################################################################
    # ###############################################################################################
    # # NU_VALUES: list = [0.001, 0.1, 0.2, 0.3, 0.4]
    # # GAMMA_VALUES: list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    # # NU_VALUES: list = [0.2]
    # GAMMA_VALUES: list = [0.8]
    # CRASH_TYPES: list = [0, 1]
    # EPOCHS: list = []
    # COSTS: list = []
    # HYPERS: list = []
    # ALGORITHM: str = 'value iteration'
    #
    # for CRASH_TYPE in CRASH_TYPES:
    #     # for NU in NU_VALUES:
    #     NU = 0.2
    #     for GAMMA in GAMMA_VALUES:
    #         print(f'\n\n_______________________________________________________________')
    #         print(f'evaluating model for crash type: {CRASH_TYPE}, gamma = {GAMMA}, nu = {NU}')
    #         model_config.TRACK_NAME = 'R-track.txt'
    #         model_config.LAP_COUNT = demo_config_value_iteration.LAP_COUNT
    #         model_config.GAMMA = GAMMA
    #         model_config.NU = NU
    #         model_config.EPOCHS = demo_config_value_iteration.EPOCHS
    #         model_config.EPISODES = demo_config_value_iteration.EPISODES
    #         model_config.RESET_AFTER_CRASH = CRASH_TYPE
    #         model_config.Q_STABILITY_ERROR = demo_config_value_iteration.Q_STABILITY_ERROR
    #         model_config.EPOCH_THRESHOLD = demo_config_value_iteration.EPOCH_THRESHOLD
    #         model_config.UPDATE_STEPS = demo_config_value_iteration.UPDATE_STEPS
    #
    #         agent = initialize_rl_agent('value iteration')
    #         value_iteration_algorithm.model_config = model_config
    #         epochs, costs = agent.train_agent()
    #         hypers = agent.print_hyperparameters()
    #         EPOCHS.append(epochs)
    #         COSTS.append(costs)
    #         HYPERS.append(hypers)
    #
    # output_file = open(f'{ALGORITHM}_{model_config.TRACK_NAME}_monte-carlo.txt', 'w')
    # output_file.write(f'{ALGORITHM} monte carlo summary stats over track {model_config.TRACK_NAME}:\n')
    # output_file.write(f'_______________________________________________________________\n')
    # for index_i in range(0, len(EPOCHS)):
    #     output_file.write(f'e{index_i} = {EPOCHS[index_i]}\ns{index_i} = {COSTS[index_i]}\n')
    # output_file.write(f'_______________________________________________________________\n')
    # for index_j in range(0, len(HYPERS)):
    #     output_file.write(f'_______________________________________________________________\n')
    #     output_file.write(f'{HYPERS[index_j]}')
    # output_file.close()

    # # ALGORITHM - SARSA
    # ##############################################################################################
    # ##############################################################################################
    # # NU_VALUES: list = [0.001, 0.1, 0.2, 0.3, 0.4]
    # # GAMMA_VALUES: list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    # # NU_VALUES: list = [0.15]
    # GAMMA_VALUES: list = [0.8]
    # CRASH_TYPES: list = [0, 1]
    # EPOCHS: list = []
    # COSTS: list = []
    # HYPERS: list = []
    # ALGORITHM: str = 'sarsa'
    #
    # for CRASH_TYPE in CRASH_TYPES:
    #     # for NU in NU_VALUES:
    #     NU = 0.2
    #     for GAMMA in GAMMA_VALUES:
    #         print(f'\n\n_______________________________________________________________')
    #         print(f'evaluating model for crash type: {CRASH_TYPE}, gamma = {GAMMA}, nu = {NU}')
    #         model_config.TRACK_NAME = 'L-track.txt'
    #         model_config.LAP_COUNT = demo_config_sarsa.LAP_COUNT
    #         model_config.GAMMA = GAMMA
    #         model_config.NU = NU
    #         model_config.EPOCHS = demo_config_sarsa.EPOCHS
    #         model_config.EPISODES = demo_config_sarsa.EPISODES
    #         model_config.RESET_AFTER_CRASH = CRASH_TYPE
    #         model_config.Q_STABILITY_ERROR = demo_config_sarsa.Q_STABILITY_ERROR
    #         model_config.EPOCH_THRESHOLD = demo_config_sarsa.EPOCH_THRESHOLD
    #         model_config.UPDATE_STEPS = demo_config_sarsa.UPDATE_STEPS
    #
    #         agent = initialize_rl_agent('sarsa')
    #         sarsa_algorithm.model_config = model_config
    #         epochs, costs = agent.train_agent()
    #         hypers = agent.print_hyperparameters()
    #         EPOCHS.append(epochs)
    #         COSTS.append(costs)
    #         HYPERS.append(hypers)
    #
    # output_file = open(f'{ALGORITHM}_{model_config.TRACK_NAME}_monte-carlo.txt', 'w')
    # output_file.write(f'{ALGORITHM} monte carlo summary stats over track {model_config.TRACK_NAME}:\n')
    # output_file.write(f'_______________________________________________________________\n')
    # for index_i in range(0, len(EPOCHS)):
    #     output_file.write(f'e{index_i} = {EPOCHS[index_i]}\ns{index_i} = {COSTS[index_i]}\n')
    # output_file.write(f'_______________________________________________________________\n')
    # for index_j in range(0, len(HYPERS)):
    #     output_file.write(f'_______________________________________________________________\n')
    #     output_file.write(f'{HYPERS[index_j]}')
    # output_file.close()

    # ALGORITHM - TD-Learning
    ##############################################################################################
    ##############################################################################################
    # NU_VALUES: list = [0.001, 0.1, 0.2, 0.3, 0.4]
    # GAMMA_VALUES: list = [0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    LAMBDA_VALUES: list = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    GAMMA_VALUES: list = [0.8]
    CRASH_TYPES: list = [0, 1]
    EPOCHS: list = []
    COSTS: list = []
    HYPERS: list = []
    ALGORITHM: str = 'td-lambda'

    for CRASH_TYPE in CRASH_TYPES:
        NU = 0.2
        # for LAMBDA in LAMBDA_VALUES:
        LAMBDA = 0.0
        for GAMMA in GAMMA_VALUES:
            print(f'\n\n_______________________________________________________________')
            print(f'evaluating model for crash type: {CRASH_TYPE}, gamma = {GAMMA}, nu = {NU}')
            model_config.TRACK_NAME = 'L-track.txt'
            model_config.LAP_COUNT = demo_config_td_lambda.LAP_COUNT
            model_config.GAMMA = GAMMA
            model_config.NU = NU
            model_config.LAMBDA = LAMBDA
            model_config.EPOCHS = demo_config_td_lambda.EPOCHS
            model_config.EPISODES = demo_config_td_lambda.EPISODES
            model_config.RESET_AFTER_CRASH = CRASH_TYPE
            model_config.Q_STABILITY_ERROR = demo_config_td_lambda.Q_STABILITY_ERROR
            model_config.EPOCH_THRESHOLD = demo_config_td_lambda.EPOCH_THRESHOLD
            model_config.UPDATE_STEPS = demo_config_td_lambda.UPDATE_STEPS

            agent = initialize_rl_agent('td-lambda')
            td_lambda_algorithm.model_config = model_config
            epochs, costs = agent.train_agent()
            hypers = agent.print_hyperparameters()
            EPOCHS.append(epochs)
            COSTS.append(costs)
            HYPERS.append(hypers)

    output_file = open(f'{ALGORITHM}_{model_config.TRACK_NAME}_monte-carlo.txt', 'w')
    output_file.write(f'{ALGORITHM} monte carlo summary stats over track {model_config.TRACK_NAME}:\n')
    output_file.write(f'_______________________________________________________________\n')
    for index_i in range(0, len(EPOCHS)):
        output_file.write(f'e{index_i} = {EPOCHS[index_i]}\ns{index_i} = {COSTS[index_i]}\n')
    output_file.write(f'_______________________________________________________________\n')
    for index_j in range(0, len(HYPERS)):
        output_file.write(f'_______________________________________________________________\n')
        output_file.write(f'{HYPERS[index_j]}')
    output_file.close()

