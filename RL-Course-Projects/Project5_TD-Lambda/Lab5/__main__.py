# __main__.py
# Kordel France
########################################################################################################################
# This file establishes the driver for a reinforcement learning model.
########################################################################################################################


import sys

from Lab5 import data_processing
from Lab5 import config
from Lab5 import model_config
from Lab5.Model import Model
from Lab5.RL.q_learning_algorithm import train_q_learning_algorithm
# from Lab5.RL.td_lambda_algorithm import train_td_lamda_algorithm
# from Lab5.RL.value_iteration_algorithm import train_value_iteration_algorithm
# from Lab5.RL.sarsa_algorithm import train_sarsa_algorithm
from Lab5.RL import demo_config_q_learning
# from Lab5.RL import demo_config_td_lambda
from Lab5.RL import demo_config_value_iteration
from Lab5.RL import demo_config_sarsa

from Lab5 import monte_carlo
monte_carlo.run_monte_carlo_situation()
sys.exit()
asa


def initialize_rl_agent(algorithm) -> Model:
    if algorithm == 'q-learning':
        function = train_q_learning_algorithm
    elif algorithm == 'td-lambda':
        function = train_td_lamda_algorithm
    elif algorithm == 'value iteration':
        function = train_value_iteration_algorithm
    else:
        function = train_sarsa_algorithm

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


if __name__ == '__main__':

    # check path i/o directory and move it to the correct place if necessary
    data_processing.rename_data_files(config.IO_DIRECTORY)

    """
    This starts a Monte-Carlo simulation of different hyperparameters for all 3 algorithms over all 3 tracks.
    It was used to construct data for the paper, but not used for demonstration.
    """
    # ###############################################################################################
    # monte_carlo.run_monte_carlo_situation()
    # ###############################################################################################

    # # ALGORITHM - TD-LAMBDA
    # ##############################################################################################
    # ##############################################################################################
    # model_config.TRACK_NAME = 'L-track.txt'
    # model_config.LAP_COUNT = demo_config_td_lambda.LAP_COUNT
    # model_config.GAMMA = demo_config_td_lambda.GAMMA
    # model_config.NU = demo_config_td_lambda.NU
    # model_config.EPOCHS = demo_config_td_lambda.EPOCHS
    # model_config.EPISODES = demo_config_td_lambda.EPISODES
    # model_config.Q_STABILITY_ERROR = demo_config_td_lambda.Q_STABILITY_ERROR
    # model_config.EPOCH_THRESHOLD = demo_config_td_lambda.EPOCH_THRESHOLD
    # model_config.UPDATE_STEPS = demo_config_td_lambda.UPDATE_STEPS
    #
    # agent = initialize_rl_agent('td-lambda')
    # agent.train_agent()
    # agent.print_hyperparameters()

    # ALGORITHM - Q-LEARNING
    ##############################################################################################
    ##############################################################################################
    model_config.TRACK_NAME = 'L-track.txt'
    model_config.LAP_COUNT = demo_config_q_learning.LAP_COUNT
    model_config.GAMMA = demo_config_q_learning.GAMMA
    model_config.NU = demo_config_q_learning.NU
    model_config.EPOCHS = demo_config_q_learning.EPOCHS
    model_config.EPISODES = demo_config_q_learning.EPISODES
    model_config.Q_STABILITY_ERROR = demo_config_q_learning.Q_STABILITY_ERROR
    model_config.EPOCH_THRESHOLD = demo_config_q_learning.EPOCH_THRESHOLD
    model_config.UPDATE_STEPS = demo_config_q_learning.UPDATE_STEPS

    agent = initialize_rl_agent('q-learning')
    agent.train_agent()
    agent.print_hyperparameters()


    # ALGORITHM - VALUE ITERATION
    ###############################################################################################
    ###############################################################################################
    # model_config.TRACK_NAME = 'L-track'
    # model_config.LAP_COUNT = demo_config_value_iteration.LAP_COUNT
    # model_config.GAMMA = demo_config_value_iteration.GAMMA
    # model_config.NU = demo_config_value_iteration.NU
    # model_config.EPOCHS = demo_config_value_iteration.EPOCHS
    # model_config.EPISODES = demo_config_value_iteration.EPISODES
    # model_config.Q_STABILITY_ERROR = demo_config_value_iteration.Q_STABILITY_ERROR
    # model_config.EPOCH_THRESHOLD = demo_config_value_iteration.EPOCH_THRESHOLD
    # model_config.UPDATE_STEPS = demo_config_value_iteration.UPDATE_STEPS
    #
    # agent = initialize_rl_agent('value iteration')
    # agent.train_agent()
    # agent.print_hyperparameters()


    # ALGORITHM - SARSA
    ###############################################################################################
    ###############################################################################################
    # model_config.TRACK_NAME = 'O-track'
    # model_config.LAP_COUNT = demo_config_sarsa.LAP_COUNT
    # model_config.GAMMA = demo_config_sarsa.GAMMA
    # model_config.NU = demo_config_sarsa.NU
    # model_config.EPOCHS = demo_config_sarsa.EPOCHS
    # model_config.EPISODES = demo_config_sarsa.EPISODES
    # model_config.Q_STABILITY_ERROR = demo_config_sarsa.Q_STABILITY_ERROR
    # model_config.EPOCH_THRESHOLD = demo_config_sarsa.EPOCH_THRESHOLD
    # model_config.UPDATE_STEPS = demo_config_sarsa.UPDATE_STEPS
    #
    # agent = initialize_rl_agent('sarsa')
    # agent.train_agent()
    # agent.print_hyperparameters()


