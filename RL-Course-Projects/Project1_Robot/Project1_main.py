
import CoFEA.environments.Project1_agent as ag1
import CoFEA.environments.Project1_env as env1


def main():
    environment = env1.RobotGame()
    agent = ag1.RlAgent()
    # Check that the environment parameters match
    if (environment.get_number_of_states() == agent.get_number_of_states()) and \
            (environment.get_number_of_actions() == agent.get_number_of_actions()):
        # Play 100 games
        for i in range(100):
            # reset the game and observe the current state
            current_state = environment.reset()
            game_end = False
            # Do until the game ends:
            while not game_end:
                action = agent.select_action(current_state)
                new_state, reward, game_end = environment.execute_action(action)
                agent.update_q(new_state, reward)
                current_state = new_state
        with open('Project1.txt', 'wt') as f:
            print(agent.q, file=f)
        print("\nProgram completed successfully.")
    else:
        print("Environment and Agent parameters do not match. Terminating program.")


if __name__ == "__main__":
    main()
