class EligibilityTrace:

    def __init__(self, decay_rate: float, q_values):
        self.decay_rate = decay_rate
        self.q_values = q_values

    def get_value(self, states, action):
        state = self.q_values[states[0]][states[1]][states[2]][states[3]]
        if state in self.q_values:
            return self.q_values[state].get_value(action, 0)
        return 0

    def decay(self, states: list, action: int):
        state = self.q_values[states[0]][states[1]][states[2]][states[3]]
        does_have_state_action_val: bool = state in self.q_values and action in self.q_values[state]
        if does_have_state_action_val:
            if state in self.q_values:
                old_val = self.get_value(state, action)
                new_val = old_val * self.decay_rate
                if state not in self.q_values:
                    self.q_values[state] = {}
                else:
                    self.q_values[state][action] = new_val

    def increment(self, states: list, action: int, set_value: int = 1):
        self.q_values[states[0]][states[1]][states[2]][states[3]][action] += set_value

    def get(self, states: list, action: int):
        state = self.q_values[states[0]][states[1]][states[2]][states[3]]
        return self.q_values[state][action]
