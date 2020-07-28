import numpy as np

from maze import MazeEnv
from value_function_plot import value_function_plot

discount = 0.9
iters = 1000

env = MazeEnv()
state_dim = env.distinct_states
state_values = np.zeros(state_dim)
Qval = np.zeros((state_dim, env.action_space.n))  # Action values
policy = np.zeros(state_dim)


def calcQ(state, action):
    """Calculate Value function for given state and action

    Args:
        state (int): Valid (discrete) state in discrete `env.observation_space`
        action (int): Valid (discrete) action in `env.action_space`

    Returns:
        v_sum: value for given state, action
    """
    Vsum = 0
    transitions = []
    slip_action = env.slip_action_map[action]
    env.set_state(state)
    slip_next_state, slip_reward, _ = env.step(slip_action, slip=False)
    transitions.append((slip_reward, slip_next_state, env.slip))
    env.set_state(state)
    next_state, reward, _ = env.step(action, slip=False)
    transitions.append((reward, next_state, 1 - env.slip))
    for reward, next_state, pi in transitions:
        Vsum += pi * (reward + discount * state_values[next_state])
    return Vsum


# Value Iteration
for i in range(iters):
    tmpV = np.zeros(state_dim)
    for state in range(state_dim):
        if env.idx2cell[int(state / 8)] == env.goal_pos:
            continue
        Vmax = float("-inf")
        for action in range(env.action_space.n):
            Vsum = calcQ(state, action)
            Vmax = max(Vmax, Vsum)
        tmpV[state] = Vmax
    state_values = np.copy(tmpV)

for state in range(state_dim):
    for action in range(env.action_space.n):
        Qval[state, action] = calcQ(state, action)

for state in range(state_dim):
    policy[state] = np.argmax(Qval[state, :])

np.save("Results/Q_Values", Qval)
print(Qval)
# np.save('Optimal_policies',optpolicies)
print("Action mapping:[0 - UP; 1 - DOWN; 2 - LEFT; 3 - RIGHT")
print("Optimal actions:")
print(policy)
value_function_plot(Qval, env)
