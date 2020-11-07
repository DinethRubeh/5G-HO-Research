import gym
import numpy as np

env = gym.make("MountainCar-v0") ## states -> position, velocity  ## actions -> left, nothing, right
env.reset()

LEARNING_RATE = 0.1
DISCOUNT = 0.95
EPISODES = 25000

env_os_high = env.observation_space.high
env_os_low = env.observation_space.low
env_as = env.action_space.n

DISCRETE_OS_SIZE = [20] * len(env_os_high)
discrete_os_win_size = (env_os_high - env_os_low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env_as]))

# env.step return state is continuous 
def get_discrete_state(state):
    discrete_state = (state - env_os_low) / discrete_os_win_size
    return tuple(discrete_state.astype(np.int)) # using tuple format, can search in the q table

discrete_state = get_discrete_state(env.reset()) # env.reset gives the initial state
print(discrete_state)
print(np.argmax(q_table[discrete_state]))

done = False
while not done:
    action = np.argmax(q_table[discrete_state])
    new_state, reward, done, _ = env.step(action)
    new_discrete_state = get_discrete_state(new_state)
    env.render()
    if not done:
        max_future_q = np.max(q_table[new_discrete_state])
        current_q = q_table[discrete_state + (action, )]
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)
        q_table[discrete_state+(action, )] = new_q # update q_table

    elif new_state[0] >= env.goal_position:
        q_table[discrete_state + (action, )] = 0

    discrete_state = new_discrete_state

env.close()