import gym
import numpy as np

env = gym.make("MountainCar-v0")
env.reset()

env_os_high = env.observation_space.high
env_os_low = env.observation_space.low
env_as = env.action_space.n

DISCRETE_OS_SIZE = [20] * len(env_os_high)
discrete_os_win_size = (env_os_high - env_os_low)/DISCRETE_OS_SIZE

q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env_as]))

done = False

while not done:
    action = 2 # right (left, nothing, right)
    new_state, reward, done, _ = env.step(action)
    env.render()

env.close()