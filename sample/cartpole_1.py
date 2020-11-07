import gym
import numpy as np

env = gym.make('CartPole-v0')

done = False
count = 0

observation = env.reset() # starting observation

while not done:
    env.render()
    count += 1

    action = env.action_space.sample()

    observation, reward, done, _ = env.step(action)

    if done:
        break

print("game lasted: ", count, "moves")