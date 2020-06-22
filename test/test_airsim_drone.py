import gym
import sys
import os

path = os.getcwd().replace('test', '', 1) #this gives us a path from which we can import air_gym
sys.path.insert(1, path)
import air_gym

env = gym.make('air_gym:airsim-drone-v0',ip_address='130.108.129.49', control_type='continuous',step_length=1, image_shape=(192,192,3), goal=[20,20,-20])

# env.render()
episodes = 10
steps = 200
episode_rewards = []
for ep in range(episodes):
    ep_reward = 0
    print("Episode {}".format(ep))
    for s in range(steps):
        act = env.action_space.sample()
        obs, reward, done, state = env.step(act)
        if done:
            break
        ep_reward += reward
        # env.render()
    episode_rewards.append(ep_reward)
    print(episode_rewards)
    env.reset()

