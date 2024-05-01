import numpy as np
import gym 
import gym.spaces
from joblib import delayed

'''
Evaluate a agent in an environment, with single process.
Input:
    agent: the agent to be evaluated, must be one agent in a population.
    env_name: the name of the environment. 
        Using the name rather than the environment object convinces the parallelization.
    max_steps: maximum number of steps in an episode.

Return:
    total_reward: total reward in an episode.
'''
def eval_policy(agent, env_name, max_steps, redundant_dim, sigma):
    env = gym.make(env_name)
    total_reward = 0
    
    obs, info = env.reset()
    obs = np.append(obs, np.random.normal(0, sigma, redundant_dim))
    
    terminated = False 
    truncated = False
    
    # Action space is continuous
    if isinstance(env.action_space, gym.spaces.Box):
        for _ in range(max_steps):
            action = agent.predict(np.array(obs).reshape(1, -1), scale="tanh")
            # In the new version, the return has additional element "truncated". 
            new_obs, reward, terminated, truncated, info = env.step(action)
            new_obs = np.append(new_obs, np.random.normal(0, sigma, redundant_dim))
            total_reward = total_reward + reward
            obs = new_obs
            if terminated or truncated:
                break

    # Action space is discrete
    elif isinstance(env.action_space, gym.spaces.Discrete):
        for _ in range(max_steps):
            action = agent.predict(np.array(obs).reshape(1, -1), scale="softmax")
            new_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward = total_reward + reward
            obs = new_obs
            if terminated or truncated:
                break
    else:
        raise ValueError("Unknown action space")
    env.close()

    return total_reward


# for parallel
eval_policy_delayed = delayed(eval_policy)


# check the distribution of the observations. 
# see the observation array as samples from iid Gaussian, 
# report the deviation 
def check_the_deviation(agent, env_name, max_steps, redundant_dim, sigma):
    env = gym.make(env_name)
    total_reward = 0
    
    obs, info = env.reset()
    obs = np.append(obs, np.random.normal(np.mean(obs), np.std(obs), redundant_dim))

    obs_list = []
    
    terminated = False 
    truncated = False
    
    for _ in range(max_steps):
        action = agent.predict(np.array(obs).reshape(1, -1), scale="tanh")
        # In the new version, the return has additional element "truncated". 
        new_obs, reward, terminated, truncated, info = env.step(action)
        new_obs = np.append(new_obs, np.random.normal(np.mean(new_obs), np.std(new_obs), redundant_dim))
        total_reward = total_reward + reward
        obs = new_obs
        for element in obs:
            obs_list.append(element)
        if terminated or truncated:
            break

    env.close()
    obs_list = np.array(obs_list)
    return obs_list.mean(), obs_list.std()
