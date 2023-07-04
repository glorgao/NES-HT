import numpy as np
import gym 
import gym.spaces
from joblib import delayed
import torch 
import cv2
from gym.spaces import Box


class ResizeAndGrayscaleFrame(gym.ObservationWrapper):
    """
    Resize frames to 84x84, and grascale image as done in the Nature paper.
    """

    def __init__(self, env, width=84, height=84, grayscale=True):
        super().__init__(env)

        assert self.observation_space.dtype == np.uint8 and len(self.observation_space.shape) == 3

        self.frame_width = width
        self.frame_height = height
        self.grayscale = grayscale
        num_channels = 1 if self.grayscale else 3

        self.observation_space = Box(
            low=0,
            high=255,
            shape=(self.frame_height, self.frame_width, num_channels),
            dtype=np.uint8,
        )

    def observation(self, observation):
        frame = observation

        # pylint: disable=no-member
        if self.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_AREA)
        # pylint: disable=no-member

        if self.grayscale:
            frame = np.expand_dims(frame, -1)

        obs = frame
        return obs


class NoopReset(gym.Wrapper):
    """Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    """

    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, terminated, truncated, _ = self.env.step(self.noop_action)
            if terminated or truncated:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, action):
        return self.env.step(action)


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
def eval_policy(agent, env_name, max_steps):
    env = gym.make(env_name)

    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=max_steps)
    env = ResizeAndGrayscaleFrame(env, width=84, height=84)
    env = gym.wrappers.FrameStack(env, 4)
    env = NoopReset(env, noop_max=30)

    total_reward = 0
    
    obs = env.reset()

    terminated = False
    truncated = False
    
    assert isinstance(env.action_space, gym.spaces.Discrete) 

    # Action space is discrete
    step_count = 0
    for _ in range(max_steps):
        obs = np.array(obs)
        obs = torch.tensor(obs, dtype=torch.float32)
        action = agent.forward(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        
        total_reward = total_reward + reward
    
        if terminated or truncated:
            break

    env.close()

    return total_reward, step_count


# for parallel
eval_policy_delayed = delayed(eval_policy)





