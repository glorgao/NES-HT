import torch 
import numpy as np 
import torch.nn as nn
# This is very important for speed up the training process,
# Using the default setting will slow down the training process by about 2 times.
torch.set_num_threads(1)
    
class Torch_CNN_MLP(nn.Module):
    def __init__(self, output_dim):
        super(Torch_CNN_MLP, self).__init__()

        # CNN layers for feature extraction
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten()
        )

        # MLP layers for mapping features to action
        self.mlp_layers = nn.Sequential(
            nn.Linear(2592, output_dim),
            # nn.ReLU(),
            # nn.Linear(256, 256),
            # nn.ReLU(),
            # nn.Linear(256, output_dim)
        )

    def forward(self, x):
        with torch.no_grad():
            # the input x is (4, 84, 84, 1), the last 1 means the number of envs(for parallelization)
            # while the input of nn.Conv2d should be (N, C, H, W)
            # so we transpose the input x from (4, 84, 84, 1) to (1, 4, 84, 84)
            x = x.transpose(0, 3).transpose(3, 1)
            x = self.cnn_layers(x)      
            x = self.mlp_layers(x)

            return torch.max(x, 1).indices.item()
        
if __name__ == "__main__":
    import gym 
    from evaluation import ResizeAndGrayscaleFrame, NoopReset
   
    env = gym.make("PongNoFrameskip-v4") 
    max_steps = 100000
    env = gym.wrappers.TimeLimit(env.env, max_episode_steps=None if max_steps <= 0 else max_steps)
    env = ResizeAndGrayscaleFrame(env, width=84, height=84)
    env = gym.wrappers.FrameStack(env, 4)
    env = NoopReset(env, noop_max=30)

    obs_shape = env.observation_space.shape
    action_dim = env.action_space.n

    agent = Torch_CNN_MLP(action_dim)
    params = torch.cat([param.view(-1) for param in agent.parameters()])
    print('params.shape', params.shape)
    obs = env.reset()
    
    import time 
    start_time = time.time()
    for _ in range(10000):
        obs = np.array(obs)
        obs = torch.from_numpy(obs).to(dtype=torch.float32)
        action = agent.forward(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
    print(time.time() - start_time)
    env.close()
