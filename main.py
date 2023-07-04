import os
import gym
import pickle
import time 
import argparse 
import numpy as np
import pandas as pd 
from tqdm import tqdm
from joblib import Parallel
from model import Torch_CNN_MLP
from evaluation import eval_policy_delayed, eval_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="Qbert-v4")   
    parser.add_argument("--exp_name", type=str, default="test")  
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--population_size", type=int, default=128)                 
    parser.add_argument("--reward_process", type=str, default="standardize")        # "standardize" or "centered_rank"
    parser.add_argument("--n_jobs", type=int, default=32)                           # The number of parallel jobs
    parser.add_argument("--seed", type=int, default=42)                             # Set it for numpy only, not for gym

    parser.add_argument("--hidden_layers", type=int, nargs='+', default=256)        # The number features we created from the kernel
    parser.add_argument("--learning_rate", type=float, default=0.01)                # OpenAI's implementation uses 0.01, 0.003, 0.0001, 0.00003             
    parser.add_argument("--lr_decay", type=float, default=0.99)                     
    parser.add_argument("--noise_std", type=float, default=0.02)                    # 0.02 is used in OpenAI's implementation, but here we use 0.1
    parser.add_argument("--noise_decay", type=float, default=1.00)                  # The noise std will decay by this factor

    parser.add_argument("--redundant", type=float, default=0.1)                     # The percentage of redundant parameters
    parser.add_argument("--ht", action="store_true")                                # Whether to use the hard threshold

    parser.add_argument("--eval_freq", type=int, default=10)                        # Evaluate the policy every eval_freq epochs
    parser.add_argument("--n_trajs", type=int, default=10)                          # Each evaluation, we collect n_trajs trajectories
    parser.add_argument("--max_steps", type=int, default=30000)                     # Follow OpenAI's setting: https://github.com/michaelnny/deep_rl_zoo/blob/main/deep_rl_zoo/gym_env.py
                                                                                    # This only works for learning, not for evaluation.
    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--log_path", type=str, default="logs/")

    args = parser.parse_args()

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    # set the seed
    np.random.seed(args.seed)

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    env.close()

    if args.load_model != "":
        raise NotImplementedError
    else:
        policy = Torch_CNN_MLP(action_dim)
    
    to_csv_file = []    

    import NES 
    es_trainner = NES.NES_Trainer(agent=policy, 
        learning_rate = args.learning_rate,
        noise_std = args.noise_std,
        noise_decay = args.noise_decay,
        lr_decay = args.lr_decay,
        decay_step=50,
        norm_rewards=True,
        truncate=args.redundant if args.ht else 0.0,
    )
    total_steps = 0
    start_time = time.time()

    for i in tqdm(range(args.epochs)):
        # break if the total training time exceeds 6 hours
        if time.time() - start_time > 6 * 3600 or total_steps > 1e9:
            break

        es_agents = es_trainner.generate_population(args.population_size)
        eval_job = (eval_policy_delayed(agent, args.env_name, args.max_steps) for agent in es_agents)
        echoes = Parallel(n_jobs=args.n_jobs)(eval_job)
        echoes = np.array(echoes)
        traj_rewards, step_counts = echoes[:, 0], echoes[:, 1]
        total_steps += np.array(step_counts).sum()

        es_trainner.update_agent(traj_rewards, args.reward_process)

        
        if i % args.eval_freq == 0:
            best_agent = es_trainner.get_agent()

            # evaluate the best agent to collect args.n_trajs trajectories
            rewards = []
            for _ in range(args.n_trajs):
                rewards.append(eval_policy(best_agent, args.env_name, max_steps=108000)[0])
            rewards = np.array(rewards)
        
            print("----------------------------------------------------")
            print(f"Epoch {i} : mean reward {rewards.mean():.3f},  step {total_steps}")
            print(f"Reward: {rewards}")
            print("----------------------------------------------------")

            
            # TODO: add logging
            to_csv_file.append([int(i), round(rewards.mean(), 1), round(rewards.std(), 1), round(es_trainner.noise_std, 5)])
            if 'ALE/' in args.env_name:
                env_name = args.env_name.split('ALE/')[1]
            else:
                env_name = args.env_name
            pd.DataFrame(to_csv_file, columns=["epoch", "mean_reward", "std_reward", "noise_std"])\
                .to_csv(f"{args.log_path}/{args.exp_name}_{env_name}_seed{args.seed}.csv", index=False)
