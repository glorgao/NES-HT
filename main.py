import os
import gym
import pickle
import argparse 
import numpy as np
import pandas as pd 
from tqdm import tqdm
from joblib import Parallel
from linear import ThreeLayerNetwork, OneLayerNetwork
from evaluation import eval_policy_delayed, eval_policy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="BipedalWalker-v3")   
    parser.add_argument("--exp_name", type=str, default="test")  
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--population_size", type=int, default=256)                 
    parser.add_argument("--reward_process", type=str, default="standardize")        # "standardize" or "centered_rank"
    parser.add_argument("--n_jobs", type=int, default=2)                            # The number of parallel jobs
    parser.add_argument("--seed", type=int, default=42)                             # Set it for numpy only, not for gym

    parser.add_argument("--hidden_layers", type=int, nargs='+', default=256)        # The number features we created from the kernel
    parser.add_argument("--learning_rate", type=float, default=0.1)                 
    parser.add_argument("--lr_decay", type=float, default=0.99)                    
    parser.add_argument("--noise_std", type=float, default=0.1)                     # 0.02 is used in OpenAI's implementation, but here we use 0.1
    parser.add_argument("--noise_decay", type=float, default=0.99)                  # The noise std will decay by this factor

    parser.add_argument("--redundant", type=float, default=0.0)                     # The percentage of redundant observations
    parser.add_argument("--ht", type=float, default=0.0)                            # The percentage of parameters to be truncated, 0.0-> regular NES.
    parser.add_argument("--sigma", type=float, default=1.0)                         # The standard deviation of the noise

    parser.add_argument("--eval_freq", type=int, default=10)                        # Evaluate the policy every eval_freq epochs
    parser.add_argument("--n_trajs", type=int, default=10)                          # Each evaluation, we collect n_trajs trajectories
    parser.add_argument("--max_steps", type=int, default=1500)                      # To be removed since often we use the env's max_steps


    parser.add_argument("--load_model", type=str, default="")
    parser.add_argument("--log_path", type=str, default="logs/")
    parser.add_argument("--save_model", action="store_true")

    args = parser.parse_args()

    if not os.path.exists("./logs"):
        os.makedirs("./logs")

    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    # set the seed
    np.random.seed(args.seed)

    env = gym.make(args.env_name)
    state_dim = env.observation_space.shape[0]
    redundant_dim = int(state_dim * args.redundant)
    state_dim += redundant_dim
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    env.close()

    if args.load_model != "":
        policy = ThreeLayerNetwork.from_model(args.load_model)
        assert policy.in_features == state_dim, "incorrect policy input dims"
        assert policy.out_features == action_dim, "incorrect policy output dims"
    else:
        policy = OneLayerNetwork(state_dim, action_dim)
    
    to_csv_file = []

    import NES 
    es_trainner = NES.NES_Trainer(agent=policy, 
        learning_rate = args.learning_rate,
        noise_std = args.noise_std,
        noise_decay = args.noise_decay,
        lr_decay = args.lr_decay,
        decay_step=50,
        norm_rewards=True,
        truncate=args.ht
    )

    for i in tqdm(range(args.epochs)):
        es_agents = es_trainner.generate_population(args.population_size)
        eval_job = (eval_policy_delayed(agent, args.env_name, args.max_steps, redundant_dim=redundant_dim, sigma=args.sigma) for agent in es_agents)
        traj_rewards = Parallel(n_jobs=args.n_jobs)(eval_job)
        traj_rewards = np.array(traj_rewards)

        es_trainner.update_agent(traj_rewards, args.reward_process)

        if i % args.eval_freq == 0:
            best_agent = es_trainner.get_agent()

            # evaluate the best agent to collect args.n_trajs trajectories
            rewards = []
            for _ in range(args.n_trajs):
                rewards.append(eval_policy(best_agent, args.env_name, args.max_steps, redundant_dim=redundant_dim, sigma=args.sigma))
            rewards = np.array(rewards)
        
            print("----------------------------------------------------")
            print(f"Epoch {i} : mean reward {rewards.mean():.3f}")
            print("----------------------------------------------------")

            # save the current policy 
            if args.save_model:
                with open(f".models/{args.exp_name}_{args.env_name}_seed{args.seed}_epoch{i}.pkl" , "wb") as f:
                    pickle.dump(policy, f)
            
            # TODO: add logging
            to_csv_file.append([int(i), round(rewards.mean(), 1), round(rewards.std(), 1), round(es_trainner.noise_std, 5)])
            pd.DataFrame(to_csv_file, columns=["epoch", "mean_reward", "std_reward", "noise_std"])\
                .to_csv(f"{args.log_path}/{args.exp_name}_{args.sigma}_{args.env_name}_seed{args.seed}.csv", index=False)
