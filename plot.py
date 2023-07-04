import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import os 


'''
绘制 truncation rate 对HT的影响。
'''

env_list = ['AtlantisNoFrameskip-', 'QbertNoFrameskip-']

LOG_DIR = 'logs/'

def smooth(x, alpha=0.9):
    y = np.zeros(len(x))
    for i in range(len(x)):
        y[i] = alpha * y[i-1] + (1 - alpha) * x[i]
    return y

def extract_rewards_from_csv(exp_name, env_name):
    file_name_list = os.listdir(LOG_DIR)
    data = []
    for file_name in file_name_list:
        if exp_name in file_name and env_name in file_name:
            try: 
                tmp = pd.read_csv(LOG_DIR + file_name)['mean_reward']
            except:
                print(file_name)
            data.append(tmp)
    res = pd.DataFrame(data).mean(axis=0) 
    return res

def plot_subfigure_for_one_env(env_name, idx):
    plt.subplot(2, 1, idx)
    plt.title('HT with different truncation rates: ' + env_name)
    
    plt.plot(smooth(extract_rewards_from_csv('LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|', env_name)), label='HT=0.0', color='gray', alpha=0.3)
    plt.plot(smooth(extract_rewards_from_csv('LR1e-2|HT=0.5|Pop=128|STD=0.02|Linear|', env_name)), label='HT=0.5', color='blue', alpha=0.3)
    plt.plot(smooth(extract_rewards_from_csv('LR1e-2|HT=0.9|Pop=128|STD=0.02|Linear|', env_name)), label='HT=0.9', color='blue', alpha=0.5)

    plt.legend(loc='lower right', fontsize=6)
 
plt.figure(figsize=(5, 6), dpi=300)
plot_subfigure_for_one_env('AtlantisNoFrameskip', 1)
plot_subfigure_for_one_env('QbertNoFrameskip', 2)

plt.suptitle('Architecture: 884 + 442', fontsize=16)

plt.tight_layout()
plt.savefig(f"plots/Performance.png")
