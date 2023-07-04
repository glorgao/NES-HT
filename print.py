import numpy as np 
import pandas as pd 

LOG_DIR = 'logs/'
env_list = ['AtlantisNoFrameskip', 'BattleZoneNoFrameskip', 'BreakoutNoFrameskip', 'PongNoFrameskip', 'QbertNoFrameskip', 'SkiingNoFrameskip', 
            'VentureNoFrameskip', 'ZaxxonNoFrameskip']
setting = 'LR1e-2|HT=0.0|Pop=128|STD=0.02|Linear|LR=0.01|'
setting = 'LR1e-2|HT=0.0|Pop=128|STD=0.02|MLP|'

for env in env_list:
    file_name = LOG_DIR + setting + '_' + env + '-v4_seed0.csv'
    data = pd.read_csv(file_name)[-10:]
    print(env, data['mean_reward'].mean())