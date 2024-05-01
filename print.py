import os 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt


log_dir = './logs/'

exp_name_list = ['red5.0|ht0.9|sigma_1.0_Walker2d', 'red10.0|ht0.9|sigma_1.0_Walker2d', 'red20.0|ht0.9|sigma_1.0_Walker2d',
            'red5.0|ht0.9|sigma_1.0_Hopper', 'red10.0|ht0.9|sigma_1.0_Hopper', 'red20.0|ht0.9|sigma_1.0_Hopper',
            'red5.0|ht0.9|sigma_1.0_HalfCheetah', 'red10.0|ht0.9|sigma_1.0_HalfCheetah', 'red20.0|ht0.9|sigma_1.0_HalfCheetah']


def extract_results_for(exp_name, file_name_list):
    exp_result = []
    for file_name in file_name_list:
        if exp_name in file_name:
            data = pd.read_csv(log_dir + file_name)
            exp_result.append(data['mean_reward'].values)
    return np.array(exp_result).mean(axis=0)[-10:].mean()

file_name_list = os.listdir(log_dir)
# filter out the files that are not related to the experiment

for exp_name in exp_name_list:
    print(exp_name, np.mean(extract_results_for(exp_name, file_name_list)))
