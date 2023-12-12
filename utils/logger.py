import os
import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
from datetime import datetime

class Logger:
    def __init__(self, args, log_dir="logs"):
        self.log_dir = os.path.join(log_dir, args.env_name)
        self.log_file = os.path.join(self.log_dir, "log.txt")
        self.figure_dir = os.path.join(self.log_dir, "figures")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(self.figure_dir):
            os.makedirs(self.figure_dir)

    def store_array_list(self, array_list, filename):
        file_n = os.path.join(self.log_dir, filename)
        print(f'Saving arrays to {file_n}')
        with open(file_n, 'wb') as fp:
            pickle.dump(array_list, fp)
        
    def log_text(self, text):
        with open(self.log_file, "a") as f:
            f.write(f"{text}\n")

    def save_frames_as_gif(self, frames, filename, fps=30):
        file_n = os.path.join(self.figure_dir, filename)
        print(f'Saving gif to {file_n}')
        frames[0].save(file_n, save_all=True, append_images=frames[1:], optimize=False, duration=1000/fps, loop=0)

    def plot_rewards(self, step_list, reward_list):
        print(' ------------- plot -------------')
        plt.plot(step_list, reward_list)

        plt.figure(figsize=(10, 6))
        plt.plot(step_list, reward_list, label='Reward')
        plt.fill_between(step_list, reward_list, alpha=0.3)
        plt.xlabel('TotalEnvInteracts')
        #plt.ylabel('Reward')
        plt.xticks(np.linspace(0, 250000, 6), ['0', '50K', '100K', '150K', '200K', '250K'])
        plt.yticks(np.linspace(0, 20000, 5), ['0', '5K', '10K', '15K', '20K'])
        plt.ylim(0, 20000)
        plt.xlim(0, 250000)
        # Adding the title
        plt.title('Hopper-v4')
        plt.grid(True)
        plt.savefig('logs/figures/train.png')
        plt.close()

'''
filename = 'logs/train_reward.pkl'
with open(filename, 'rb') as file:
    reward_list = pickle.load(file)

filename = 'logs/steps.pkl'
with open(filename, 'rb') as file:
    step_list = pickle.load(file)

logger = Logger()
logger.plot_rewards(step_list, reward_list)
'''