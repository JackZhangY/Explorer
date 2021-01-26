import os
import glob
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.helper import make_dir

# plt.style.use('seaborn-darkgrid')
plt.style.use('seaborn-paper')
import seaborn as sns
import collections

import warnings

warnings.filterwarnings("ignore")

from tensorboard.backend.event_processing import event_accumulator

radius = 20
num_sample = 308


def collect_tf_file(env_dir, alg_name):
    file_list = []
    for folder in os.listdir(env_dir):
        if folder.startswith(alg_name):
            file_list.append(env_dir + '/' + folder)
    return file_list

def plot_sns_img(file_list, items,  color, condition, length, million):
    smooth_scores = []
    cur_value_data_list = []
    cur_length_list = []
    convkernel_1 = np.ones(2 * radius + 1)
    for log_dir in file_list:
        data_log = event_accumulator.EventAccumulator(log_dir)
        data_log.Reload()
        try:
            value_log = data_log.scalars.Items(items)
        except KeyError:
            return
        value_data = [i.value for i in value_log]
        cur_value_data_list.append(value_data)
        cur_length_list.append(len(value_data))
    cur_min_length = min(cur_length_list)
    print(cur_min_length)
    # smooth_scores.append(value_data[:length])
    for v_data in cur_value_data_list:
        _smooth_v1_data = np.convolve(v_data, convkernel_1, mode='same') \
                          / np.convolve(np.ones_like(v_data), convkernel_1, mode='same')
        smooth_scores.append(_smooth_v1_data[:cur_min_length])

    x_data = np.arange(0, cur_min_length) / cur_min_length * (million * cur_min_length / length)
    ymean = np.mean(smooth_scores, axis=0)
    ystd = np.std(smooth_scores, axis=0)
    ystderr = ystd / np.sqrt(len(smooth_scores))
    plt.plot(x_data, ymean, color=color, linestyle='-', label=condition)
    plt.fill_between(x_data, ymean - ystderr / 1.5, ymean + ystderr / 1.5, color=color, alpha=.2)

def plot_option(env_dir, alg_names, item_name, y_label, length, million, COLORS):
    color_idx=0
    for alg_name, alg_params in alg_names.items():
        for alg_prefix in alg_params:
            files = collect_tf_file(env_dir + alg_name, alg_prefix)
            print(files)
            plot_sns_img(files, '{}'.format(item_name), COLORS[color_idx], alg_prefix, length, million)
            color_idx += 1


    plt.xlabel('Training steps (Million)', fontsize=16)
    plt.ylabel(y_label, fontsize=16)
    plt.legend(loc=4, fontsize=10)
    plt.grid(ls='--')
    plt.rc('grid', linestyle='-.')
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

envs_dict = {}
# Asterix
temp_alg_dict = {
                 'DQN': ['gpu_DQN_lr=0.00025'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9',
                           'gpu_ALDQN_alpha=0.8',
                           'gpu_ALDQN_alpha=0.7'],
                 # 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8',
                               'gpu_clipALDQN_alpha=0.8_ratio=0.8'],
                 # 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8'],
                 'genALDQN': ['gpu_genALDQN_alpha=0.9_ratio=0.8_beta=0.7']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Asterix-MinAtar-v0'] = temp_alg_names

# Breakout
temp_alg_dict = {
                 'DQN': ['gpu_DQN_lr=0.00025'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9',
                           'gpu_ALDQN_alpha=0.8',
                           'gpu_ALDQN_alpha=0.7'],
                 # 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8',
                               'gpu_clipALDQN_alpha=0.8_ratio=0.8'],
                 # 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8'],
                 'genALDQN': ['gpu_genALDQN_alpha=0.9_ratio=0.8_beta=0.7']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Breakout-MinAtar-v0'] = temp_alg_names

# Freeway
temp_alg_dict = {
                 'DQN': ['gpu_DQN_lr=0.00025'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9',
                           'gpu_ALDQN_alpha=0.8' ],
                 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8'],
                 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Freeway-MinAtar-v0'] = temp_alg_names

# Space_invaders
temp_alg_dict = {
                 'DQN': ['gpu_DQN_lr=0.00025'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9',
                           'gpu_ALDQN_alpha=0.8',
                           'gpu_ALDQN_alpha=0.7'],
                 # 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8',
                               'gpu_clipALDQN_alpha=0.8_ratio=0.8'],
                 # 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8']
                 'genALDQN': ['gpu_genALDQN_alpha=0.9_ratio=0.8_beta=0.7']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Space_invaders-MinAtar-v0'] = temp_alg_names

# Seaquest
temp_alg_dict = {
                 'DQN': ['gpu_DQN_lr=0.00025'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9',
                           'gpu_ALDQN_alpha=0.8',
                           'gpu_ALDQN_alpha=0.7'],
                 # 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8',
                               'gpu_clipALDQN_alpha=0.8_ratio=0.8'],
                 # 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8'],
                 'genALDQN': ['gpu_genALDQN_alpha=0.9_ratio=0.8_beta=0.7']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Seaquest-MinAtar-v0'] = temp_alg_names

# LunarLander
temp_alg_dict = {
                 'DQN': ['gpu_DQN'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9'],
                 # 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8'],
                 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['LunarLander-v2'] = temp_alg_names

# Catcher
temp_alg_dict = {
                 'DQN': ['gpu_DQN'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9_'],
                 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8'],
                 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Catcher-PLE-v0'] = temp_alg_names

# Pixelcopter
temp_alg_dict = {
                 'DQN': ['gpu_DQN'],
                 'ALDQN': ['gpu_ALDQN_alpha=0.9_'],
                 'PALDQN': ['gpu_PALDQN_alpha=0.9'],
                 'clipALDQN': ['gpu_clipALDQN_alpha=0.9_ratio=0.8'],
                 'clipPALDQN': ['gpu_clipPALDQN_alpha=0.9_ratio=0.8']
                 }
temp_alg_names = collections.OrderedDict(**temp_alg_dict)
envs_dict['Pixelcopter-PLE-v0'] = temp_alg_names

# plot params tabular
COLORS = ['red', 'green', 'blue', 'm', 'magenta', 'brown', 'yellow', 'purple', 'orange']
length_dict = {'LunarLander-v2': [301, 1.5],
               'Catcher-PLE-v0': [701, 3.5],
               'Pixelcopter-PLE-v0': [401, 2],
               'Asterix-MinAtar-v0': [1001, 5],
               'Breakout-MinAtar-v0': [1001, 5],
               'Freeway-MinAtar-v0': [1001, 5],
               'Space_invaders-MinAtar-v0': [1001, 5],
               'Seaquest-MinAtar-v0': [1001, 5]
               }

# begin plot
envs = ['LunarLander-v2', 'Catcher-PLE-v0', 'Pixelcopter-PLE-v0', 'Asterix-MinAtar-v0',
        'Breakout-MinAtar-v0', 'Freeway-MinAtar-v0', 'Space_invaders-MinAtar-v0',
        'Seaquest-MinAtar-v0']
env = envs[7]

length = length_dict[env][0]
million = length_dict[env][1]

alg_names = envs_dict[env]

# public logs files dir
env_dir = '../all_logs/Explorer/' + env + '/'

item_names = ['Y_mean', 'action_gap']
y_labels = ['Q value estimation', 'Action gaps']

for item_name, y_label in zip(item_names, y_labels):

    plt.figure(figsize=(16, 8), dpi=80)
    # plot averaged return performance
    pic1 = plt.subplot(121)
    plot_option(env_dir, alg_names, 'reward/eval_episodes_mean', 'Average episode Return', length, million, COLORS)
    # plot the analysis plot
    pic2 = plt.subplot(122)
    plot_option(env_dir, alg_names, 'analysis_log/{}'.format(item_name), y_label, length, million, COLORS)

    save_path = os.path.join('./logs/plots/' + item_name + '/')
    make_dir(save_path)
    # plt.savefig(save_path + env + '.png', figsize=(10, 15))
    plt.xlim()
    plt.show()


