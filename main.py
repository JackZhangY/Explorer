import os
import sys
import argparse

from utils.sweeper import Sweeper
from utils.helper import make_dir
from experiment import Experiment

def main(config_file, config_idx, seed=1, gpu=False):
  parser = argparse.ArgumentParser(description="Config file")
  parser.add_argument('--config_file', type=str, default='./configs/{}'.format(config_file), help='Configuration file for the chosen model')
  parser.add_argument('--config_idx', type=int, default=config_idx, help='Configuration index')
  args = parser.parse_args()
  
  sweeper = Sweeper(args.config_file)
  cfg = sweeper.generate_config_for_idx(args.config_idx)
  
  # Set config dict default value
  cfg.setdefault('network_update_frequency', 1)
  cfg['env'].setdefault('max_episode_steps', -1)
  cfg.setdefault('show_tb', False)
  cfg.setdefault('render', False)
  cfg.setdefault('gradient_clip', -1)
  cfg.setdefault('hidden_act', 'ReLU')
  cfg.setdefault('output_act', 'Linear')
  cfg.setdefault('num_eval_episodes', 10)

  # 强制根据函数参数设置seed
  cfg['generate_random_seed'] = False
  cfg['seed'] = seed
  
  # Set experiment namd and log paths
  cfg['exp'] = args.config_file.split('/')[-1].split('.')[0]

  if cfg['agent']['name'] == 'DQN' or cfg['agent']['name'] == 'DDQN':
    logs_dir = f"../all_logs/Explorer-v2/{cfg['env']['name']}/{cfg['agent']['name']}/{cfg['agent']['name']}_" \
               f"lr={cfg['optimizer']['kwargs']['lr']}_{cfg['seed']}"
  elif cfg['agent']['name'] == 'AveragedDQN' or cfg['agent']['name'] == 'MaxminDQN':
    logs_dir = f"../all_logs/Explorer-v2/{cfg['env']['name']}/{cfg['agent']['name']}/{cfg['agent']['name']}_" \
               f"lr={cfg['optimizer']['kwargs']['lr']}_N={cfg['agent']['target_networks_num']}_{cfg['seed']}"
  elif cfg['agent']['name'] == 'AdamDQN':
    logs_dir = f"../all_logs/Explorer-v2/{cfg['env']['name']}/{cfg['agent']['name']}/{cfg['agent']['name']}_" \
               f"lr={cfg['optimizer']['kwargs']['lr']}_eps={cfg['optimizer']['kwargs']['eps']}_{cfg['seed']}"
  elif cfg['agent']['name'] == 'MunchausenDQN':
    logs_dir = f"../all_logs/Explorer-v2/{cfg['env']['name']}/{cfg['agent']['name']}/{cfg['agent']['name']}_" \
               f"tau={cfg['agent']['entropy_temperature']}_alpha={cfg['agent']['log_reward_scaling']}_" \
               f"clip={cfg['agent']['log_reward_clipping']}_{cfg['seed']}"
  elif cfg['agent']['name'] == 'AvgMunchausenDQN':
    logs_dir = f"../all_logs/Explorer-v2/{cfg['env']['name']}/{cfg['agent']['name']}/{cfg['agent']['name']}_" \
               f"tau={cfg['agent']['entropy_temperature']}_alpha={cfg['agent']['log_reward_scaling']}_" \
               f"clip={cfg['agent']['log_reward_clipping']}_num_pi={cfg['agent']['num_policies']}_{cfg['seed']}"
  elif cfg['agent']['name'] == 'VanillaDQN':
    logs_dir = f"../all_logs/Explorer-v2/{cfg['env']['name']}/{cfg['agent']['name']}/{cfg['agent']['name']}_" \
               f"_{cfg['seed']}"
  else:
    raise Exception('logs_dir is empty!!!!')

  # determinate whether use gpu or not
  if gpu:
    cfg['device'] = 'cuda'
    base_path, file_name = os.path.split(logs_dir)
    file_name = 'gpu_' + file_name
    logs_dir = os.path.join(base_path, file_name)
    logs_dir += '/'
  else:
    cfg['device'] = 'cpu'
    logs_dir += '/'

  train_log_path = logs_dir + 'result_Train.feather'
  test_log_path = logs_dir + 'result_Test.feather'
  model_path = logs_dir + 'model.pt'
  cfg_path = logs_dir + 'config.json'
  cfg['logs_dir'] = logs_dir
  cfg['train_log_path'] = train_log_path
  cfg['test_log_path'] = test_log_path
  cfg['model_path'] = model_path
  cfg['cfg_path'] = cfg_path

  make_dir(cfg['logs_dir'])

  print(logs_dir)
  exp = Experiment(cfg)
  exp.run()

# if __name__=='__main__':
#   main(sys.argv)