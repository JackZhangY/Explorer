import multiprocessing as mp
import warnings

from main import main

warnings.filterwarnings("ignore")


def run(config_file, config_idx, seed, gpu=False):
    main(config_file=config_file, config_idx=config_idx, seed=seed, gpu=gpu)


def train(config_file, config_idx, seeds, gpu=False):
    process = []
    for seed in seeds:
        p = mp.Process(target=run, args=(config_file, config_idx, seed, gpu))
        p.start()
        process.append(p)
    for p in process:
        p.join()


if __name__ == '__main__':
    mp.set_start_method('spawn')
    # clip = -0.25
    # train(config_file='MunchausenDQN/lunarlander.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    # train(config_file='MunchausenDQN/catcher.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    # train(config_file='MunchausenDQN/copter.json', config_idx=1, seeds=[1, 2, 3, 4, 5])

    # fintune alpha(0.8, 0.95) with clip=-0.5
    # train(config_file='MunchausenDQN/lunarlander.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    # train(config_file='MunchausenDQN/catcher.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    # train(config_file='MunchausenDQN/copter.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    #
    # train(config_file='MunchausenDQN/lunarlander.json', config_idx=2, seeds=[1, 2, 3, 4, 5])
    # train(config_file='MunchausenDQN/catcher.json', config_idx=2, seeds=[1, 2, 3, 4, 5])
    # train(config_file='MunchausenDQN/copter.json', config_idx=2, seeds=[1, 2, 3, 4, 5])

    # train(config_file='AvgMunchausenDQN/lunarlander.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    # train(config_file='AvgMunchausenDQN/copter.json', config_idx=1, seeds=[1, 2, 3, 4, 5])
    # train(config_file='AvgMunchausenDQN/catcher.json', config_idx=1, seeds=[1, 2, 3, 4, 5])

    train(config_file='minatar.json', config_idx=1, seeds=[1, 2, 3, 4, 5], gpu=True)
    train(config_file='minatar.json', config_idx=2, seeds=[1, 2, 3, 4, 5], gpu=True)
    train(config_file='minatar.json', config_idx=3, seeds=[1, 2, 3, 4, 5], gpu=True)
    train(config_file='minatar.json', config_idx=4, seeds=[1, 2, 3, 4, 5], gpu=True)
    train(config_file='minatar.json', config_idx=5, seeds=[1, 2, 3, 4, 5], gpu=True)

    # train(config_file='minatar.json', config_idx=1, seeds=[1])
    # train(config_file='minatar.json', config_idx=2, seeds=[1, 2, 3, 4, 5])
    # train(config_file='minatar.json', config_idx=3, seeds=[1, 2, 3, 4, 5])
    # train(config_file='minatar.json', config_idx=4, seeds=[1, 2, 3, 4, 5])
    #