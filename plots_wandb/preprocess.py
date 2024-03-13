import wandb
import pandas as pd


def download(metrics, runs):
    files = {}
    for index, run in enumerate(runs):
        data = api.run(run)
        for metric in metrics:
            filename = f"{index}"
            if metric in files:
                if filename in files[metric]:
                    files[metric][filename] = files[metric][filename].append(getattr(data.history(samples=20000), metric).dropna())
                else:
                    files[metric][filename] = getattr(data.history(samples=20000), metric).dropna()
            else:
                files[metric] = {filename: getattr(data.history(samples=20000), metric).dropna()}
        iterations = data.history(samples=20000).global_iterations.dropna()

    concatenated_metric = {}
    for metric, metric_dict in files.items():
        concatenated_metric[metric] = pd.concat(metric_dict.values(), axis=1)
    return concatenated_metric, iterations


def preprocess(env, experiment, metrics, runs):
    data_files, iterations_file = download(metrics, runs)
    for metric in metrics:
        df = data_files[metric]
        # fill missing values
        df = df.fillna(df.rolling(6, min_periods=1).mean())
        # create avg/min/max values
        df['avg'] = df.mean(axis=1)
        df['min'] = df.min(axis=1)
        df['max'] = df.max(axis=1)
        df = df.drop([metric], axis=1)
        # add iterations, drop new NaN's and set as index
        df = pd.concat([df, iterations_file], axis=1)
        df = df.dropna()
        df.set_index('global_iterations', inplace=True)
        # smoothing
        window_size=500 # 50 but for episodic return 500
        df['avg'] = df['avg'].rolling(window=window_size).mean()
        df['min'] = df['min'].rolling(window=window_size).mean()
        df['max'] = df['max'].rolling(window=window_size).mean()
        df = df.dropna()
        # save df
        filename = f"{env}_{experiment}_{metric}"
        df.to_csv(f"data/_{filename}.csv", sep=',', index=True, encoding='utf-8')

if __name__ == '__main__':
    """
    Create a preprocessed CSV with the avg/min/max data of a specific metric of a specific experiment
    
    Env = the game environment, where
        QB = QBert
        BO = BreakOut
        SI = SpaceInvaders
        In case of Atari games append with _1/2/3 depending on the atari run
    Experiment can be standard, layernorm, or boundedloss-br0.3 (br3, ar0.3, ar3) or incase of RF reincarnate
    Metrics is a list of the metrics from the experiments to preprocess
    Runs is a list of the paths to the run's in wandb
    """

    api = wandb.Api()

    env = 'BO'
    experiment = 'standard'
    metrics = ['episodic_return']
    runs = ['demipeek/project_relu_demi/w8m0kzgm']
    preprocess(env, experiment, metrics, runs)
