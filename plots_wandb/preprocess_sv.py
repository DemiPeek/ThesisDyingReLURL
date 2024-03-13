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
                    files[metric][filename] = files[metric][filename].append(
                        getattr(data.history(samples=20000), metric).dropna())
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
        # if env == 'RF' and metric == 'average_reward':
        iterations_file = iterations_file[~iterations_file.index.duplicated(keep='first')]
        df = pd.merge(df, iterations_file, left_index=True, right_index=True, how='inner')
        # else:
        #     df = pd.concat([df, iterations_file], axis=1)
        df = df.dropna()
        df.set_index('global_iterations', inplace=True)
        df = df[~df.index.duplicated(keep='first')]
        # smoothing
        window_size = 30
        df['avg'] = df['avg'].rolling(window=window_size, min_periods=1).mean()
        df['min'] = df['min'].rolling(window=window_size, min_periods=1).mean()
        df['max'] = df['max'].rolling(window=window_size, min_periods=1).mean()
        df = df.dropna()
        # df = df[df.index.astype(str).str.endswith('000')]
        # save df
        words = metric.split(' ')
        name = []
        for word in words:
            name.append(word.lower())
        name = '_'.join(name)
        filename = f"{env}_{experiment}_{name}"
        df.to_csv(f"data/_{filename}.csv", sep=',', index=True, encoding='utf-8')


if __name__ == '__main__':
    """
    Create a preprocessed CSV with the avg/min/max data of a specific metric of a specific experiment

    Env = the game environment
    Experiment is the list of experiment types
    Metrics is a list of the metrics from the experiments to preprocess
    Runs is a list of the paths to the run's in wandb
    """

    api = wandb.Api()

    env = 'SV'
    experiment = 'mnist'
    metrics = ['Fraction Fully Dead Test', 'Fraction Weighted Dead Test',
               'Fraction Fully Dead Train', 'Fraction Weighted Dead Train',
               'Test Accuracy', 'Train Accuracy']
    runs = ['demipeek/project_relu_demi/z1i2olk1', 'demipeek/project_relu_demi/29j00i6n', 'demipeek/project_relu_demi/wy6m0764']
    preprocess(env, experiment, metrics, runs)