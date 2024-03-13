import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def make_legend_name(name, experiments, metrics):
    words = name.split('_')
    transformed_words = []
    if len(experiments) > 1: # if the plot shows more than one experiment they need to be specified in the legend.
        if words[0] == 'layernorm':
            transformed_words.append('Layer Normalization')
        elif words[1] == 'br0.3':
            transformed_words.append('Bounded Loss Before ReLU 0.3')
        elif words[1] == 'br3':
            transformed_words.append('Bounded Loss Before ReLU 3')
        elif words[1] == 'ar0.3':
            transformed_words.append('Bounded Loss After ReLU 0.3')
        elif words[1] == 'ar3':
            transformed_words.append('Bounded Loss After ReLU 3')
        elif words[1] == 'dimension' or words[1] == 'rate' or words[1] == 'scale':
            for word in words[:3]:
                transformed_words.append(word.capitalize())
        else:
            transformed_words.append(words[0:-1].capitalize())
    if len(metrics) > 1:
        for word in words[1:]:
            transformed_words.append(word.capitalize())
    transformed_name = ' '.join(transformed_words)
    return transformed_name


def make_y_label(metric):
    metric_split = metric.split('_')
    if metric == 'episodic_return':
        y = 'Episodic Return'
    elif metric_split[0] == 'std':
        y = 'Standard Deviation'
    elif metric_split[0] == 'ratio':
        y = 'Ratio Negative Numbers'
    elif metric_split[-1] == 'reward':
        y = 'Average Reward'
    elif metric_split[-1] == 'mean':
        y = 'Mean'
    elif metric_split[1] == 'loss':
        y = 'Q-Loss'
    elif metric_split[1] == 'accuracy':
        y = 'Accuracy'
    else:
        y = 'Fraction Dead Neurons' # elif metric_split[-1] == 'latent' or metric_split[-1] == 'relu':
    return y

def get_data(env, experiment, metric):
    filename = f"{env}_{experiment}_{metric}.csv"
    data_df = pd.read_csv(f"data/_{filename}", index_col='global_iterations')
    return data_df


def plot(env, experiments, metrics, title_type):
    all_data = {}
    for experiment in experiments:
        for metric in metrics:
            data_for_plots = get_data(env, experiment, metric)
            all_data[f"{experiment}_{metric}"] = data_for_plots

    # Generate a list of distinct colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    fig, ax = plt.subplots(figsize=(6, 4))

    legend_handles = []
    legend_labels = []
    metric_nr = 0
    for (name, data), color in zip(all_data.items(), colors):
        # Set Y-axis and legend label
        legend_name = make_legend_name(name, experiments, metrics)
        metric = '_'.join(name.split('_')[1:])  # dit werkt niet helemaal optimaal want boundedloss_br3 etc split hij dus ook op! nu niet erg maar wel op letten

        ylabel = make_y_label(metric)

        if len(metrics) == 1 or metrics[0].split('_')[0] == metrics[1].split('_')[0] or metrics[0].split('_')[1] == metrics[1].split('_')[1]:
            # Plot the filled area between min and max
            ax.fill_between(data.index, data['min'], data['max'], color=color, alpha=0.1)  # label='Min-Max Range'
            # Plot the average line with a different color and a thicker line
            line, = ax.plot(data.index, data['avg'], color=color, linewidth=2, label=f'{legend_name}')
            ax.set_ylabel(ylabel)

            if metrics[0].split('_')[0] == 'fraction':
                ax.set_ylim(0, 1)
            # elif metrics[0].split('_')[1] == 'accuracy':
            #     ax.set_ylim(0, 1)
        else:
            if metric_nr == 0:
                ax.fill_between(data.index, data['min'], data['max'], color=color, alpha=0.1)  # label='Min-Max Range'
                line, = ax.plot(data.index, data['avg'], color=color, linewidth=2, label=f'{legend_name}')
                ax.set_ylabel(ylabel)

                if metrics[metric_nr].split('_')[0] == 'fraction':
                    ax.set_ylim(0,1)
                # if metrics[0].split('_')[1] == 'accuracy':
                #     ax.set_ylim(0, 1)
            else:
                ax2 = ax.twinx()
                ax2.fill_between(data.index, data['min'], data['max'], color=color, alpha=0.1)  # label='Min-Max Range'
                line, = ax2.plot(data.index, data['avg'], color=color, linewidth=2, label=f'{legend_name}')
                ax2.set_ylabel(ylabel)

                if metrics[metric_nr] == 'episodic_return' and env.split('_')[0] == 'BO':
                    ax2.set_ylim(0, 600)
                elif metrics[metric_nr] == 'episodic_return' and env.split('_')[0] == 'SI':
                    ax2.set_ylim(0, 5000)

        legend_handles.append(line)
        legend_labels.append(legend_name)
        metric_nr += 1
        ax.set_xlim(data.index.min(), data.index.max())

    # Add labels and title
    ax.set_xlabel('Global Iterations')
    if len(experiments) > 1 or len(metrics) > 1:
        ax.legend(legend_handles, legend_labels, loc='best') # , fontsize=7 alleen fontsize voor die 6 experimenten bij elkaar

    ax.grid(False)

    # Save the plot
    experiments_str = '_'.join(experiments)
    metrics_str = '_'.join(metrics)
    plotname = '_'.join([env, experiments_str, metrics_str])
    plt.savefig(f"plots/{plotname}.png")
    # Show the plot
    plt.show()


if __name__ == '__main__':
    """
    Create a plot based on the CSV's for specific experiments/metrics

    Env = the game environment, where
        QB = QBert
        BO = BreakOut
        SI = SpaceInvaders
        RF = Reward Finding
        In case of Atari games append with _1/2/3 depending on the atari run
    Experiment can be 'standard', 'boundedloss_br0.3', 'boundedloss_ar0.3', 'boundedloss_br3', 'boundedloss_ar3', 'layernorm'
    Metrics is a list of the metrics from the experiments to preprocess
    """

    env = 'BO'
    experiments = ['standard']
    metrics = ['fraction_weighted_dead_latent_2', 'fraction_fully_dead_latent_2']
    title_type = 'env'
    plot(env, experiments, metrics, title_type)
