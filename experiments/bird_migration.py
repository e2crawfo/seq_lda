import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
from collections import OrderedDict
import os
from functools import partial

from scipy.stats.mstats import zscore
from sklearn.utils import check_random_state
from spectral_dagger.utils import run_experiment_and_plot
from clify import command_line

from seq_lda.algorithms import GmmHmmMSSG, GmmHmm1x1, GmmHmmAgg
from seq_lda import (
    generate_multitask_sequence_data,
    RMSE_score, log_likelihood_score)


data_path = '/data/bird_migration/data/'
result_directory = '/data/seq_lda/'


def do_interpolate(data, k=1, plot=False):
    data = data[:, [0, 1, 3]]

    # perform interpolation
    # tck, u = interpolate.splprep(data, k=k)
    # u = np.linspace(0, 1, len(data))
    # interpolated_points = interpolate.splev(u, tck)

    if plot:
        do_plot(data=data)
        #do_plot(spline=interpolated_points, data=data)


def do_plot(spline=None, data=None, title=""):
    lat = data[:, 0]
    lon = data[:, 1]
    height = data[:, 2]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    colors = cm.get_cmap('YlOrRd')(np.linspace(0, 1, data.shape[0]))
    ax.scatter(lat, lon, height, c=colors)
    ax.set_xlabel('lat')
    ax.set_ylabel('lon')
    ax.set_zlabel('z')
    # if spline is not None:
    #     ax.plot(spline[0], spline[1], spline[2], marker='<', c='red')
    plt.title(title)
    plt.show()


def random_sample_with_min_count(indices, n, minimum, process_data, rng=None):
    """ Pick a bird for each task, process the data.

    process_data: Function
        A function that accepts a dataframe and returns a list of list of Observations,
        each corresponding to a sequence.

    """
    indices = indices[:]  # make a copy so we can modify it
    rng = check_random_state(rng)

    chosen = OrderedDict()

    while len(chosen) < n and indices:
        idx = rng.choice(indices)
        df = pd.read_csv(os.path.join(data_path, str(idx)))
        sequences = process_data(df)
        if len(sequences) >= n:
            chosen[idx] = sequences

        indices.remove(idx)

    if len(df) < n:
        raise Exception("Sampling cannot be satisfied.")

    return chosen


def make_process_data(max_sample_rate, horizon=None, standardize=False):
    def process_data(df):
        df = df['timestamp location-long location-lat ground-speed height-above-ellipsoid'.split()]
        df = df.set_index(pd.DatetimeIndex(df['timestamp']))
        df = df.resample(max_sample_rate).mean().dropna()
        g = df.groupby([df.index.year, df.index.month, df.index.day])
        sequences = [(np.array(df.loc[seq])[:horizon]).copy() for seq in g.groups.values()]

        if standardize:
            sequences = [zscore(seq, axis=0) for seq in sequences]
        return sequences

    return process_data


def generate_bird_migration_data(
        n_core_tasks=0,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=0,
        core_test_wpt=0,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        max_sample_rate=None,
        horizon=24,
        standardize=False,
        random_state=None):
    """ Generate data from the bird migration dataset.

        Each task is an animal. Each sequence is a day.

    """
    if horizon == 0 or horizon == np.inf:
        raise Exception("Horizon (%d) must be finite and positive." % horizon)

    indices = [int(s) for s in os.listdir(data_path)]
    process_data = make_process_data(max_sample_rate, horizon, standardize=standardize)
    random_state = check_random_state(random_state)

    n_seqs_per_task = max(
        core_train_wpt + core_test_wpt,
        transfer_train_wpt + transfer_test_wpt,
        test_wpt)

    n_tasks = n_core_tasks + n_transfer_tasks + n_test_tasks

    data = random_sample_with_min_count(
        indices, n_seqs_per_task, n_tasks, process_data, rng=random_state)

    data = data.values()

    # Makes the ndarrays hashable.
    for task_data in data:
        for seq in task_data:
            seq.flags.writeable = False
    to_hashable = lambda x: hash(x.data)

    train_data, test_data = generate_multitask_sequence_data(
        data, n_core_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=random_state,
        context=dict(learn_halt=False),
        to_hashable=to_hashable)

    return (train_data, test_data,
            dict(max_topics=6,
                 max_states=20, max_states_per_topic=5))


def main(
        n_core_tasks=0,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=15,
        core_test_wpt=200,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        max_sample_rate=None,
        horizon=24,
        hmm_verbose=0,
        lda_verbose=1,
        random_state=None,
        standardize=0,
        plot=None,
        show_data_stats=0,
        run_func=None):

    run_func = run_func or run_experiment_and_plot
    max_sample_rate = max_sample_rate or "1h"  # default 1 hour
    data_kwargs = locals()
    del data_kwargs['run_func']
    del data_kwargs['show_data_stats']
    del data_kwargs['plot']
    del data_kwargs['random_state']
    del data_kwargs['hmm_verbose']
    del data_kwargs['lda_verbose']

    if plot is not None:
        # Plot should be a string that evaluates to a dict that maps
        # animal indices to day indices to be plotted.
        assert isinstance(plot, str)
        plot = eval(plot)
        assert isinstance(plot, dict)
        process_data = make_process_data(max_sample_rate, horizon, standardize=standardize)

        for animal_idx, day_indices in plot.items():
            df = pd.read_csv(os.path.join(data_path, str(animal_idx)))
            sequences = process_data(df)

            if day_indices:
                for day_idx in day_indices:
                    do_interpolate(sequences[day_idx], plot=True)
            else:
                do_interpolate(np.vstack(sequences), plot=True)
        return

    if show_data_stats:
        process_data = make_process_data(max_sample_rate, horizon, standardize=standardize)

        indices = sorted([int(s) for s in os.listdir(data_path)])
        for idx in indices:
            df = pd.read_csv(os.path.join(data_path, str(idx)))
            sequences = process_data(df)

            print "Sequence lengths for idx: {0}:".format(idx)
            print "{}".format([len(s) for s in sequences])

        return

    random_state = np.random.RandomState(101)

    n_dim = 4

    data_generator = generate_bird_migration_data

    estimators = [
        GmmHmmAgg(
            bg_kwargs=dict(
                n_dim=n_dim, n_components=1,
                max_iter=1000, thresh=1e-4, verbose=0, cov_type='full',
                careful=True)),
        GmmHmm1x1(
            bg_kwargs=dict(
                n_dim=n_dim, n_components=1,
                max_iter=1000, thresh=1e-4, verbose=0, cov_type='full',
                careful=True)),
        GmmHmmMSSG(
            n_samples_scale=2,
            bg_kwargs=dict(
                n_dim=n_dim, n_components=1,
                max_iter=1000, thresh=1e-4, verbose=0, cov_type='full',
                careful=True),
            verbose=lda_verbose)]

    _log_likelihood_score = partial(
        log_likelihood_score, string=False)
    _log_likelihood_score.__name__ = "log_likelihood"

    exp_kwargs = dict(
        mode='data', base_estimators=estimators,
        generate_data=data_generator,
        data_kwargs=data_kwargs,
        search_kwargs=dict(n_iter=10),
        directory=result_directory,
        score=[RMSE_score, _log_likelihood_score],  # , one_norm_score],
        x_var_name='n_core_tasks',
        x_var_values=range(1, 21, 2),
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[2, 3, 4], n_repeats=2, search_kwargs=dict(n_iter=2))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_func(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)


if __name__ == "__main__":
    command_line(main)()
