import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D
from collections import OrderedDict, defaultdict
import os
from functools import partial

from scipy.stats.mstats import zscore
from sklearn.base import clone
from sklearn.utils import check_random_state
from spectral_dagger.utils import run_experiment_and_plot
from spectral_dagger.sequence import GaussianHMM

from seq_lda.algorithms import MSSG, SingleMSSG, OneByOne, Aggregate
from seq_lda import (
    generate_multitask_sequence_data,
    RMSE_score, log_likelihood_score)


PATH = '/data/seq_lda/weather'
data_directory = '/data/seq_lda/'


with open(os.path.join(PATH, 'HEADERS.txt'), 'r') as f:
    f.readline()
    HEADERS = f.readline().strip().split()


def random_sample_with_min_count(years, min_seq_count, n_tasks, process_data, rng=None):
    station_years = [os.path.join(y, s) for y in years for s in os.listdir(os.path.join(PATH, y))]
    rng = check_random_state(rng)
    chosen = OrderedDict()

    while len(chosen) < n_tasks and station_years:
        sy = rng.choice(station_years)
        df = pd.read_csv(os.path.join(PATH, sy), sep='\s+', names=HEADERS, header=None)

        date = pd.to_datetime(df['UTC_DATE'], format="%Y%m%d")
        timedelta = pd.to_timedelta(60 * df['UTC_TIME'] / 100, unit='m')
        df['timestamp'] = date + timedelta

        sequences = process_data(df)
        if len(sequences) >= min_seq_count:
            chosen[sy] = sequences

        station_years.remove(sy)

    if len(chosen) < n_tasks:
        raise Exception("Sampling cannot be satisfied.")

    return chosen


def make_process_data(max_sample_rate, horizon=None, standardize=False, delta=False, fields=None):
    #default_fields = 'timestamp T_CALC P_CALC SUR_TEMP RH_HR_AVG'.split()
    default_fields = 'timestamp T_CALC SUR_TEMP RH_HR_AVG'.split()
    fields = fields or default_fields

    def process_data(df):
        df = df[fields]
        df = df.set_index(pd.DatetimeIndex(df['timestamp']))
        df = df.resample(max_sample_rate).mean().dropna()
        g = df.groupby([df.index.year, df.index.month, df.index.day])
        sequences = [(np.array(df.loc[seq])[:horizon]).copy() for seq in g.groups.values()]
        if horizon:
            sequences = [s for s in sequences if len(s) == horizon]
        if delta:
            sequences = [np.diff(s, axis=0) for s in sequences]
        if standardize:
            sequences = [zscore(seq, axis=0) for seq in sequences]
        return sequences

    return process_data


def to_hashable(x):
    x = x.copy()
    x.flags.writeable = False
    return hash(x.data)


def generate_weather_data(
        years=None,
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
        delta=False,
        random_state=None):

    if not years:
        years = [y for y in os.listdir(PATH) if y.startswith('20')]
    else:
        years = [str(y) for y in years]

    if horizon == 0 or horizon == np.inf:
        raise Exception("Horizon (%d) must be finite and positive." % horizon)

    process_data = make_process_data(
        max_sample_rate, horizon, standardize=standardize, delta=delta)
    random_state = check_random_state(random_state)

    n_seqs_per_task = max(
        core_train_wpt + core_test_wpt,
        transfer_train_wpt + transfer_test_wpt,
        test_wpt)

    n_tasks = n_core_tasks + n_transfer_tasks + n_test_tasks

    data = random_sample_with_min_count(
        years, n_seqs_per_task, n_tasks, process_data, rng=random_state)

    data = data.values()

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
        name="weather",
        n_core_tasks=0,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=15,
        core_test_wpt=50,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        max_sample_rate=None,
        x_var_max=31,
        x_var_min=1,
        x_var_step=2,
        n_repeats=10,
        horizon=4,
        standardize=1,
        delta=0,
        hmm_verbose=0,
        lda_verbose=1,
        random_state=None,
        use_time=1,
        directory=data_directory):

    max_sample_rate = max_sample_rate or "1h"  # default 1 hour
    data_kwargs = locals().copy()
    non_data = ('random_state hmm_verbose lda_verbose name use_time directory '
                'x_var_min x_var_max x_var_step n_repeats plot show_data_stats')
    for attr in non_data.split():
        data_kwargs.pop(attr, None)

    ghmm = GaussianHMM(
        covariance_type='diag', min_covar=1e-3,
        startprob_prior=1.0, transmat_prior=1.0,
        means_prior=0, means_weight=0,
        covars_prior=1e-2, covars_weight=1,
        algorithm="viterbi", n_iter=10, tol=1e-2, verbose=hmm_verbose,
        params="stmc", init_params="stmc")

    mixture = SingleMSSG(
        bg=clone(ghmm),
        n_samples_scale=10,
        verbose=lda_verbose, name="Mixture",
        to_hashable=to_hashable)

    estimators = [
        MSSG(
            bg=clone(ghmm),
            n_samples_scale=10,
            verbose=lda_verbose, name="GaussianHMM/MSSG"),
        OneByOne(bg=clone(mixture), name="GaussianHMM/MSSG1x1"),
        OneByOne(bg=clone(ghmm), name="GaussianHMM/1x1"),
        Aggregate(bg=clone(mixture), name="GaussianHMM/MSSGAgg"),
        Aggregate(bg=clone(ghmm), name="GaussianHMM/Agg")
    ]

    if n_transfer_tasks > 0:
        name += "_transfer"
        estimators.extend([
            Aggregate(bg=clone(mixture), add_transfer_data=True, name="GaussianHMM/MSSGAgg,add_transfer"),
            Aggregate(bg=clone(ghmm), add_transfer_data=True, name="GaussianHMM/Agg,add_transfer")
        ])
    data_generator = generate_weather_data

    _log_likelihood_score = partial(
        log_likelihood_score, string=False)
    _log_likelihood_score.__name__ = "log_likelihood"

    x_var_values = range(x_var_min, x_var_max + 1, x_var_step)

    exp_kwargs = dict(
        mode='data', base_estimators=estimators,
        generate_data=data_generator,
        data_kwargs=data_kwargs,
        search_kwargs=dict(n_iter=10),
        directory=directory,
        score=[RMSE_score, _log_likelihood_score],
        x_var_name='n_core_tasks',
        name=name,
        x_var_values=x_var_values,
        use_time=use_time,
        n_repeats=n_repeats)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[2, 3, 4], n_repeats=2, search_kwargs=dict(n_iter=2))

    score_display = [
        'RMSE',
        'Log Likelihood',
        'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)


if __name__ == "__main__":
    from clify import command_line
    command_line(main)()
