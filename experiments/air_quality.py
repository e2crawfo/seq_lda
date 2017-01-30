import numpy as np
import pandas as pd
from collections import defaultdict
import six
from sklearn.decomposition import PCA
import argparse
from functools import partial

from spectral_dagger.datasets.air_quality import load_data
from spectral_dagger.utils import run_experiment_and_plot

from seq_lda.algorithms import GmmHmmMSSG, GmmHmm1x1, GmmHmmAgg
from seq_lda import (
    generate_multitask_sequence_data, RMSE_score, log_likelihood_score)

random_state = np.random.RandomState(2)


def generate_air_quality_data(
        n_train_tasks=12,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=0,
        core_test_wpt=0,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        difference=True,
        whiten=True,
        sample_every=1,
        horizon=np.inf,
        random_state=None):

    df = load_data()

    df = df.replace(-200, np.nan)
    df = df.interpolate()
    df = df.set_index('Date_Time')

    mn = df.index.min()
    mn -= pd.DateOffset(hours=mn.hour)

    mx = df.index.max()
    mx += pd.DateOffset(hours=24-mx.hour)

    data = defaultdict(list)

    for day in pd.date_range(mn, mx, freq='D'):
        key = '{year}_{month}'.format(year=day.year, month=day.month)
        sequence = df.loc[day:day+pd.DateOffset(hours=23, minutes=59), :]
        sequence = np.ascontiguousarray(sequence.as_matrix())
        sequence.flags.writeable = False

        if sequence.shape[0]:
            data[key].append(sequence)

    data = [v for k, v in six.iteritems(data)]
    data = data[1:-1]  # Get rid of first and last months since they won't have enough sequences.

    to_hashable = lambda x: hash(x.data)

    train_data, test_data = generate_multitask_sequence_data(
        data, n_train_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=random_state,
        to_hashable=to_hashable,
        context=dict(learn_halt=False, n_input=13))

    all_seqs = []
    for task in train_data.data.as_sequences():
        for seq in task:
            all_seqs.append(seq)
    all_seqs = np.vstack(all_seqs)

    pca = PCA(whiten=True)
    pca.fit(all_seqs)

    # train_data and test_data share a dictionary, so this modifies both datasets.
    train_data.dictionary.words = [
        pca.transform(seq) for seq in train_data.dictionary.words]

    return (train_data, test_data,
            dict(max_topics=10,
                 max_states=30,
                 max_states_per_topic=10,
                 transform=pca))


if __name__ == "__main__":
    train_data, test_data, context = generate_air_quality_data(core_train_wpt=3, core_test_wpt=3)

    hmm_verbose = False
    lda_verbose = True

    random_state = np.random.RandomState(101)

    parser = argparse.ArgumentParser()
    parser.add_argument("--core-train-wpt", type=int, default=21)
    args, _ = parser.parse_known_args()

    data_kwargs = dict(
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=args.core_train_wpt,
        core_test_wpt=1,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        difference=True,
        whiten=True,
        sample_every=1,
        horizon=np.inf,
        random_state=None)
    data_generator = generate_air_quality_data

    n_dim = 13

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
        directory='/data/seq_lda/',
        score=[RMSE_score, _log_likelihood_score],  # , one_norm_score],
        x_var_name='n_train_tasks',
        x_var_values=range(1, 12, 1),
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[2, 3], n_repeats=1, search_kwargs=dict(n_iter=2))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    # x_var_display = '\# Training Samples per Task'
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)
