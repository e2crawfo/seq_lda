from __future__ import print_function
import numpy as np
from collections import defaultdict
from functools import partial
from sklearn.utils import check_random_state
import argparse

from spectral_dagger.datasets import pendigits
from spectral_dagger.utils import run_experiment_and_plot, sample_multinomial

from seq_lda.algorithms import GmmHmm1x1, GmmHmmAgg, GmmHmmMSSG
from seq_lda import (
    generate_multitask_sequence_data,
    RMSE_score, log_likelihood_score)  # , one_norm_score)


def generate_pendigit_data_sparse(
        max_tasks=44,
        alpha=0.01,
        n_train_tasks=33,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=0,
        core_test_wpt=0,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        difference=True,
        sample_every=1,
        simplify=None,
        horizon=np.inf,
        use_digits=None,
        random_state=None):

    if use_digits is None:
        use_digits = range(10)
    idx_map = {i: d for i, d in enumerate(use_digits)}
    n_topics = len(use_digits)

    rng = check_random_state(random_state)
    try:
        alpha = float(alpha)
        alpha = alpha * np.ones(n_topics)
    except:
        alpha = np.array(alpha)
    assert alpha.ndim == 1
    assert alpha.shape[0] == n_topics
    assert (alpha >= 0.0).all()

    n_tasks = n_train_tasks + n_transfer_tasks + n_test_tasks

    if n_tasks > max_tasks:
        raise ValueError(
            "Requested number of tasks (%d) is greater "
            "than the maximum number of tasks "
            "allowed (%d)." % (n_tasks, max_tasks))

    n_words_per_task = max(
        core_train_wpt + core_test_wpt,
        transfer_train_wpt + transfer_test_wpt,
        test_wpt)

    data, labels = pendigits.get_data(
        difference, sample_every=sample_every, simplify=simplify,
        ignore_multisegment=False, use_digits=use_digits)
    assert len(data) >= n_tasks

    data = data[:max_tasks]
    labels = labels[:max_tasks]

    indices = rng.permutation(range(len(data)))
    data = [data[i] for i in indices]
    labels = [labels[i] for i in indices]

    data = data[:n_tasks]
    labels = labels[:n_tasks]

    if horizon is not None and horizon is not np.inf:
        _data, _labels = [], []
        for dd, ll in zip(data, labels):
            _data.append([])
            _labels.append([])
            for d, l in zip(dd, ll):
                if len(d) >= horizon:
                    _data[-1].append(d[:horizon])
                    _labels[-1].append(l)
        data, labels = _data, _labels

    all_sequences = []
    for i in range(n_tasks):
        quick_dict = defaultdict(list)
        for d, l in zip(data[i], labels[i]):
            quick_dict[l].append(d)

        _alpha = alpha.copy()
        for l in use_digits:
            if len(quick_dict[l]) == 0:
                _alpha[l] = 0.0

        theta = rng.dirichlet(_alpha)

        sequences = []
        for j in range(n_words_per_task):
            digit = idx_map[sample_multinomial(theta, rng)]
            candidates = quick_dict[digit]
            sequence = candidates[rng.randint(len(candidates))]
            sequences.append(sequence)

        # Makes the ndarrays hashable.
        for seq in sequences:
            seq.flags.writeable = False
        all_sequences.append(sequences)

    to_hashable = lambda x: hash(x.data)


    train_data, test_data = generate_multitask_sequence_data(
        all_sequences, n_train_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=rng,
        to_hashable=to_hashable,
        context=dict(learn_halt=True, n_input=2))

    return (train_data, test_data,
            dict(max_topics=n_topics,
                 max_states=30,
                 max_states_per_topic=10))

if __name__ == "__main__":
    gmm_hmm_verbose = False
    lda_verbose = True
    #use_digits = [0, 1, 2, 3, 4]
    use_digits = [0, 2]

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--core-train-wpt", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=5)
    args, _ = parser.parse_known_args()

    def point_distribution(self, context):
        return dict()#dict(n_topics=[len(use_digits)]), n_states=range(2, int(context['max_states_per_topic'])))
    GmmHmmMSSG.point_distribution = point_distribution

    # def point_distribution(self, context):
    #     return dict()
    # GmmHmm1x1.point_distribution = point_distribution
    # GmmHmmAgg.point_distribution = point_distribution

    estimators = [
        GmmHmmMSSG(n_states=10, n_samples_scale=3, n_topics=2, name="GmmHmmMSSG(n_states=10)",
            bg_kwargs=dict(
                n_dim=2, n_components=1, max_iter=100,
                verbose=gmm_hmm_verbose, careful=True, cov_type='full',
                left_to_right=True, reuse=False, n_restarts=5, max_attempts=1),
            lda_settings=dict(em_max_iter=30, em_tol=0.01),
            verbose=lda_verbose),
        GmmHmm1x1(bg_kwargs=dict(
                      n_components=1, n_dim=2, cov_type='full', careful=True,
                      max_iter=30, verbose=gmm_hmm_verbose, n_restarts=5)),
        GmmHmmAgg(bg_kwargs=dict(
                      n_components=1, n_dim=2, cov_type='full', careful=True,
                      max_iter=30, verbose=gmm_hmm_verbose, n_restarts=5))]

    random_state = np.random.RandomState()

    data_kwargs = dict(
        use_digits=use_digits,
        max_tasks=10,
        n_transfer_tasks=5,
        core_train_wpt=args.core_train_wpt,
        transfer_train_wpt=10,
        transfer_test_wpt=100,
        alpha=args.alpha,
        horizon=args.horizon,
        sample_every=3,
        simplify=5)
    data_generator = generate_pendigit_data_sparse
    learn_halt = False

    _log_likelihood_score = partial(
        log_likelihood_score, string=learn_halt)
    _log_likelihood_score.__name__ = "log_likelihood"

    exp_kwargs = dict(
        mode='data', base_estimators=estimators,
        generate_data=data_generator,
        data_kwargs=data_kwargs,
        search_kwargs=dict(n_iter=10),
        directory='/data/seq_lda/',
        score=[RMSE_score, _log_likelihood_score],  # , one_norm_score],
        x_var_name='n_train_tasks',
        x_var_values=range(1, 10),
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], n_repeats=5, search_kwargs=dict(n_iter=5))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display,
        title=title,
        legend_loc='right')
