from __future__ import print_function
import numpy as np
import seaborn
from collections import defaultdict
import argparse
from functools import partial
from sklearn.utils import check_random_state

from spectral_dagger.datasets import pendigits
from spectral_dagger.utils import run_experiment_and_plot, sample_multinomial
from spectral_dagger.sequence import GenerativeRNN, GenerativeGRU, GenerativeLSTM

from seq_lda.algorithms import Neural1x1, NeuralAgg, NeuralMSSG
from seq_lda import (
    generate_multitask_sequence_data,
    RMSE_score, log_likelihood_score)  # , one_norm_score)


def generate_pendigit_data_single_task(
        max_tasks=44,
        n_train_words=0,
        n_test_words=0,
        per_digit=True,
        permute=True,
        difference=True,
        sample_every=1,
        use_digits=None,
        random_state=None):
    rng = check_random_state(random_state)

    try:
        use_digits = int(use_digits)
        assert use_digits < 11, "Requesting too many digit types: %d." % use_digits
        use_digits = rng.choice(10, use_digits, replace=False)
        print("Using digits: ", use_digits)
    except (ValueError, TypeError):
        pass

    if use_digits is None:
        use_digits = range(10)

    if per_digit:
        data, labels = pendigits.get_data(difference, sample_every, use_digits=use_digits)
        assert len(data) >= max_tasks
        if permute:
            p = rng.permutation(range(len(data)))
            data = [data[i] for i in p]
            labels = [labels[i] for i in p]

        data = data[:max_tasks]
        labels = labels[:max_tasks]

        data = [d for dd in data for d in dd]
        labels = [l for ll in labels for l in ll]
        p = rng.permutation(range(len(data)))
        data = [data[i] for i in p]
        labels = [labels[i] for i in p]

        final_data = []
        for digit in use_digits:
            n_examples = 0
            for d, l in zip(data, labels):
                if l == digit:
                    final_data.append(d)
                    n_examples += 1
                    if n_examples >= n_train_words + n_test_words:
                        break
        data = rng.permutation(final_data)
        n_train_words = len(use_digits) * n_train_words
        n_test_words = len(use_digits) * n_test_words
    else:
        data, labels = pendigits.get_data(difference, sample_every, use_digits=use_digits)
        assert len(data) >= max_tasks
        if permute:
            data = rng.permutation(data)
        data = data[:max_tasks]

        data = [d for dd in data for d in dd]
        data = rng.permutation(data)

        n_words_per_task = n_train_words + n_test_words
        assert len(data) >= n_words_per_task
        data = data[:n_words_per_task]

    if difference:
        size = 0.0
        n = 0
        for seq in data:
            for s in seq:
                size += np.linalg.norm(s, ord=2)
                n += 1
        print("Average distance: %f" % (size/n))

    # Makes the ndarrays hashable.
    for seq in data:
        seq.flags.writeable = False

    to_hashable = lambda x: hash(x.data)

    train_data, test_data = generate_multitask_sequence_data(
        [data], 1, 0,
        train_split=(n_train_words, n_test_words), random_state=rng,
        to_hashable=to_hashable,
        context=dict(learn_halt=True))

    return (train_data, test_data,
            dict(max_topics=10,
                 max_states=30,
                 max_states_per_topic=10))


def generate_pendigit_data(
        max_tasks=44,
        n_train_tasks=33,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=0,
        core_test_wpt=0,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        permute=False,
        difference=True,
        sample_every=1,
        use_digits=None,
        random_state=None):
    rng = check_random_state(random_state)

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

    data, _ = pendigits.get_data(difference, sample_every, use_digits=use_digits)

    if permute:
        data = rng.permutation(data)

    all_sequences = []
    for i in range(len(data)):
        sequences = data[i]

        if len(sequences) >= n_words_per_task:
            sequences = rng.permutation(sequences)[:n_words_per_task]
            all_sequences.append(sequences)

            # Makes the ndarrays hashable.
            for seq in sequences:
                seq.flags.writeable = False

        if len(all_sequences) == n_tasks:
            break

    if len(all_sequences) != n_tasks:
        raise Exception(
            "Not enough tasks have at least %s "
            "sequences." % (n_words_per_task))

    to_hashable = lambda x: hash(x.data)

    train_data, test_data = generate_multitask_sequence_data(
        all_sequences, n_train_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=rng,
        to_hashable=to_hashable,
        context=dict(learn_halt=True))

    return (train_data, test_data,
            dict(max_topics=10,
                 max_states=30,
                 max_states_per_topic=10))


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
        permute=False,
        difference=True,
        sample_every=1,
        use_digits=None,
        random_state=None):

    if use_digits is None:
        use_digits = range(10)
    use_digits = sorted(use_digits)
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
        difference, sample_every,
        ignore_multisegment=False, use_digits=use_digits)
    assert len(data) >= n_tasks

    if permute:
        indices = rng.permutation(range(len(data)))
        data = [data[i] for i in indices]
        labels = [labels[i] for i in indices]

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


seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

if __name__ == "__main__":
    neural_verbose = False
    lda_verbose = True
    use_digits = [1, 3]# [0, 1, 2, 3, 4]

    def point_distribution(self, context):
        return dict()

    Neural1x1.point_distribution = point_distribution
    NeuralAgg.point_distribution = point_distribution
    NeuralMSSG.point_distribution = point_distribution

    parser = argparse.ArgumentParser()
    parser.add_argument("--bg", type=str, default='rnn', choices=['rnn', 'gru', 'lstm'])
    args, _ = parser.parse_known_args()
    bg_class = dict(rnn=GenerativeRNN, gru=GenerativeGRU, lstm=GenerativeLSTM)[args.bg]

    estimators = [
        Neural1x1(
            bg_class,
            n_hidden=2,
            bg_kwargs=dict(
                max_epochs=100000, use_dropout=False, patience=10,
                validFreq=200, verbose=neural_verbose)),
        NeuralAgg(
            bg_class,
            n_hidden=2,
            bg_kwargs=dict(
                max_epochs=100000, use_dropout=False, patience=10,
                validFreq=200, verbose=neural_verbose)),
        NeuralMSSG(
            bg_class, n_hidden=2, n_samples=100, n_topics=len(use_digits),
            bg_kwargs=dict(
                max_epochs=100000, use_dropout=False, reuse=True,
                patience=10, validFreq=200, verbose=neural_verbose),
            #lda_settings=dict(em_max_iter=30, em_tol=0.01),
            verbose=lda_verbose)]

    random_state = np.random.RandomState()

    data_kwargs = dict(
        use_digits=use_digits,
        max_tasks=30,
        core_train_wpt=50,
        core_test_wpt=50,
        permute=True,
        alpha=0.01,
        sample_every=3)
    data_generator = generate_pendigit_data_sparse
    learn_halt = True

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
        x_var_values=[1, 6, 11, 16, 21],
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[1, 3, 5], n_repeats=5, search_kwargs=dict(n_iter=5))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display,
        title=title)
