from __future__ import print_function
import numpy as np
import seaborn

from functools import partial

from sklearn.utils import check_random_state

from spectral_dagger.datasets import pendigits
from spectral_dagger.utils import run_experiment_and_plot

from seq_lda.algorithms import Lstm1x1, LstmAgg, LstmLDA
from seq_lda import (
    generate_multitask_sequence_data,
    RMSE_score, log_likelihood_score)  # , one_norm_score)


def generate_pendigit_data_sparse(
        core_train_wpt=0,
        core_test_wpt=0,
        difference=True,
        sample_every=1,
        use_digits=None,
        random_state=None):

    if use_digits is None:
        use_digits = range(10)
    n_topics = len(use_digits)

    rng = check_random_state(random_state)

    data, labels = pendigits.get_data(
        difference, sample_every,
        ignore_multisegment=False, use_digits=use_digits)
    data = [d for dd in data for d in dd]
    data = rng.permutation(data)

    assert len(data) >= core_train_wpt + core_test_wpt
    data = data[:core_train_wpt + core_test_wpt]

    # Makes the ndarrays hashable.
    for seq in data:
        seq.flags.writeable = False
    data = [data]

    to_hashable = lambda x: hash(x.data)

    train_data, test_data = generate_multitask_sequence_data(
        data, 1, 0, train_split=(core_train_wpt, core_test_wpt),
        random_state=rng,
        to_hashable=to_hashable,
        context=dict(learn_halt=True, n_input=2))

    return (train_data, test_data,
            dict(max_topics=n_topics,
                 max_states=30,
                 max_states_per_topic=10))


seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

if __name__ == "__main__":
    lstm_verbose = False
    lda_verbose = True
    use_digits = [0, 1, 2]

    def point_distribution(self, context):
        return dict()
    Lstm1x1.point_distribution = point_distribution

    estimators = [
        Lstm1x1(n_hidden=2, name="n_hidden=2",
                lstm_kwargs=dict(max_epochs=100000, use_dropout=False, patience=1, validFreq=200, verbose=True)),
        Lstm1x1(n_hidden=5, name="n_hidden=5",
                lstm_kwargs=dict(max_epochs=100000, use_dropout=False, patience=1, validFreq=200, verbose=True)),
        Lstm1x1(n_hidden=10, name="n_hidden=10",
                lstm_kwargs=dict(max_epochs=100000, use_dropout=False, patience=1, validFreq=200, verbose=True))]

    random_state = np.random.RandomState(50)

    data_kwargs = dict(
        use_digits=use_digits,
        core_train_wpt=30,
        core_test_wpt=100,
        sample_every=1)
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
        x_var_name='core_train_wpt',
        x_var_values=[50, 100, 150, 200],
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[5, 10], n_repeats=5, search_kwargs=dict(n_iter=5))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display,
        title=title)
