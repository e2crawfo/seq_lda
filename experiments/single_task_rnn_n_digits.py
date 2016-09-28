from __future__ import print_function
import numpy as np
from functools import partial
import argparse

from spectral_dagger.utils import run_experiment_and_plot
from spectral_dagger.sequence import GenerativeRNN, GenerativeGRU, GenerativeLSTM

from seq_lda.algorithms.baseline import Neural1x1
from seq_lda import RMSE_score, log_likelihood_score  # , one_norm_score)

from pendigits import generate_pendigit_data_single_task


if __name__ == "__main__":
    neural_verbose = True
    lda_verbose = True
    use_digits = range(10)

    def point_distribution(self, context):
        return dict()

    Neural1x1.point_distribution = point_distribution

    parser = argparse.ArgumentParser()
    parser.add_argument("--bg", type=str, default='rnn', choices=['rnn', 'gru', 'lstm'])
    args, _ = parser.parse_known_args()
    bg_class = dict(rnn=GenerativeRNN, gru=GenerativeGRU, lstm=GenerativeLSTM)[args.bg]

    n_hidden = [2, 12, 22, 32]
    estimators = [
        Neural1x1(
            bg_class,
            n_hidden=nh,
            bg_kwargs=dict(
                max_epochs=100000, use_dropout=False, patience=10, valid_pct=0.2,
                validFreq=200, verbose=neural_verbose, theano_optimizer='fast_compile'),
            name="%s(n_hidden=%s)" % (args.bg, nh))
        for nh in n_hidden]

    random_state = np.random.RandomState()

    data_kwargs = dict(
        max_tasks=44,
        n_train_words=512,
        n_test_words=512,
        per_digit=True,
        permute=True,
        sample_every=3)
    data_generator = generate_pendigit_data_single_task
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
        x_var_name='use_digits',
        x_var_values=[1, 3, 5, 7, 9],
        n_repeats=10,
        name="single_task_rnn_bg=%s_use_digits" % args.bg)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[1, 2], n_repeats=2, search_kwargs=dict(n_iter=1))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display,
        title=title)
