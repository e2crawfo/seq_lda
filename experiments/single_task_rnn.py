from __future__ import print_function
import numpy as np
from functools import partial

from spectral_dagger.utils import run_experiment_and_plot

from seq_lda.algorithms.baseline import Neural1x1
from seq_lda import RMSE_score, log_likelihood_score  # , one_norm_score)

from pendigits import generate_pendigit_data


if __name__ == "__main__":
    lstm_verbose = False
    lda_verbose = True
    use_digits = [0, 1, 2, 3, 4]

    def point_distribution(self, context):
        return dict()

    Neural1x1.point_distribution = point_distribution

    estimators = [
        Neural1x1(
            n_hidden=10,
            bg_kwargs=dict(
                max_epochs=100000, use_dropout=False, patience=1,
                validFreq=200, verbose=lstm_verbose))]

    random_state = np.random.RandomState(50)

    data_kwargs = dict(
        use_digits=use_digits,
        max_tasks=1,
        n_train_tasks=1,
        core_train_wpt=30,
        core_test_wpt=100,
        permute=True,
        alpha=0.05,
        sample_every=1)
    data_generator = generate_pendigit_data
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
        x_var_values=[1, 6, 11, 16, 21],
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[5, 10, 15, 20], n_repeats=5, search_kwargs=dict(n_iter=1))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display,
        title=title)