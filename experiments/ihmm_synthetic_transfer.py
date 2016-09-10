from __future__ import print_function
import numpy as np
import seaborn
from functools import partial
import argparse

from spectral_dagger.utils import run_experiment_and_plot

from seq_lda.algorithms import (
    Spectral1x1, SpectralAgg, EmPfaLDA, ExpMax1x1, ExpMaxAgg)
from seq_lda import (
    word_correct_rate, log_likelihood_score, one_norm_score)

from ihmm_synthetic import generate_ihmm_synthetic_data


seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

random_state = np.random.RandomState(4)

if __name__ == "__main__":
    hmm_verbose = False
    lda_verbose = False

    estimators = [
        EmPfaLDA(
            n_samples=1000,
            em_kwargs=dict(
                pct_valid=0.0, alg='bw', verbose=hmm_verbose,
                hmm=False, treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5),
            verbose=lda_verbose, name="bw,n_samples=1000"),
        ExpMax1x1(
            name="ExpMax1x1",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5)),
        ExpMaxAgg(
            name="ExpMaxAgg,with_transfer=True",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5), add_transfer_data=True),
        ExpMaxAgg(
            name="ExpMaxAgg,with_transfer=False",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5), add_transfer_data=False)]

    random_state = np.random.RandomState(100)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--core-train-wpt", type=int, default=15)
    args, _ = parser.parse_known_args()

    horizon = 0
    data_kwargs = dict(
        core_train_wpt=args.core_train_wpt,
        n_transfer_tasks=5,
        transfer_train_wpt=10,
        transfer_test_wpt=100,
        noise=0.05,
        alpha=args.alpha,
        horizon=horizon)

    data_generator = generate_ihmm_synthetic_data

    _log_likelihood_score = partial(
        log_likelihood_score, string=(horizon == np.inf or horizon == 0))
    _log_likelihood_score.__name__ = "log_likelihood"

    exp_kwargs = dict(
        mode='data', base_estimators=estimators,
        generate_data=data_generator,
        data_kwargs=data_kwargs,
        search_kwargs=dict(n_iter=10),
        directory='/data/seq_lda/',
        score=[word_correct_rate, _log_likelihood_score, one_norm_score],
        x_var_name='n_train_tasks',
        x_var_values=range(1, 21),
        n_repeats=10)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[5, 6], n_repeats=2, search_kwargs=dict(n_iter=2))

    score_display = [
        'Correct Prediction Rate',
        'Log Likelihood',
        'Negative One Norm']
    x_var_display = '\# Training Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)
