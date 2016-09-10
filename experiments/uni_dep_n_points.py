from __future__ import print_function
import numpy as np
import seaborn

from functools import partial

from spectral_dagger.datasets import uni_dep
from spectral_dagger.utils import run_experiment_and_plot

from seq_lda.algorithms import (
    MarkovLDA, Markov1x1, MarkovAgg,
    Spectral1x1, SpectralAgg, EmPfaLDA, ExpMax1x1, ExpMaxAgg)
from seq_lda import (
    word_correct_rate, log_likelihood_score, one_norm_score)

from uni_dep import generate_unidep_data


seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

if __name__ == "__main__":
    hmm_verbose = False
    lda_verbose = False

    estimators = [
        Spectral1x1(), SpectralAgg(),
        EmPfaLDA(
            n_samples=1000,
            em_kwargs=dict(
                pct_valid=0.0, alg='bw', verbose=hmm_verbose,
                hmm=False, treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5),
            verbose=lda_verbose, name="bw,n_samples=1000,max_iters=10"),
        EmPfaLDA(
            n_samples=1000,
            em_kwargs=dict(
                pct_valid=0.0, alg='bw', verbose=hmm_verbose,
                hmm=False, treba_args="--threads=4", n_restarts=1,
                max_iters=1000, max_delta=0.5),
            verbose=lda_verbose, name="bw,n_samples=1000,max_iters=1000"),
        ExpMax1x1(
            name="ExpMax1x1",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=1000, max_delta=0.5)),
        ExpMax1x1(
            name="ExpMax1x1,max_iters=10",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5)),
        ExpMaxAgg(
            name="ExpMaxAgg",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=1000, max_delta=0.5)),
        ExpMaxAgg(
            name="ExpMaxAgg,max_iters=10",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5))]

    random_state = np.random.RandomState(4)

    max_languages = 10
    languages = uni_dep.languages()[:max_languages]
    languages = list(random_state.permutation(languages))

    data_kwargs = dict(
        languages=languages,
        n_train_tasks=10,
        core_test_wpt=1000,
        permute_languages=True,
        length=np.inf)
    data_generator = generate_unidep_data
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
        score=[word_correct_rate, _log_likelihood_score, one_norm_score],
        x_var_name='core_train_wpt',
        x_var_values=[5, 10, 15, 20, 25, 30, 35, 40, 45],
        n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[5, 6], n_repeats=2, search_kwargs=dict(n_iter=2))

    score_display = [
        'Correct Prediction Rate',
        'Log Likelihood',
        'Negative One Norm']
    x_var_display = '\# Samples per Task'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)
