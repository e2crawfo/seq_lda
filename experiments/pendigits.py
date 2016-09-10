
from __future__ import print_function
import numpy as np
import seaborn

from functools import partial

from sklearn.utils import check_random_state

from spectral_dagger.datasets import uni_dep
from spectral_dagger.utils import run_experiment_and_plot

from seq_lda.algorithms import (
    MarkovLDA, Markov1x1, MarkovAgg,
    Spectral1x1, SpectralAgg, EmPfaLDA, ExpMax1x1, ExpMaxAgg)
from seq_lda import (
    generate_multitask_sequence_data,
    word_correct_rate, log_likelihood_score, one_norm_score)


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
        length=np.inf,
        permute=False,
        difference=True,
        sample_every=1,
        random_state=None):

    rng = check_random_state(random_state)

    n_tasks = n_train_tasks + n_transfer_tasks + n_test_tasks

    if n_tasks > len(max_tasks):
        raise ValueError(
            "Requested number of tasks (%d) is greater "
            "than the maximum number of tasks "
            "allowed (%d)." % (n_tasks, max_tasks))

    n_words_per_task = max(
        core_train_wpt + core_test_wpt,
        transfer_train_wpt + transfer_test_wpt,
        test_wpt)

    data = get_data(difference, sample_every)

    if permute:
        data = random_state.permutation(data)
    if len(data) > max_tasks:
        data = data[:max_tasks]

    all_sequences = []
    for sequences in data:
        if length != np.inf:
            sequences = [
                seq[:length] for seq in sequences
                if len(seq) >= length]

        if len(sequences) >= n_words_per_task:
            sequences = random_state.permutation(sequences)[:n_words_per_task]
            all_sequences.append(sequences)

    if not len(all_sequences) == n_tasks:
        raise Exception(
            "Not enough tasks have at least %d sequences of length "
            "at least %d." % (n_words_per_task, length))

    train_data, test_data = generate_multitask_sequence_data(
        all_sequences, n_train_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=rng,
        context=dict(learn_halt=True))

    return (train_data, test_data,
            dict(max_topics=10,
                 max_states=30,
                 max_states_per_topic=10))

seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

if __name__ == "__main__":
    hmm_verbose = False
    lda_verbose = False

    estimators = [
        Spectral1x1(), SpectralAgg(),
        MarkovLDA(), Markov1x1(), MarkovAgg(),
        EmPfaLDA(
            n_samples=1000,
            em_kwargs=dict(
                pct_valid=0.0, alg='bw', verbose=hmm_verbose,
                hmm=False, treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5),
            verbose=lda_verbose, name="bw,n_samples=1000,max_iters=10"),
        ExpMax1x1(
            name="ExpMax1x1,max_iters=10",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5)),
        ExpMaxAgg(
            name="ExpMaxAgg,max_iters=10",
            em_kwargs=dict(
                pct_valid=0.0, hmm=False,
                treba_args="--threads=4", n_restarts=1,
                max_iters=10, max_delta=0.5))]

    random_state = np.random.RandomState(10)

    max_languages = 30
    languages = uni_dep.languages()[:max_languages]
    languages = list(random_state.permutation(languages))

    data_kwargs = dict(
        languages=languages,
        core_train_wpt=20,
        core_test_wpt=100,
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
        x_var_name='n_train_tasks',
        x_var_values=range(1, 21),
        n_repeats=10)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[1, 2], n_repeats=2, search_kwargs=dict(n_iter=2))

    score_display = [
        'Correct Prediction Rate',
        'Log Likelihood',
        'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)