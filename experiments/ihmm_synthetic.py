from __future__ import print_function
import numpy as np
import seaborn
import argparse

from functools import partial

from sklearn.utils import check_random_state

from spectral_dagger.utils import run_experiment_and_plot, normalize
from spectral_dagger.sequence import HMM, MixtureStochAuto

from seq_lda.algorithms import EmPfaLDA, ExpMax1x1, ExpMaxAgg
from seq_lda import (
    generate_multitask_sequence_data, GenericMultitaskPredictor,
    word_correct_rate, log_likelihood_score, one_norm_score)


seaborn.set(style="white")
seaborn.set_context(rc={'lines.markeredgewidth': 0.1})

random_state = np.random.RandomState(4)


def generate_ihmm_synthetic_data(
        n_train_tasks=0,
        n_transfer_tasks=0,
        n_test_tasks=0,
        core_train_wpt=0,
        core_test_wpt=0,
        transfer_train_wpt=0,
        transfer_test_wpt=0,
        test_wpt=0,
        task_idx=None,
        noise=0.05,
        horizon=0,
        alpha=0.01,
        random_state=None):
    """ Generate data from the synthetic multitask HMM example.

    ``n_tasks`` is either an integer, in which case each of the base
        tasks will be replicated that number of times, or a list of length
        three, where the value at index i gives the number of times to
        replicate task i.

    """
    random_state = check_random_state(random_state)
    halt = horizon == 0 or horizon == np.inf
    horizon = np.inf if halt else horizon

    pi = [0] * 3
    A = [0] * 3
    B = [0] * 3

    pi[0] = np.array([.695, .3050])
    A[0] = np.array([[.8, .2], [.2, .8]])
    B[0] = np.array(
        [[.05, .1, .7, .1, .05, 0, 0, 0],
         [0, 0, 0, .05, .1, .7, .1, .05]])

    pi[1] = np.array([.8724, .1276])
    A[1] = np.array([[.2, .8], [.8, .2]])
    B[1] = np.array(
        [[0, 0, .05, .1, .7, .1, .05, 0],
         [0, .05, .1, .7, .1, .05, 0, .0]])

    pi[2] = np.array([.4729, .5271])
    A[2] = np.array([[0.5, 0.5], [0.5, 0.5]])
    B[2] = np.array(
        [[.05, 0, 0, 0, .05, .1, .7, .1],
         [.1, .05, 0, 0, 0, .05, .1, .7]])

    stop_prob = np.array([0.25, 0.25]) if halt else None

    try:
        alpha = float(alpha)
        alpha = alpha * np.ones(3)
    except:
        alpha = np.array(alpha)
    assert alpha.ndim == 1
    assert alpha.shape[0] == 3
    assert (alpha >= 0.0).all()

    assert noise > 0
    state_noise_mask = np.array([[1, 1], [1, 1]])
    obs_noise_mask = np.array(
        [[1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1]])

    n_tasks = n_train_tasks + n_transfer_tasks + n_test_tasks
    task_coefficients = random_state.dirichlet(alpha, n_tasks)

    n_symbols = 8

    n_words_per_task = max(
        core_train_wpt + core_test_wpt,
        transfer_train_wpt + transfer_test_wpt,
        test_wpt)
    all_sequences = []
    generators = []

    for tc in task_coefficients:
        hmms_for_task = []
        for idx in range(3):
            _A = A[idx].copy()
            _A[state_noise_mask > 0] += (
                random_state.uniform(0.0, noise, size=_A.shape)[state_noise_mask > 0])
            _A = normalize(_A, ord=1, axis=1)

            _B = B[idx].copy()
            _B[obs_noise_mask > 0] += (
                random_state.uniform(0.0, noise, size=_B.shape)[obs_noise_mask > 0])
            _B = normalize(_B, ord=1, axis=1)

            hmm = HMM(pi[idx], _A, _B, stop_prob)
            hmms_for_task.append(hmm)

        generator = MixtureStochAuto(tc, hmms_for_task)
        generators.append(generator)

        sequences = generator.sample_episodes(
            n_words_per_task, horizon=horizon, random_state=random_state)

        all_sequences.append(sequences)

    ground_truth = GenericMultitaskPredictor(generators, name="Ground Truth")

    train_data, test_data = generate_multitask_sequence_data(
        all_sequences, n_train_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=random_state,
        context=dict(learn_halt=True, n_symbols=n_symbols),
        to_hashable=lambda x: tuple(x))

    return (train_data, test_data,
            dict(n_symbols=n_symbols, max_topics=6,
                 max_states=20, max_states_per_topic=5,
                 true_model=ground_truth))


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

    random_state = np.random.RandomState(101)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--core-train-wpt", type=int, default=15)
    args, _ = parser.parse_known_args()

    horizon = 0
    data_kwargs = dict(
        core_train_wpt=args.core_train_wpt,
        core_test_wpt=100,
        alpha=args.alpha,
        noise=0.05,
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
    # x_var_display = '\# Training Samples per Task'
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)
