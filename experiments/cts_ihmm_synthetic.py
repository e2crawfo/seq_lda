from __future__ import print_function
import numpy as np
import argparse
from functools import partial
from sklearn.utils import check_random_state

from spectral_dagger.utils import run_experiment_and_plot, normalize
from spectral_dagger.sequence import GmmHmm, MixtureSeqGen

from seq_lda.algorithms import GmmHmmMSSG, GmmHmm1x1, GmmHmmAgg
from seq_lda import (
    generate_multitask_sequence_data, GenericMultitaskPredictor,
    RMSE_score, log_likelihood_score)

random_state = np.random.RandomState(2)


def generate_cts_ihmm_synthetic_data(
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
        horizon=10,
        alpha=0.01,
        random_state=None):
    """ Generate data from the synthetic multitask HMM example.

    ``n_tasks`` is either an integer, in which case each of the base
        tasks will be replicated that number of times, or a list of length
        three, where the value at index i gives the number of times to
        replicate task i.

    """
    random_state = check_random_state(random_state)
    if horizon == 0 or horizon == np.inf:
        raise Exception("Horizon (%d) must be finite and positive." % horizon)

    pi = [0] * 3
    T = [0] * 3
    mu = [0] * 3
    sigma = [0] * 3
    M = [0] * 3

    n_states = 2
    n_components = 1
    n_dim = 8

    pi[0] = np.array([.695, .3050])
    T[0] = np.array([[.8, .2], [.2, .8]])
    mu[0] = np.array(
        [[.05, .1, .7, .1, .05, 0, 0, 0],
         [0, 0, 0, .05, .1, .7, .1, .05]]).T
    mu[0] = mu[0][:, :, None]
    sigma[0] = np.tile(np.eye(n_dim)[:, :, None, None], (1, 1, n_states, n_components))
    M[0] = np.ones((n_states, n_components))

    pi[1] = np.array([.8724, .1276])
    T[1] = np.array([[.2, .8], [.8, .2]])
    mu[1] = np.array(
        [[0, 0, .05, .1, .7, .1, .05, 0],
         [0, .05, .1, .7, .1, .05, 0, .0]]).T
    mu[1] = mu[1][:, :, None]
    sigma[1] = np.tile(np.eye(n_dim)[:, :, None, None], (1, 1, n_states, n_components))
    M[1] = np.ones((n_states, n_components))

    pi[2] = np.array([.4729, .5271])
    T[2] = np.array([[0.5, 0.5], [0.5, 0.5]])
    mu[2] = np.array(
        [[.05, 0, 0, 0, .05, .1, .7, .1],
         [.1, .05, 0, 0, 0, .05, .1, .7]]).T
    mu[2] = mu[2][:, :, None]
    sigma[2] = np.tile(np.eye(n_dim)[:, :, None, None], (1, 1, n_states, n_components))
    M[2] = np.ones((n_states, n_components))

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
    mu_noise_mask = np.array(
        [[1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1]]).T
    mu_noise_mask = mu_noise_mask[:, :, None]

    n_tasks = n_train_tasks + n_transfer_tasks + n_test_tasks
    task_coefficients = random_state.dirichlet(alpha, n_tasks)

    n_words_per_task = max(
        core_train_wpt + core_test_wpt,
        transfer_train_wpt + transfer_test_wpt,
        test_wpt)
    all_sequences = []
    generators = []

    for tc in task_coefficients:
        hmms_for_task = []
        for idx in range(3):
            _T = T[idx].copy()
            _T[state_noise_mask > 0] += (
                random_state.uniform(0.0, noise, size=_T.shape)[state_noise_mask > 0])
            _T = normalize(_T, ord=1, axis=1)

            _mu = mu[idx].copy()
            _mu[mu_noise_mask > 0] += (
                random_state.uniform(0.0, noise, size=_mu.shape)[mu_noise_mask > 0])
            _mu = normalize(_mu, ord=1, axis=1)

            hmm = GmmHmm(parameters=dict(pi=pi[idx], T=_T, mu=_mu, sigma=sigma[idx], M=M[idx]))
            hmms_for_task.append(hmm)

        generator = MixtureSeqGen(tc, hmms_for_task)
        generators.append(generator)

        sequences = generator.sample_episodes(
            n_words_per_task, horizon=horizon, random_state=random_state)

        sequences = [np.array(seq) for seq in sequences]
        for seq in sequences:
            seq.flags.writeable = False

        all_sequences.append(sequences)

    to_hashable = lambda x: hash(x.data)
    ground_truth = GenericMultitaskPredictor(generators, name="Ground Truth")

    train_data, test_data = generate_multitask_sequence_data(
        all_sequences, n_train_tasks, n_transfer_tasks,
        train_split=(core_train_wpt, core_test_wpt),
        transfer_split=(transfer_train_wpt, transfer_test_wpt),
        test_wpt=test_wpt, random_state=random_state,
        context=dict(learn_halt=False),
        to_hashable=to_hashable)

    return (train_data, test_data,
            dict(max_topics=6,
                 max_states=20, max_states_per_topic=5,
                 true_model=ground_truth))


if __name__ == "__main__":
    hmm_verbose = False
    lda_verbose = True

    def point_distribution(self, context):
        return dict()

    GmmHmm1x1.point_distribution = point_distribution
    GmmHmmAgg.point_distribution = point_distribution
    GmmHmmMSSG.point_distribution = point_distribution

    estimators = [
        GmmHmm1x1(
            n_states=2,
            bg_kwargs=dict(
                n_dim=8, n_components=1,
                max_iter=10, thresh=1e-4, verbose=0, cov_type='spherical',
                careful=False)),
        GmmHmmAgg(
            n_states=2,
            bg_kwargs=dict(
                n_dim=8, n_components=1,
                max_iter=10, thresh=1e-4, verbose=0, cov_type='spherical',
                careful=False)),
        GmmHmmMSSG(
            n_states=2,
            n_samples=1000,
            n_topics=3,
            bg_kwargs=dict(
                n_dim=8, n_components=1,
                max_iter=10, thresh=1e-4, verbose=0, cov_type='spherical',
                careful=False),
            verbose=lda_verbose)]

    random_state = np.random.RandomState(101)

    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--core-train-wpt", type=int, default=10)
    args, _ = parser.parse_known_args()

    data_kwargs = dict(
        core_train_wpt=args.core_train_wpt,
        core_test_wpt=100,
        alpha=args.alpha,
        noise=0.05,
        horizon=6)

    data_generator = generate_cts_ihmm_synthetic_data

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
        x_var_values=range(1, 21),
        n_repeats=10)

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
