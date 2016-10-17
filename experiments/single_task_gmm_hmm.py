from __future__ import print_function
import numpy as np
from functools import partial
import argparse
from sklearn.utils import check_random_state

from spectral_dagger.utils import run_experiment_and_plot, normalize
from spectral_dagger.sequence import GmmHmm

from seq_lda.algorithms.baseline import GmmHmm1x1
from seq_lda import (
    GenericMultitaskPredictor, generate_multitask_sequence_data,
    RMSE_score, log_likelihood_score)

from pendigits import generate_pendigit_data_single_task


def generate_cts_ihmm_synthetic_data_single_task(
        n_train=0,
        n_test=0,
        n_states=10,
        n_dim=8,
        T_density=1.0,
        mu_noise=1.0,
        sigma_noise=1.0,
        horizon=10,
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

    n_components = 1

    pi = random_state.dirichlet(T_density * np.ones(n_states))
    T = random_state.dirichlet(T_density * np.ones(n_states), size=n_states)

    mu = random_state.normal(0, 1, size=(n_dim, n_states))
    mu = mu_noise * normalize(mu, axis=0, ord=2)
    mu = mu[:, :, None]

    sigma = np.tile(
        sigma_noise * np.eye(n_dim)[:, :, None, None],
        (1, 1, n_states, n_components))

    M = np.ones((n_states, n_components))

    hmm = GmmHmm(parameters=dict(pi=pi, T=T, mu=mu, sigma=sigma, M=M))
    sequences = hmm.sample_episodes(
        n_train + n_test, horizon=horizon, random_state=random_state)

    sequences = [np.array(seq) for seq in sequences]
    for seq in sequences:
        seq.flags.writeable = False

    to_hashable = lambda x: hash(x.data)

    ground_truth = GenericMultitaskPredictor([hmm], name="Ground Truth")

    train_data, test_data = generate_multitask_sequence_data(
        [sequences], 1, 0,
        train_split=(n_train, n_test),
        random_state=random_state,
        context=dict(learn_halt=False),
        to_hashable=to_hashable)

    return (train_data, test_data, dict(max_states=20, true_model=ground_truth))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=6)
    args, _ = parser.parse_known_args()

    hmm_verbose = False
    lda_verbose = True

    def point_distribution(self, context):
        return dict()

    GmmHmm1x1.point_distribution = point_distribution

    random_state = np.random.RandomState()
    _log_likelihood_score = partial(
        log_likelihood_score, string=False)
    _log_likelihood_score.__name__ = "log_likelihood"

    data_kwargs = dict(
        max_tasks=20,
        n_train_words=200,
        n_test_words=200,
        per_digit=False,
        permute_tasks=False,
        sample_every=3,
        simplify=5,
        horizon=5)
    data_generator = generate_pendigit_data_single_task
    n_states = [5, 10, 15]#, 20, 25]
    estimators = [
        GmmHmm1x1(
            n_states=n, name="n_states=%d" % n,
            bg_kwargs=dict(
                n_dim=2, n_components=1, n_restarts=5,
                max_iter=2, thresh=1e-4, verbose=0,
                cov_type='full', careful=True))
        for n in n_states]
    exp_kwargs = dict(
        mode='data', base_estimators=estimators,
        generate_data=data_generator,
        data_kwargs=data_kwargs,
        search_kwargs=dict(n_iter=10),
        directory='/data/seq_lda/',
        score=[RMSE_score, _log_likelihood_score],  # , one_norm_score],
        x_var_name='use_digits',
        x_var_values=range(1, 11),
        n_repeats=20)
    # data_kwargs = dict(
    #     n_states=10,
    #     n_dim=5,
    #     n_test=1000,
    #     mu_noise=10.0,
    #     sigma_noise=1.0,
    #     T_density=0.5,
    #     horizon=args.horizon)
    # data_generator = generate_cts_ihmm_synthetic_data_single_task
    # estimators = [
    #     GmmHmm1x1(
    #         n_states=data_kwargs['n_states'],
    #         bg_kwargs=dict(
    #             n_dim=data_kwargs['n_dim'], n_components=1,
    #             max_iter=10, thresh=1e-4, verbose=0, cov_type='full',
    #             careful=False))]
    # exp_kwargs = dict(
    #     mode='data', base_estimators=estimators,
    #     generate_data=data_generator,
    #     data_kwargs=data_kwargs,
    #     search_kwargs=dict(n_iter=10),
    #     directory='/data/seq_lda/',
    #     score=[RMSE_score, _log_likelihood_score],  # , one_norm_score],
    #     x_var_name='n_train',
    #     x_var_values=[4, 8, 16, 32, 64, 128, 256, 512],
    #     n_repeats=5)

    quick_exp_kwargs = exp_kwargs.copy()
    quick_exp_kwargs.update(
        x_var_values=[2, 3], n_repeats=1, search_kwargs=dict(n_iter=10))

    score_display = ['RMSE', 'Log Likelihood']  # , 'Negative One Norm']
    x_var_display = '\# Tasks'
    title = 'Performance on Test Set'

    run_experiment_and_plot(
        exp_kwargs, quick_exp_kwargs,
        random_state=random_state,
        x_var_display=x_var_display,
        score_display=score_display, title=title)