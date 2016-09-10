import numpy as np
import os
import shutil
import itertools

import pytest

from spectral_dagger.sequence import MarkovChain, AdjustedMarkovChain
from spectral_dagger.utils import rmse, normalize

from seq_lda import (
    Dictionary, fit_lda, fit_markov_lda,
    fit_callback_lda, lda_inference, SequentialLDA)
from seq_lda.algorithms import generate_markov_chains
from seq_lda.algorithms.spectral_lda import SimpleSpectralSAFromDist
from seq_lda.algorithms import LDA, SpectralLDA

np.set_printoptions(suppress=True)


def find_permutation(theta1, theta2):
    """ Find permutation of the columns of ``theta1`` which
        minimizes RMSE with ``theta2``.
    """
    n_cols = theta1.shape[1]
    errors = (
        (p, rmse(theta1[:, p], theta2))
        for p in itertools.permutations(range(n_cols)))
    best = min(errors, key=lambda x: x[1])
    return best


def run_lda_test(
        n_docs, n_words_per_doc, docs, gt_model,
        fit, delete, directory='test_results'):
    dictionary = Dictionary.from_docs(docs.core_data)
    data = dictionary.encode_as_bow(docs.core_data)

    n_word_types = len(data.words)

    if not os.path.isdir(directory):
        os.makedirs(directory)
    settings = os.path.join(directory, 'settings.txt')

    try:
        learned_model = fit(data, n_word_types, directory, settings)
        learned_topics = learned_model.log_topics(data.words)

        likelihood, learned_gamma, learned_phi = lda_inference(
            data, directory, gt_model.n_topics, n_word_types,
            learned_model.alpha_, learned_topics, settings)

        learned_theta = normalize(learned_gamma, ord=1, axis=1)

        gamma = [gt_model.gamma_[i] for i in gt_model.task_indices]
        gt_theta = normalize(gamma, ord=1, axis=1)

        perm, error = find_permutation(learned_theta, gt_theta)
        learned_topics = learned_topics[perm, :]
        learned_theta = learned_theta[:, perm]

        print "Theta" + "*" * 80
        print "Learned"
        for t in learned_theta:
            print t
        print "Ground Truth"
        for t in gt_theta:
            print t

        print "Topics" + "*" * 80
        print "Learned"
        for k in learned_topics:
            print k
        print "Ground Truth"
        print gt_model.generators_[0].get_string_prob([1])
        gt_topics = gt_model.log_topics(data.words)
        for k in gt_topics:
            print k

        assert error < 0.2

    finally:
        if delete:
            try:
                os.remove(settings)
            except:
                pass
            try:
                shutil.rmtree(directory)
            except:
                pass

    return learned_model


@pytest.mark.parametrize('learn_halt', [True, False])
def test_simple_markov(learn_halt, delete=True):
    rng = np.random.RandomState(1)

    length = np.inf if learn_halt else 5
    n_docs = 30
    n_words_per_doc = 200

    alpha = 1.0

    if learn_halt:
        mcs = [
            AdjustedMarkovChain(
                [0.0, 1.0], [[0.0, 1.0], [0.0, 1.0]],
                [0.3, 0.5]),
            AdjustedMarkovChain(
                [1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]],
                [0.1, 0.6])]
    else:
        mcs = [
            MarkovChain([0.0, 1.0], [[0.0, 1.0], [0.0, 1.0]]),
            MarkovChain([1.0, 0.0], [[1.0, 0.0], [1.0, 0.0]])]

    markov_lda = SequentialLDA(alpha, mcs)
    docs, _, _ = markov_lda.generate_data(
        length=length, n_train_docs=n_docs,
        n_train_wpd=n_words_per_doc, seed=rng)

    n_topics = len(mcs)
    n_symbols = mcs[0].n_observations

    def fit(data, n_word_types, directory, settings):
        alpha, markov_chains, gamma = fit_markov_lda(
            data, directory, n_topics, n_word_types,
            n_symbols, learn_halt, settings={'filename': settings})
        return SequentialLDA(alpha, markov_chains, gamma)

    learned_model = run_lda_test(
        n_docs, n_words_per_doc, docs, markov_lda, fit, delete)

    print "Markov Chains" + "*" * 80
    print "Learned"
    for k in learned_model.generators_:
        print k
    print "Ground Truth"
    for k in markov_lda.generators_:
        print k


@pytest.mark.parametrize('learn_halt', [True, False])
def test_markov(learn_halt, delete=True):
    rng = np.random.RandomState(1)

    alpha = 1.0
    beta = 1.0
    length = np.inf if learn_halt else 5
    halts = 0.5 if learn_halt else 0
    n_symbols = 3
    n_docs = 100
    n_topics = 3
    n_words_per_doc = 200

    generators = generate_markov_chains(
        beta, n_symbols, n_topics, halts=halts, rng=rng)
    markov_lda = SequentialLDA(alpha, generators)
    docs, _, _ = markov_lda.generate_data(
        length=length, n_train_docs=n_docs,
        n_train_wpd=n_words_per_doc, seed=rng)

    def fit(data, n_word_types, directory, settings):
        alpha, markov_chains, gamma = fit_markov_lda(
            data, directory, n_topics, n_word_types,
            n_symbols, learn_halt, settings={'filename': settings})
        return SequentialLDA(alpha, markov_chains, gamma)

    learned_model = run_lda_test(
        n_docs, n_words_per_doc, docs, markov_lda, fit, delete)

    print "Markov Chains" + "*" * 80
    print "Learned"
    for k in learned_model.generators_:
        print k
    print "Ground Truth"
    for k in markov_lda.generators_:
        print k


@pytest.mark.parametrize('learn_halt', [True, False])
def test_callback_markov(learn_halt, delete=True):
    rng = np.random.RandomState(1)

    alpha = 1.0
    beta = 1.0
    length = np.inf if learn_halt else 5
    halts = 0.5 if learn_halt else 0
    n_symbols = 3
    n_docs = 10
    n_topics = 3
    n_words_per_doc = 200

    generators = generate_markov_chains(
        beta, n_symbols, n_topics, halts=halts, rng=rng)
    markov_lda = SequentialLDA(alpha, generators)
    docs, _, _ = markov_lda.generate_data(
        length=length, n_train_docs=n_docs,
        n_train_wpd=n_words_per_doc, seed=rng)

    dictionary = Dictionary.from_docs(docs.core_data)

    def fit(data, n_word_types, directory, settings):
        _mcs = [0]

        def callback(class_word):
            mcs = []
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)
            for k, dist in enumerate(class_word):
                mc = AdjustedMarkovChain.from_distribution(
                    dist, dictionary.words, learn_halt, n_symbols)
                mcs.append(mc)
                for w, word in enumerate(dictionary.words):
                    log_prob_w[k][w] = (
                        mc.get_prefix_prob(word, log=True)
                        if not learn_halt
                        else mc.get_string_prob(word, log=True))
            _mcs[0] = mcs
            return log_prob_w

        alpha, log_topics, gamma = fit_callback_lda(
            data, directory, n_topics, n_word_types, callback,
            settings={'filename': settings})
        return SequentialLDA(alpha, _mcs[0], gamma)

    run_lda_test(
        n_docs, n_words_per_doc, docs, markov_lda, fit, delete)


@pytest.mark.parametrize('learn_halt', [True, False])
def _test_callback_spectral(learn_halt, delete=True):
    rng = np.random.RandomState(1)

    pautomac_args = dict(
        kind='hmm', n_states=3, n_symbols=5, symbol_density=0.5,
        transition_density=0.5, alpha=1.0, halts=learn_halt, rng=rng)
    alpha = 0.1
    length = np.inf if learn_halt else 5
    n_docs = 10
    n_topics = 3
    n_words_per_doc = 100

    generators = [
        make_pautomac_like(**pautomac_args) for i in range(n_topics)]
    spectral_lda = SpectralLDA(alpha, generators)
    docs, _, _ = spectral_lda.generate_data(
        length=length, n_train_docs=n_docs,
        n_train_wpd=n_words_per_doc, seed=rng)
    dictionary = Dictionary.from_docs(docs.core_data)

    learner_args = dict(
        max_components=4*pautomac_args['n_states'],
        lmbda=0.0, n_observations=pautomac_args['n_symbols'],
        max_basis_size=100, estimator='substring')

    def fit(data, n_word_types, directory, settings):
        _stoch_autos = [0]

        def callback(class_word):
            global stoch_autos
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            stoch_autos = []
            for k, dist in enumerate(class_word):
                sa = SimpleSpectralSAFromDist(**learner_args)
                sa.fit(dist, dictionary.words, learn_halt)
                stoch_autos.append(sa)

                for w, word in enumerate(dictionary.words):
                    log_prob_w[k][w] = (
                        sa.get_prefix_prob(word, log=True)
                        if not learn_halt
                        else sa.get_string_prob(word, log=True))
                print("n_components for topic %d: %d" % (k, sa.n_components_))

            print("Error induced by dist projection: ")
            print(np.abs(class_word - np.exp(log_prob_w)).sum(1)/class_word.shape[1])

            # Normalizing the distribution...
            # Not sure if we ultimately want to do this or not.
            #_sum = np.exp(log_prob_w).sum(1)
            #log_prob_w = log_prob_w - np.log(_sum).reshape(-1, 1)
            _stoch_autos[0] = stoch_autos
            return log_prob_w

        alpha, log_prob_w, gamma = fit_callback_lda(
            data, directory, n_topics, n_word_types, callback,
            settings={'filename': settings})
        return SequentialLDA(alpha, stoch_autos[0], gamma)

    run_lda_test(
        n_docs, n_words_per_doc, docs, markov_lda, fit, delete)


def test_lda(delete=True):
    rng = np.random.RandomState(1)

    alpha = 1.0
    beta = 1.0
    n_word_types = 20
    n_docs = 20
    n_topics = 3
    n_words_per_doc = 1000

    topics = rng.dirichlet(beta * np.ones(n_word_types), n_topics)
    lda = LDA(alpha, np.log(topics))
    docs, _, _ = lda.generate_data(
        n_train_docs=n_docs, n_test_docs=0,
        n_words_per_doc=n_words_per_doc)

    def fit(data, n_word_types, directory, settings):
        alpha, log_topics, gamma = fit_lda(
            data, directory, n_topics, n_word_types,
            settings={'filename': settings})
        return LDA(alpha, log_topics, gamma)

    run_lda_test(n_docs, n_words_per_doc, docs, lda, fit, delete)


if __name__ == "__main__":
    # test_simple_markov(False, False)
    # test_markov(False, False)
    # test_basic(delete=False)
    # test_callback_markov(True, delete=False)
    # test_callback_markov(False, delete=False)
    _test_callback_spectral(False, delete=False)
    pass
