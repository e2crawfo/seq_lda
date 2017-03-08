from __future__ import print_function
import numpy as np

from spectral_dagger.sequence import AdjustedMarkovChain
from spectral_dagger import Estimator

from seq_lda import fit_markov_lda, lda_inference, SequentialLDA


def generate_markov_chains(
        alpha, n_symbols, n_chains, halts, rng=None):
    try:
        len(alpha)
    except TypeError:
        alpha = alpha * np.ones(n_symbols)

    halts = float(halts)

    mcs = []
    for i in range(n_chains):
        init_dist = rng.dirichlet(alpha)

        T = np.zeros((n_symbols, n_symbols))
        stop = np.zeros(n_symbols)
        for k in range(n_symbols):
            T[k, :] = rng.dirichlet(alpha)
            if halts > 0:
                stop[k] = rng.beta(halts, 1)

        mcs.append(AdjustedMarkovChain(init_dist, T, stop))
    return mcs


class MarkovLDA(SequentialLDA, Estimator):
    record_attrs = ['n_topics']

    def __init__(self, n_topics=1, directory="results/lda_markov/",
                 name="MarkovLDA"):
        self._init(locals())

    def point_distribution(self, context):
        return dict(n_topics=np.arange(2, context['max_topics']))

    def fit(self, X, y=None):
        """ X is instance of ``MultitaskSequenceDataset``
            with a ``dictionary`` attr.  """
        self.record_indices(X.indices)

        corpus = X.core_data
        training_words = corpus.get_all_words()
        n_word_types = len(training_words)

        self.alpha_, self.generators_, self.gamma_ = fit_markov_lda(
            corpus, self.directory, self.n_topics, n_word_types,
            X.n_symbols, X.learn_halt, log_name="markov.log")

        if X.n_transfer > 0:
            transfer_corpus = X.transfer_data
            transfer_words = transfer_corpus.get_all_words()

            log_topics = self.log_topics(
                transfer_words, prefix=not X.learn_halt)
            n_transfer_word_types = len(transfer_words)

            lhood, transfer_gamma, phi = lda_inference(
                transfer_corpus, self.directory, self.n_topics,
                n_transfer_word_types, self.alpha_, log_topics)

            self.gamma_ = np.concatenate((self.gamma_, transfer_gamma))

        return self
