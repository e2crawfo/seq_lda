from __future__ import print_function
import numpy as np
from scipy.stats import uniform, expon

from sklearn.utils.extmath import randomized_svd

from spectral_dagger.utils import normalize, Estimator
from spectral_dagger.sequence import (
    SpectralSA, top_k_basis, fixed_length_basis,
    build_frequencies, hankels_from_callable)

from seq_lda import (
    SequentialLDA, fit_callback_lda, lda_inference)


class SimpleSpectralSAFromDist(SpectralSA):
    """
    If ``n_samples==0``, we aren't going to sample the distribution.
    Instead we directly create a Hankel from it.

    """
    def __init__(
            self, lmbda, max_components, n_observations,
            max_basis_size, estimator, selection="cv",
            n_samples=1000, n_dev_samples=100,
            loss="kl", n_candidates=10, n_basis_symbols=1):

        self.b_0 = None
        self.b_inf = None
        self.B_o = None

        self.max_components = max_components
        self.lmbda = lmbda
        self._observations = range(n_observations)
        self.max_basis_size = max_basis_size
        self.estimator = estimator

        # cv = do cross validation, using samples from the distribution as
        #     a validation set, and negative log-likelihood (plus lambda * dim)
        #     as the validation loss
        # sv = cutoff singular values that do not reach the required threshold.
        # fixed = treat lmbda as a fixed number of components to use.
        assert selection in ['cv', 'sv', 'fixed', 'max']
        self.selection = selection

        self.n_samples = n_samples
        self.n_dev_samples = n_dev_samples

        # Only used when selection == 'cv'
        assert loss in ['kl', 'one-norm']
        self.loss = loss

        self.n_candidates = n_candidates
        self.n_basis_symbols = n_basis_symbols

        self.n_oversamples = 10
        self.n_iter = 5

        self.rng = np.random.RandomState(1)

    def select_n_components(
            self, dist, words, hp, hs, hankel, symbol_hankels, learn_halt):

        if self.selection == "fixed":
            return int(self.lmbda)

        if self.selection == "max":
            return min(hankel.shape)

        if (self.max_components == np.inf or
                self.max_components >= min(hankel.shape)):

            U, Sigma, VT = np.linalg.svd(hankel, full_matrices=False)
            n_avail_components = len(Sigma)
        else:
            U, Sigma, VT = randomized_svd(
                hankel, self.max_components, self.n_oversamples,
                self.n_iter, random_state=self.rng)
            n_avail_components = self.max_components

        if self.selection == 'sv':
            # Hacky: always keep at least 2 singular values, compare against
            # the second sv since the first tends to be on a different order
            # of magnitude than the rest.
            for i, s in enumerate(Sigma[2:]):
                if s < self.lmbda * Sigma[1]:
                    return i + 2
        elif self.selection == 'cv':
            # Potential alternative: instead of sampling the space of
            # components evenly, just keep increasing by one until we no longer
            # have an improvement in the regularized loss function. Depends how
            # much the loss function fluctuates.

            # Assume for now that we are sampling the dev data.
            dev_data, dev_indices = self.sample_words(
                self.n_dev_samples, dist, words, self.rng)

            V = VT.T

            n_candidates = min(n_avail_components, self.n_candidates)
            candidates = np.linspace(
                1, n_avail_components, n_candidates).astype('i')
            assert len(candidates) > 0

            best = (np.inf, 0)
            for n_components in candidates:
                _U = U[:, :n_components]
                _V = V[:, :n_components]
                _Sigma = np.diag(Sigma[:n_components])

                self._estimate(hp, hs, symbol_hankels, _U, _Sigma, _V)

                loss = self.lmbda * n_components
                if self.loss == 'kl':
                    for dd, idx in zip(dev_data, dev_indices):
                        loss -= (
                            self.get_string_prob(dd, log=True)
                            if learn_halt
                            else self.get_prefix_prob(dd, log=True))

                elif self.loss == 'one-norm':
                    for dd, idx in zip(dev_data, dev_indices):
                        est_prob = (
                            self.get_string_prob(dd, log=False)
                            if learn_halt
                            else self.get_prefix_prob(dd, log=False))
                        loss += abs(est_prob - dist[idx])
                else:
                    raise NotImplementedError("Invalid loss: %s." % self.loss)

                if loss < best[0]:
                    best = (loss, n_components)

            return best[1]
        else:
            raise NotImplementedError(
                "%s is not a valid selection method." % self.selection)

        return len(Sigma)

    @staticmethod
    def sample_words(n_samples, dist, words, rng):
        sample = rng.multinomial(n_samples, dist)
        sampled_words = []
        sampled_indices = []
        for idx, (s, word) in enumerate(zip(sample, words)):
            for i in range(s):
                sampled_words.append(word)
                sampled_indices.append(idx)
        return sampled_words, sampled_indices

    def fit(self, dist, words, learn_halt):
        # learn_halt is a feature of the data. Basically if all the
        # strings in the dataset are the same length by design, then
        # learn_halt should be false.
        data, _ = self.sample_words(self.n_samples, dist, words, self.rng)

        if learn_halt:
            basis = top_k_basis(
                data, self.max_basis_size, self.estimator)
        else:
            if self.n_basis_symbols:
                basis = fixed_length_basis(
                    self.observations, self.n_basis_symbols, with_empty=True)
            else:
                basis = top_k_basis(
                    data, self.max_basis_size, self.estimator,
                    max_length=len(data[0])/2)
                # length = len(data[-1])
                # if length % 2 == 1:
                #     prefix_length = suffix_length = int(length/2)
                # else:
                #     prefix_length = int(length/2)
                #     suffix_length = int(length/2) - 1
                # basis = fixed_length_basis(
                #     self.observations, prefix_length,
                #     suffix_length, with_empty=False)

        self.basis = basis
        mapping = {tuple(w): p for w, p in zip(words, dist)}

        f = build_frequencies(mapping, self.estimator, basis)
        hankels = hankels_from_callable(
            f, basis, self.observations, sparse=False)

        hp, hs, hankel, symbol_hankels = hankels

        n_components = self.select_n_components(
            dist, words, hp, hs, hankel, symbol_hankels, learn_halt)
        self.n_components_ = n_components

        # H = U Sigma V^T
        U, Sigma, VT = randomized_svd(
            hankel, n_components, self.n_oversamples,
            self.n_iter, random_state=self.rng)
        Sigma = np.diag(Sigma)

        V = VT.T

        self._estimate(hp, hs, symbol_hankels, U, Sigma, V)

        return self

    def _estimate(self, hp, hs, symbol_hankels, U, Sigma, V):
        # P^+ = (HV)^+ = (U Sigma)^+ = Sigma^+ U+ = Sigma^-1 U.T
        P_plus = np.linalg.pinv(Sigma).dot(U.T)

        # S^+ = (V.T)^+ = V
        S_plus = V

        self.B_o = {}
        for o in self.observations:
            B_o = P_plus.dot(symbol_hankels[o]).dot(S_plus)
            self.B_o[o] = B_o

        self.B = sum(self.B_o.values())

        # b_0 S = hs => b_0 = hs S^+
        b_0 = hs.dot(S_plus)

        # P b_inf = hp => b_inf = P^+ hp
        b_inf = P_plus.dot(hp)

        self.compute_start_end_vectors(b_0, b_inf, self.estimator)
        self.reset()


class SpectralLDA(SequentialLDA, Estimator):
    record_attrs = ['n_topics', 'lmbda']

    def __init__(
            self, n_topics=1, lmbda=1.0, max_components=np.inf,
            max_basis_size=100, estimator='substring', selection="cv",
            n_samples=1000, n_dev_samples=200, loss="kl", n_candidates=10,
            n_basis_symbols=1, directory="results/lda_spectral/",
            name="SpectralLDA"):

        self._init(locals())

    def point_distribution(self, context):
        d = dict(n_topics=np.arange(2, context['max_topics']))

        if self.selection == 'sv':
            d['lmbda'] = uniform(0, 1)
        elif self.selection == 'cv':
            d['lmbda'] = expon(scale=1.0)
        elif self.selection == "fixed":
            d['lmbda'] = [context['max_states_per_topic']]
        elif self.selection == "max":
            d['lmbda'] = [0]
        else:
            raise NotImplementedError()

        return d

    def fit(self, X, y=None):
        """ X is instance of ``MultitaskSequenceDataset``
            with a ``dictionary`` attr.  """
        self.record_indices(X.indices)

        corpus = X.core_data
        training_words = corpus.get_all_words()
        n_word_types = len(training_words)

        def spectral_callback(class_word):
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            self.generators_ = []

            for k, dist in enumerate(class_word):

                sa = SimpleSpectralSAFromDist(
                    self.lmbda, self.max_components, X.n_symbols,
                    self.max_basis_size, self.estimator,
                    self.selection, self.n_samples, self.n_dev_samples,
                    self.loss, self.n_candidates, self.n_basis_symbols)

                sa.fit(dist, training_words, X.learn_halt)

                for w, word in enumerate(training_words):
                    log_prob_w[k][w] = (
                        sa.get_prefix_prob(word, log=True)
                        if not X.learn_halt
                        else sa.get_string_prob(word, log=True))

                self.generators_.append(sa)

            return log_prob_w

        self.alpha_, _, self.gamma_ = fit_callback_lda(
            corpus, self.directory, self.n_topics,
            n_word_types, spectral_callback, log_name="spectral.log")

        if X.n_transfer > 0:
            transfer_corpus = X.transfer_data
            transfer_words = transfer_corpus.get_all_words()
            log_topics = self.log_topics(
                transfer_words, prefix=not X.log_topics)
            n_transfer_word_types = len(transfer_words)

            lhood, transfer_gamma, phi = lda_inference(
                transfer_corpus, self.directory, self.n_topics,
                n_transfer_word_types, self.alpha_, log_topics)

            self.gamma_ = np.concatenate((self.gamma_, transfer_gamma))

        return self
