from __future__ import print_function
import os
import numpy as np
from sklearn.utils import check_random_state
from sklearn.base import clone
from scipy.misc import logsumexp

from spectral_dagger import Estimator
from spectral_dagger.sequence import sample_words
from spectral_dagger.utils import normalize
from spectral_dagger.sequence import SequenceModel
from spectral_dagger.utils.dists import MixtureDist

from seq_lda import (
    SequentialLDA, fit_callback_lda, lda_inference, write_settings)
from seq_lda import Dictionary

LOG_ZERO = -10000


class MSSG(SequentialLDA, Estimator):
    def __init__(
            self, bg=None, n_topics=1, n_samples=100, n_samples_scale=None,
            lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False):
        self._set_attrs(locals())

    @property
    def record_attrs(self):
        return super(MSSG, self).record_attrs or set(['n_topics'])

    def point_distribution(self, context):
        pd = super(MSSG, self).point_distribution(context)
        if 'max_topics' in context:
            pd.update(n_topics=range(2, context['max_topics']))
        return pd

    def fit(self, X, y=None):
        """ X is instance of ``MultitaskSequenceDataset``.  """
        random_state = check_random_state(self.random_state)

        self.record_indices(X.indices)

        corpus = X.core_data
        training_words = corpus.get_all_words()
        n_word_types = len(training_words)

        self.base_generators_ = None

        def callback(class_word):
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            old_bgs = self.base_generators_
            self.base_generators_ = []

            for k, dist in enumerate(class_word):
                if self.n_samples_scale is not None:
                    n_samples = int(self.n_samples_scale * len(dist))
                else:
                    n_samples = self.n_samples

                data, _ = sample_words(n_samples, dist, training_words, random_state)

                if old_bgs is None:
                    bg = self.fit_base_generator(data, X)
                else:
                    bg = self.fit_base_generator(data, X, previous=old_bgs[k])

                for w, word in enumerate(training_words):
                    if X.learn_halt:
                        log_prob_w[k][w] = bg.string_prob(word, log=True)
                    else:
                        log_prob_w[k][w] = bg.prefix_prob(word, log=True)

                self.base_generators_.append(bg)

            log_prob_w[np.isnan(log_prob_w)] = LOG_ZERO
            log_prob_w[np.isinf(log_prob_w)] = LOG_ZERO

            return log_prob_w

        settings = None
        if self.lda_settings is not None:
            settings = os.path.join(self.directory, 'settings.txt')
            write_settings(settings, **self.lda_settings)

        self.alpha_, _, self.gamma_ = fit_callback_lda(
            corpus, self.directory, self.n_topics,
            n_word_types, callback,
            settings=settings,
            log_name="" if self.verbose else ("%s.log" % self.name))

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

    def fit_base_generator(self, data, X, previous=None):
        bg = clone(self.bg)
        bg.fit(data)
        return bg


class SingleMSSG(SequenceModel, Estimator):
    def __init__(
            self, bg=None, n_topics=1, n_samples=100, n_samples_scale=None,
            lda_settings=None, directory=None, name=None,
            random_state=None, verbose=False, to_hashable=lambda x: tuple(x),
            learn_halt=False):
        self._set_attrs(locals())

    @property
    def record_attrs(self):
        return super(SingleMSSG, self).record_attrs or set(['n_topics'])

    def point_distribution(self, context):
        pd = super(SingleMSSG, self).point_distribution(context)
        pd.update(n_topics=range(2, context['max_topics']))
        return pd

    def fit_base_generator(self, data, X, previous=None):
        bg = clone(self.bg)
        bg.fit(data)
        return bg

    def fit(self, X, y=None):
        # X is a single list of sequences (so just 1 task), turn it into a corpus.
        random_state = check_random_state(self.random_state)

        d = Dictionary()
        corpus = d.encode([X], to_hashable=self.to_hashable)

        training_words = corpus.get_all_words()
        n_word_types = len(training_words)

        self.base_generators_ = None

        def callback(class_word):
            class_word = normalize(class_word, ord=1, axis=1)
            log_prob_w = np.zeros_like(class_word)

            old_bgs = self.base_generators_
            self.base_generators_ = []

            for k, dist in enumerate(class_word):
                if self.n_samples_scale is not None:
                    n_samples = int(self.n_samples_scale * len(dist))
                else:
                    n_samples = self.n_samples

                data, _ = sample_words(n_samples, dist, training_words, random_state)

                if old_bgs is None:
                    bg = self.fit_base_generator(data, X)
                else:
                    bg = self.fit_base_generator(data, X, previous=old_bgs[k])

                for w, word in enumerate(training_words):
                    if self.learn_halt:
                        log_prob_w[k][w] = bg.string_prob(word, log=True)
                    else:
                        log_prob_w[k][w] = bg.prefix_prob(word, log=True)

                self.base_generators_.append(bg)

            log_prob_w[np.isnan(log_prob_w)] = LOG_ZERO
            log_prob_w[np.isinf(log_prob_w)] = LOG_ZERO

            return log_prob_w

        settings = None
        if self.lda_settings is not None:
            settings = os.path.join(self.directory, 'settings.txt')
            write_settings(settings, **self.lda_settings)

        self.alpha_, _, self.gamma_ = fit_callback_lda(
            corpus, self.directory, self.n_topics,
            n_word_types, callback,
            settings=settings,
            log_name="" if self.verbose else ("%s.log" % self.name))

        return self

    @property
    def coefficients(self):
        if not hasattr(self, 'gamma_'):
            raise Exception("SingleMSSG has no coefficients, hasn't been fit yet.")
        assert self.gamma_.shape[0] == 1
        gamma = self.gamma_[0, :].copy()
        theta = normalize(gamma, ord=1)
        return theta

    @property
    def seq_gens(self):
        if not hasattr(self, 'gamma_'):
            raise Exception("SingleMSSG has no coefficients, hasn't been fit yet.")
        return self.base_generators_

    @property
    def log_coef(self):
        return np.log(self.coefficients)

    @property
    def action_space(self):
        return None

    @property
    def observation_space(self):
        return self.sg.observation_space

    @property
    def n_observations(self):
        n_obs = getattr(self.seq_gens[0], 'n_observations', None)
        if n_obs is None:
            raise Exception(
                "Cannot determine ``n_observations`` for mixture model, "
                "component sequence generators are continuous.")
        return n_obs

    @property
    def can_terminate(self):
        return all(seq_gen.can_terminate for seq_gen in self.seq_gens)

    def in_terminal_state(self):
        return self.choice.terminal

    def has_terminal_states(self):
        return self.seq_gens[0].can_terminate

    def has_reward(self):
        return False

    def reset(self, initial=None):
        for sg in self.seq_gens:
            sg.reset()
        self.state_dist = self.log_coef.copy()

    def check_terminal(self, obs):
        """ Not implemented since we use our own method of generation. """
        raise NotImplementedError()

    def _reset(self, initial=None):
        """ Not implemented since we use our own method of generation. """
        raise NotImplementedError()

    def update(self, o):
        """ Update state upon seeing an observation. """
        log_cond_probs = np.array([np.log(sg.cond_obs_prob(o)) for sg in self.seq_gens])
        log_p = log_cond_probs + self.state_dist
        self.state_dist = log_p - logsumexp(log_p)

        for sg in self.seq_gens:
            sg.update(o)

    def cond_obs_prob(self, o, log=False):
        """ Get probability of observation for next time step. """
        weights = np.array([np.log(sg.cond_obs_prob(o)) for sg in self.seq_gens])
        p = self.state_dist + weights
        ans = logsumexp(p)
        return ans if log else np.exp(ans)

    def cond_termination_prob(self, log=False):
        """ Get probability of terminating.  """
        weights = np.array([
            np.log(sg.cond_termination_prob()) for sg in self.seq_gens])
        p = self.state_dist + weights
        ans = logsumexp(p)
        return ans if log else np.exp(ans)

    def cond_predict(self):
        idx = np.argmax(self.state_dist)
        return self.seq_gens[idx].cond_predict()

    def cond_obs_dist(self, log=False):
        """ Get distribution over observations for next time step,
            including termination observations.

        """
        return MixtureDist(
            np.exp(self.state_dist),
            [sg.cond_obs_dist() for sg in self.seq_gens])

    def string_prob(self, string, log=False):
        """ Get probability of string. """
        log_weights = np.array([
            sg.string_prob(string, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + self.log_coef)
        return log_prob if log else np.exp(log_prob)

    def delayed_string_prob(self, string, log=False):
        """ Get probability of string. """
        log_weights = np.array([
            sg.delayed_string_prob(string, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + self.log_coef)
        return log_prob if log else np.exp(log_prob)

    def prefix_prob(self, prefix, log=False):
        """ Get probability of prefix. """
        log_weights = np.array([
            sg.prefix_prob(prefix, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + self.log_coef)
        return log_prob if log else np.exp(log_prob)

    def delayed_prefix_prob(self, prefix, t, log=False):
        """ Get probability of observing prefix at a delay of ``t``.

        delayed_prefix_prob(p, 0) is equivalent to prefix_prob(p).

        """
        log_weights = np.array([
            sg.delayed_prefix_prob(prefix, log=True) for sg in self.seq_gens])
        log_prob = logsumexp(log_weights + self.log_coef)
        return log_prob if log else np.exp(log_prob)

    def substring_expectation(self, substring):
        """ Get expected number of occurrences of a substring. """
        weights = np.array([
            sg.substring_expectation(substring) for sg in self.seq_gens])

        return (self.coefficients * weights).sum()
