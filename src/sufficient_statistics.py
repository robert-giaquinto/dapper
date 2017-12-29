import logging
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)


class SufficientStatistics(object):
    """
    Container for holding and update sufficient statistics of DAP model
    """
    def __init__(self, num_topics, vocab_size, num_authors, num_personas, num_times):
        self.num_topics = num_topics
        self.vocab_size = vocab_size
        self.num_authors = num_authors
        self.num_personas = num_personas
        self.num_times = num_times

        # initialize the sufficient statistics
        self.beta = np.zeros((num_topics, vocab_size))
        self.kappa = np.zeros((num_authors, num_personas))
        self.x = np.zeros((num_times, num_personas))
        self.x2 = np.zeros((num_times, num_personas))
        self.alpha = np.zeros((num_times, num_topics, num_personas))
        self.batch_size = 0

    def reset(self):
        """
        reset the sufficient statistics (done after each e step)
        :return:
        """
        self.beta = np.zeros((self.num_topics, self.vocab_size))
        self.kappa = np.zeros((self.num_authors, self.num_personas))
        self.x = np.zeros((self.num_times, self.num_personas))
        self.x2 = np.zeros((self.num_times, self.num_personas))
        self.alpha = np.zeros((self.num_times, self.num_topics, self.num_personas))
        self.batch_size = 0

    def update(self, doc, doc_m, doc_tau, log_phi):
        """
        update the ss given some document and variational parameters
        :param doc:
        :param vp:
        :return:
        """
        t = doc.time_id
        self.batch_size += 1
        self.kappa[doc.author_id, :] += doc_tau
        self.x[t, :] += doc_tau
        self.x2[t, :] += doc_tau**2
        self.beta[:, doc.words] += np.exp(log_phi + np.log(doc.counts))
        for p in range(self.num_personas):
            self.alpha[t, :, p] += doc_m * doc_tau[p]

    def merge(self, other):
        """
        merge in sufficient statistics given a batch of other sufficient statistics
        this is needed to run e-step in parallel
        :param stats:
        :return:
        """
        if other is not None:
            self.x += other.x
            self.x2 += other.x2
            self.beta += other.beta
            self.alpha += other.alpha
            self.kappa += other.kappa
            self.batch_size += other.batch_size

    def __str__(self):
        rval = "SufficientStatistics derived from batch_size = {}".format(self.batch_size)
        return rval



