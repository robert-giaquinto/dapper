import numpy as np
import time
import logging
from scipy.special import psi, gammaln
from scipy.misc import logsumexp
from math import ceil
from multiprocessing import Pool, Queue, cpu_count
from collections import deque

from src.sufficient_statistics import SufficientStatistics
from src.utilities import softmax, sum_normalize, matrix2str, dirichlet_expectation, sample_batches


logger = logging.getLogger(__name__)
SHOW_EVERY = 0


class DAPPER(object):
    """
    Dynamic Author-Personas Performed Exceedingly Rapidly (DAPPER) Topic Model
    """

    def __init__(self, num_topics, num_personas,
                 process_noise=0.2, measurement_noise=0.8, regularization=0.0, normalization="sum",
                 max_epochs=10, em_convergence=1e-03, step_size=0.7, queue_size=10,
                 local_param_iter=50, batch_size=-1, learning_offset=10, learning_decay=0.7,
                 num_workers=1):
        self.batch_size = batch_size
        self.learning_offset = learning_offset
        self.learning_decay = learning_decay
        self.step_size = step_size

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.regularization = regularization
        self.normalization = normalization
        self.num_workers = num_workers

        self.num_topics = num_topics
        self.num_personas = num_personas

        self.current_em_iter = 0
        self.total_epochs = 0
        self.max_epochs = max_epochs
        self.em_convergence = em_convergence
        self.local_param_iter = local_param_iter
        self.queue_size=queue_size
        self._check_params()

    def _check_params(self):
        """
        Check that given parameters to the model are legitimate.
        :return:
        """
        if self.num_personas is None or self.num_personas <= 0:
            raise ValueError("number of personas > 0")
        if self.num_topics is None or self.num_topics <= 0:
            raise ValueError("number of topics > 0")
        if self.num_workers > cpu_count():
            logger.info(
                "Cannot have more workers than available CPUs, setting number workers to {}".format(cpu_count() - 1))
            self.num_workers = cpu_count() - 1
        if self.learning_decay < 0.5 or self.learning_decay > 1.0:
            raise ValueError("learning decay must by in [0.5, 1.0] to ensure convergence.")
        if self.step_size < 0 or self.step_size > 1.0:
            raise ValueError("step size must be in [0, 1].")
        if self.process_noise < 0:
            raise ValueError(
                "process noise must be positive (recommended to be between 0.5 to 0.1 times measurement_noise).")
        if self.measurement_noise < 0:
            raise ValueError(
                "process noise must be positive (recommended to be between 2 to 10 times process_noise).")
        if self.regularization < 0.0 or self.regularization > 0.5:
            raise ValueError("regularization parameter is recommended to be between [0, 0.5]")

    def _check_corpora(self, train_corpus, test_corpus):
        """
        Check that corpus being tested on has the necessary same meta data as the
        training corpus
        :param train_corpus:
        :param test_corpus:
        :return:
        """
        if train_corpus.vocab != test_corpus.vocab:
            raise ValueError("vocabularies for training and test sets must be equal")
        if train_corpus.num_authors != test_corpus.num_authors:
            raise ValueError("Training and test sets must have the same number of authors")

        common_authors = train_corpus.author2id.items() & test_corpus.author2id.items()
        if len(common_authors) != len(train_corpus.author2id):
            raise ValueError("Training and test sets must have the same authors (and author2id lookup tables)")

        # TODO should setup test corpus capable of having *fewer* time steps than training set
        if train_corpus.num_times != test_corpus.num_times:
            raise ValueError("Training and test sets must have the same number of time steps")

    def init_from_corpus(self, corpus):
        """
        initialize model based on the corpus that will be fit
        :param corpus:
        :return:
        """
        # initializations from corpus
        self.vocab_size = corpus.vocab_size
        self.num_times = corpus.num_times
        self.times = corpus.times

        self.total_documents = corpus.total_documents
        self.total_words = corpus.total_words
        self.num_authors = corpus.num_authors
        self.vocab = np.array(corpus.vocab)
        self.id2author = {y: x for x, y in corpus.author2id.items()}

    def parameter_init(self):
        # initialize matrices
        self.omega = np.ones(self.num_personas) * (1.0 / self.num_personas)
        self._delta = np.random.gamma(500., 0.1, (self.num_authors, self.num_personas))
        self.E_log_kappa = dirichlet_expectation(self._delta)

        self.eta = 1.0 / self.num_topics
        self._lambda = np.random.gamma(500., 0.1, (self.num_topics, self.vocab_size))
        self.E_log_beta = dirichlet_expectation(self._lambda)
        self.log_beta = np.zeros((self.num_topics, self.vocab_size))

        self.mu0 = np.ones(self.num_topics) * (1.0 / self.num_topics)
        self.alpha = np.zeros((self.num_times, self.num_topics, self.num_personas))
        self.sigma = np.zeros((self.num_times, self.num_topics, self.num_topics))
        self.sigma_inv = np.zeros((self.num_times, self.num_topics, self.num_topics))

        for t in range(self.num_times):
            alpha_init = np.random.uniform(0.01, 5.0, (self.num_topics, self.num_personas))
            self.alpha[t, :, :] = alpha_init

            if t == 0:
                self.sigma[t, :, :] = np.eye(self.num_topics) * self.process_noise * (self.times[1] - self.times[0])
            else:
                self.sigma[t, :, :] = np.eye(self.num_topics) * self.process_noise * (self.times[t] - self.times[t - 1])

            self.sigma_inv[t, :, :] = np.linalg.inv(self.sigma[t, :, :])

        # normalize alpha
        for p in range(self.num_personas):
            self.alpha[:, :, p] = softmax(self.alpha[:, :, p], axis=1)

        self.alpha_hat = np.copy(self.alpha)

        self.ss = SufficientStatistics(num_topics=self.num_topics,
                                       vocab_size=self.vocab_size,
                                       num_authors=self.num_authors,
                                       num_personas=self.num_personas,
                                       num_times=self.num_times)
        self.ss.reset()
        self.ss_queue = deque(maxlen=self.queue_size)

    def cvi_gamma_update(self, doc_topic_param_1, doc_topic_param_2, doc_zeta_factor, doc_tau, sum_phi, doc_word_count, t):
        sigma_inv = self.sigma_inv[t, :, :]
        alpha = self.alpha[t, :, :]

        nat_param_1 = sigma_inv.dot(alpha.dot(doc_tau))
        nat_param_1 += sum_phi

        nat_param_2 = -0.5 * np.diag(sigma_inv)
        nat_param_2 -= (doc_word_count / (2 * np.sum(np.exp(doc_zeta_factor)))) * np.exp(doc_zeta_factor)

        new_doc_topic_param_1 = self.step_size * nat_param_1 + (1 - self.step_size) * doc_topic_param_1
        new_doc_topic_param_2 = self.step_size * nat_param_2 + (1 - self.step_size) * doc_topic_param_2

        new_doc_m = -1 * (new_doc_topic_param_1 / (2 * new_doc_topic_param_2))
        new_doc_vsq = -1 / (2 * new_doc_topic_param_2)

        return new_doc_m, new_doc_vsq, new_doc_topic_param_1, new_doc_topic_param_2

    def cvi_tau_update(self, doc_tau, doc_persona_param, doc_m, t, a):
        alpha = self.alpha[t, :, :]
        sigma_inv = self.sigma_inv[t, :, :]
        sigma = self.sigma[t, :, :]
        gradient = alpha.T.dot(sigma_inv).dot(doc_m - alpha.dot(doc_tau))
        for p in range(self.num_personas):
            S = np.eye(self.num_topics) * (np.diag(alpha[:, p] ** 2) + sigma)
            gradient[p] -= 0.5 * np.trace(sigma_inv.dot(S))

        new_doc_persona_param = self.step_size * (self.E_log_kappa[a, :] + gradient) + (1 - self.step_size) * doc_persona_param
        # new_doc_tau = np.exp(new_doc_persona_param) / np.sum(np.exp(new_doc_persona_param))
        new_doc_tau = doc_tau * np.exp(new_doc_persona_param)
        new_doc_tau /= np.sum(new_doc_tau)

        return new_doc_tau, new_doc_persona_param

    """=================================================================================================================
            E-step and M-step of the Variational Inference algorithm
    ================================================================================================================="""

    def m_step_s(self, batch_size, num_docs_per_time):
        """
        Stochastic M-step: update the variational parameter for topics using a mini-batch of documents
        """
        if batch_size == self.total_documents:
            rhot = 1.0
        else:
            rhot = np.power(self.learning_offset + self.current_em_iter / 2, -self.learning_decay)

        # topic maximization
        if self.queue_size > 1:
            self.ss_queue.append(self.total_documents * self.ss.beta / batch_size)
            if len(self.ss_queue) < self.queue_size:
                self.beta_ss = sum(self.ss_queue) / len(self.ss_queue)
            else:
                self.beta_ss += (self.ss_queue[-1] - self.ss_queue[0]) / self.queue_size

            lambda_gradient = ((self.eta - self._lambda) + self.beta_ss)
            self._lambda = self._lambda + rhot * lambda_gradient
        else:
            new_lambda = self.eta + self.total_documents * self.ss.beta / batch_size
            self._lambda = (1 - rhot) * self._lambda + rhot * new_lambda

        self.E_log_beta = dirichlet_expectation(self._lambda)
        # normalize
        beta = np.where(self.ss.beta <= 0, 1e-30, self._lambda)
        beta /= np.sum(beta, axis=1, keepdims=True)
        self.log_beta = np.log(beta)

        # beta = np.exp(dirichlet_expectation(self._lambda))
        # beta /= np.sum(beta, axis=1, keepdims=True)

        # update the kappa terms
        new_delta = self.omega + (self.total_documents * self.ss.kappa / batch_size)
        self._delta = (1 - rhot) * self._delta + rhot * new_delta
        self.E_log_kappa = dirichlet_expectation(self._delta)

        # update omega
        # self.omega = self._delta.sum(axis=0) / self.num_authors
        # logger.info('omega\n' + ' '.join([str(round(elt, 2)) for elt in self.omega]) + "\n")

        # estimate a new noisy estimate of alpha
        new_alpha_hat = self.estimate_alpha(num_docs_per_time)
        # logger.info('new ahat\n' + matrix2str(new_alpha_hat[0:15, :, 0], 3))

        # update alpha hat using svi update rule
        self.alpha_hat = (1 - rhot) * self.alpha_hat + rhot * new_alpha_hat
        # logger.info('svi ahat\n' + matrix2str(self.alpha_hat[0:15, :, 0], 3))

        # normalize alpha hat before applying kalman filter smoother
        if self.normalization == 'softmax':
            for p in range(self.num_personas):
                try:
                    self.alpha_hat[:, :, p] = softmax(self.alpha_hat[:, :, p], axis=1)
                except Warning:
                    for t in range(self.num_times):
                        if np.any(np.abs(self.alpha_hat[t, :, p]) > 50.0):
                            self.alpha_hat[t, :, p] /= (np.max(np.abs(self.alpha_hat[t, :, p])) / 50.0)
                    self.alpha_hat[:, :, p] = softmax(self.alpha_hat[:, :, p], axis=1)
            # logger.info('normal ahat\n' + matrix2str(self.alpha_hat[0:15, :, 0], 3))
        elif self.normalization == "sum":
            for p in range(self.num_personas):
                self.alpha_hat[:, :, p] = sum_normalize(self.alpha_hat[:, :, p], axis=1)
            # logger.info('normal ahat\n' + matrix2str(self.alpha_hat[0:15, :, 0], 3))

        # kalman smoothing
        self.alpha = self.smooth_alpha()
        # logger.info('smoothed norm\n' + matrix2str(self.alpha[0:15, :, 0], 3))

        # update priors mu0
        for k in range(self.num_topics):
            self.mu0[k] = np.sum(self.alpha[0, k, :]) * (1.0 / self.num_personas)
        # logger.info('mu\n' + ' '.join([str(round(elt, 3)) for elt in self.mu0]) + '\n')

    def estimate_alpha(self, num_docs_per_time):
        """
        Estimation of alpha in the estep
        Solved using system of linear equations: Ax = b where unknown x is alpha_hat
        See Equation (6) of paper
        :return:
        """
        # compute noisy alpha hat at each time step
        # solve: A (alpha) = b
        alpha_hat = np.zeros((self.num_times, self.num_topics, self.num_personas))
        for t in range(self.num_times):
            if num_docs_per_time[t] == 0:
                if t > 0:
                    alpha_hat[t, :, :] = alpha_hat[t-1, :, :]
                continue

            if t == 0:
                b = self.mu0[:, np.newaxis] + self.ss.alpha[t, :, :] - self.ss.x[t, :]
            else:
                b = alpha_hat[t-1, :, :] + self.ss.alpha[t, :, :] - self.ss.x[t, :]

            denom = self.ss.x2[t, :] + 1.0
            if self.regularization > 0:
                A = np.ones((self.num_personas, self.num_personas))
                A *= (self.regularization / self.num_personas) * num_docs_per_time[t]
                A[np.diag_indices_from(A)] = denom
                try:
                    alpha_hat[t, :, :] = np.linalg.solve(A, b.T).T
                except np.linalg.linalg.LinAlgError:
                    logger.warning("Singular matrix in solving for alpha hat. A:\n" + matrix2str(A, 2) + "\n")
                    logger.warning("Singular matrix in solving for alpha hat. b^T:\n" + matrix2str(b.T, 2) + "\n")
                    alpha_hat[t, :, :] = 1.0 * b / denom[np.newaxis, :]
            else:
                alpha_hat[t, :, :] = 1.0 * b / denom[np.newaxis, :]

        return alpha_hat

    def smooth_alpha(self):
        # compute forward and backward variances
        forward_var, backward_var, P = self.variance_dynamics()

        # smooth noisy alpha using forward and backwards equations
        backwards_alpha = np.zeros((self.num_times, self.num_topics, self.num_personas))
        for p in range(self.num_personas):
            # forward equations
            # initialize forward_alpha[p]
            forwards_alpha = np.zeros((self.num_times, self.num_topics))
            forwards_alpha[0, :] = self.alpha_hat[0, :, p]
            for t in range(1, self.num_times):
                step = P[t, :] / (self.measurement_noise + P[t, :])
                forwards_alpha[t, :] = step * self.alpha_hat[t, :, p] + (1 - step) * forwards_alpha[t - 1, :]

            # backward equations
            for t in range(self.num_times - 1, -1, -1):
                if t == (self.num_times - 1):
                    backwards_alpha[self.num_times - 1, :, p] = forwards_alpha[self.num_times - 1, :]
                    continue

                if t == 0:
                    delta = self.process_noise * (self.times[1] - self.times[0])
                else:
                    delta = self.process_noise * (self.times[t] - self.times[t - 1])

                step = delta / (forward_var[t, :] + delta)
                backwards_alpha[t, :, p] = step * forwards_alpha[t, :] + (1 - step) * backwards_alpha[t + 1, :, p]

        return backwards_alpha

    def variance_dynamics(self):
        # compute forward variance
        forward_var = np.zeros((self.num_times, self.num_topics))
        P = np.zeros((self.num_times, self.num_topics))
        for t in range(self.num_times):
            if t == 0:
                P[t, :] = np.diag(self.sigma[0, :, :]) + self.process_noise * (self.times[1] - self.times[0])
            else:
                P[t, :] = forward_var[t-1, :] + self.process_noise * (self.times[t] - self.times[t-1])

            # use a fixed estimate of the measurement noise
            forward_var[t, :] = self.measurement_noise * P[t, :] / (P[t, :] + self.measurement_noise)

        # compute backward variance for persona p
        backward_var = np.zeros((self.num_times, self.num_topics))
        for t in range(self.num_times-1, -1, -1):
            backward_var[t, :] = forward_var[t, :]
            if t != (self.num_times - 1):
                backward_var[t, :] += (forward_var[t, :]**2 / P[t+1, :]**2) * (backward_var[t+1, :] - P[t+1, :])

        return forward_var, backward_var, P

    def compute_topic_lhood(self):
        # compute the beta terms lhood
        rval = self.num_topics * (gammaln(np.sum(self.eta)) - np.sum(gammaln(self.eta)))
        rval += np.sum(np.sum(gammaln(self._lambda), axis=1) - gammaln(np.sum(self._lambda, axis=1)))

        # # calculation used in online LDA code (doesn't give a reasonable value though...)
        # E_log_beta = psi(self._lambda) - psi(self._lambda.sum(axis=1, keepdims=True))
        # t1 = np.sum((self.eta - self._lambda) * E_log_beta)
        # t2 = np.sum(gammaln(self._lambda) - gammaln(self.eta))
        # t3 = np.sum(gammaln(self.vocab_size * self.eta) - gammaln(np.sum(self._lambda, axis=1)))
        # old = t1 + t2 + t3
        return rval

    def compute_author_lhood(self):
        # compute the kappa terms lhood
        rval = self.num_authors * (gammaln(self.omega.sum()) - gammaln(self.omega).sum())

        # note: cancellation of omega between model and entropy term
        rval += np.sum((self._delta - 1.0) * self.E_log_kappa)

        # entropy term
        rval += np.sum(gammaln(self._delta) - gammaln(self._delta.sum(axis=1, keepdims=True)))
        return rval

    def compute_persona_lhood(self):
        """
        The bound on just the alpha terms is usually small since it primarly depends on
        the different between alpha_t and alpha_[t-1] which is small if the model
        is properly smoothed.
        :return:
        """
        rval = 0.0
        for t in range(self.num_times):
            if t == 0:
                alpha_prev = np.tile(self.mu0, self.num_personas).reshape((self.num_personas, self.num_topics)).T
            else:
                alpha_prev = self.alpha[t - 1, :, :]
            alpha = self.alpha[t, :, :]
            alpha_dif = alpha - alpha_prev
            sigma_inv = self.sigma_inv[t, :, :]
            sigma = self.sigma[t, :, :]

            # 0.5 * log | Sigma^-1| - log 2 pi
            rval -= self.num_personas * 0.5 * np.log(np.linalg.det(sigma))
            rval -= self.num_topics * self.num_personas * 0.5 * np.log(2 * 3.1415)
            # quadratic term
            delta = self.process_noise * (self.times[t] - self.times[t - 1] if t > 0 else self.times[1] - self.times[0])
            for p in range(self.num_personas):
                rval -= 0.5 * (alpha_dif[:, p].T.dot(sigma_inv).dot(alpha_dif[:, p]) + (1.0 / delta) * np.trace(sigma))

        return rval

    def compute_doc_lhood(self, doc, doc_tau, doc_m, doc_vsq, log_phi):
        sigma = self.sigma[doc.time_id, :, :]
        sigma_inv = self.sigma_inv[doc.time_id, :, :]
        alpha = self.alpha[doc.time_id, :, :]

        # term1: sum_P [ tau * (digamma(delta) - digamma(sum(delta))) ]
        lhood = self.E_log_kappa[doc.author_id, :].dot(doc_tau)

        # term 2: log det inv Sigma - k/2 log 2 pi +
        # term 2: -0.5 * (gamma - alpha*tau) sigma_inv (gamma - alpha_tau) +
        #         -0.5 * tr(sigma_inv vhat) +
        #         -0.5 * tr(sigma_inv diag(tau alpha + sigma_hat)
        # note - K/2 log 2 pi cancels with term 6
        lhood -= 0.5 * np.log(np.linalg.det(sigma) + 1e-30)
        alpha_tau = alpha.dot(doc_tau)
        lhood -= 0.5 * (doc_m - alpha_tau).T.dot(sigma_inv).dot(doc_m - alpha_tau)
        lhood -= 0.5 * np.sum(doc_vsq * np.diag(sigma_inv))
        S = np.zeros((self.num_topics, self.num_topics))
        for p in range(self.num_personas):
            S += np.eye(self.num_topics) * doc_tau[p] * (np.diag(alpha[:, p] ** 2) + sigma)
        lhood -= 0.5 * np.trace(sigma_inv.dot(S))

        # term 3: - zeta_inv Sum( exp(gamma + vhat) ) + 1 - log(zeta)
        # use the fact that doc_zeta = np.sum(np.exp(doc_m+0.5*doc_v2)), to cancel the factors
        lhood += -1.0 * logsumexp(doc_m + 0.5 * doc_vsq) * np.sum(doc.counts)
        # zeta = np.sum(np.exp(doc_m + 0.5 * doc_vsq))
        # lhood += (-1.0 / zeta) * np.sum(np.exp(doc_m + doc_vsq * 0.5)) * np.sum(doc.counts) + 1 - np.log(zeta) * np.sum(doc.counts)

        # term 4: Sum(gamma * phi)
        lhood += np.sum(np.sum(np.exp(log_phi) * doc.counts, axis=1) * doc_m)

        # term 5: -tau log tau
        lhood -= doc_tau.dot(np.log(doc_tau + 1e-30))

        # term 6: v_hat in entropy
        # Note K/2 log 2 pi cancels with term 2
        lhood += 0.5 * np.sum(np.log(doc_vsq))

        # term 7: phi log phi
        lhood -= np.sum(np.exp(log_phi) * log_phi * doc.counts)
        return lhood

    def compute_word_lhood(self, doc, log_phi):
        # lhood = np.sum(np.exp(log_phi + np.log(doc.counts)) * self.log_beta[:, doc.words])
        lhood = np.sum(np.exp(log_phi + np.log(doc.counts)) * self.E_log_beta[:, doc.words])
        return lhood

    def em_step_s(self, docs, total_docs):
        """
        Performs stochastic EM-update for one iteration using a mini-batch of documents
        and compute the training log-likelihood
        """
        batch_size = len(docs)
        self.current_em_iter += 1
        self.ss.reset()

        # e-step
        clock_e_step = time.process_time()
        if self.num_workers > 1:
            # run e-step in parallel
            batch_lhood, words_lhood = self.e_step_parallel(docs=docs, save_ss=True)
        else:
            batch_lhood, words_lhood = self.e_step(docs=docs, save_ss=True)
        clock_e_step = time.process_time() - clock_e_step
        doc_lhood = batch_lhood * total_docs / batch_size

        # count docs per timestep in this batch for adjusting the regularization
        num_docs_per_time = [0] * self.num_times
        for d in docs:
            time_id = d.time_id
            num_docs_per_time[time_id] += 1

        # m-step
        clock_m_step = time.process_time()
        self.m_step_s(batch_size=batch_size, num_docs_per_time=num_docs_per_time)
        clock_m_step = time.process_time() - clock_m_step

        topic_lhood = self.compute_topic_lhood()
        persona_lhood = self.compute_author_lhood()
        alpha_lhood = self.compute_persona_lhood()

        model_lhood = doc_lhood + topic_lhood + persona_lhood + alpha_lhood
        total_time = clock_e_step + clock_m_step
        return model_lhood, words_lhood, total_time

    def e_step_parallel(self, docs, save_ss):
        batch_size = len(docs)

        # setup stream that returns chunks of documents at each iteration
        # determine size of partitions of documents
        max_docs_per_worker = int(ceil(1.0 * batch_size / self.num_workers))
        docs_per_worker = []
        while sum(docs_per_worker) < batch_size:
            if sum(docs_per_worker) + max_docs_per_worker < batch_size:
                n = max_docs_per_worker
            else:
                n = batch_size - sum(docs_per_worker)
            docs_per_worker.append(n)

        # set up pool of workers and arguments passed to each worker
        job_queue = Queue(maxsize=self.num_workers)
        result_queue = Queue()

        queue_size, reallen = [0], 0
        batch_lhood, words_lhood = [0.0], [0.0]
        def process_result_queue():
            """
            clear result queue, merge intermediate SS
            :return:
            """
            while not result_queue.empty():
                # collect summary statistics and merge all sufficient statistics
                bl, wl, partial_ss = result_queue.get()
                batch_lhood[0] += bl
                words_lhood[0] += wl
                self.ss.merge(partial_ss)
                queue_size[0] -= 1

        with Pool(self.num_workers, _doc_e_step_worker, (job_queue, result_queue,)) as pool:
            # loop through chunks of documents placing chunks on the queue
            for chunk_id, chunk_size in enumerate(docs_per_worker):
                doc_mini_batch = docs[reallen:(reallen + chunk_size)]
                reallen += len(doc_mini_batch)  # track how many documents have been seen
                chunk_put = False
                while not chunk_put:
                    try:
                        args = (self, doc_mini_batch, save_ss)
                        job_queue.put(args, block=False, timeout=0.1)
                        chunk_put = True
                        queue_size[0] += 1
                    except job_queue.full():
                        process_result_queue()

                process_result_queue()

            while queue_size[0] > 0:
                process_result_queue()

            if reallen != batch_size:
                raise RuntimeError("input corpus size changed during training (don't use generators as input)")

        return batch_lhood[0], words_lhood[0]

    def e_step(self, docs, save_ss=True):
        """
        E-step: update the variational parameters for topic proportions and topic assignments.
        """
        # iterate over all documents
        batch_lhood, batch_lhood_d = 0, 0
        words_lhood = 0
        for doc in docs:
            # initialize gamma for this document
            doc_m = np.ones(self.num_topics)  * (1.0 / self.num_topics)
            doc_vsq = np.ones(self.num_topics)
            doc_topic_param_1 = np.zeros(self.num_topics)
            doc_topic_param_2 = np.zeros(self.num_topics)
            doc_persona_param = np.zeros(self.num_personas)
            doc_tau = np.ones(self.num_personas) * (1.0 / self.num_personas)
            doc_word_count = np.sum(doc.counts)

            # update zeta in close form
            doc_zeta_factor = doc_m + 0.5 * doc_vsq

            iters = 0
            while iters < self.local_param_iter:
                prev_doc_m = doc_m
                iters += 1
                # update phi in closed form
                log_phi = doc_m[:, np.newaxis] + self.log_beta[:, doc.words]
                log_phi -= logsumexp(log_phi, axis=0, keepdims=True)

                # CVI update to m and v
                sum_phi = np.sum(np.exp(log_phi) * doc.counts[np.newaxis, :], axis=1)
                doc_m, doc_vsq, doc_topic_param_1, doc_topic_param_2 = self.cvi_gamma_update(
                    doc_topic_param_1, doc_topic_param_2, doc_zeta_factor, doc_tau, sum_phi, doc_word_count, doc.time_id)

                # CVI update to tau
                doc_tau, doc_persona_param = self.cvi_tau_update(doc_tau, doc_persona_param, doc_m,
                                                                 doc.time_id, doc.author_id)

                # update zeta in closed form
                doc_zeta_factor = doc_m + 0.5 * doc_vsq

                mean_change = np.mean(abs(doc_m - prev_doc_m))
                if mean_change < self.em_convergence:
                    break

            # compute word likelihoods
            words_lhood_d = self.compute_word_lhood(doc, log_phi)
            words_lhood += words_lhood_d

            # collect sufficient statistics
            if save_ss:
                self.ss.update(doc, doc_m, doc_tau, log_phi)

                # if updating the model (save_ss) then also compute variational likelihoods
                batch_lhood_d = self.compute_doc_lhood(doc, doc_tau, doc_m, doc_vsq, log_phi)
                batch_lhood += batch_lhood_d

            if SHOW_EVERY > 0 and doc.doc_id % SHOW_EVERY == 0:
                logger.info("Variational parameters for document: {}, converged in {} steps".format(doc.doc_id, iters))
                logger.info("Per-word likelihood for doc[{}]: ({} + {}) / {} = {:.2f}".format(
                    doc.doc_id, words_lhood_d, batch_lhood_d, doc_word_count, (words_lhood_d + batch_lhood_d) / doc_word_count))
                logger.info("new zeta: " + ' '.join([str(round(g, 3)) for g in doc_zeta_factor]))
                logger.info("new doc_m[{}]: ".format(doc.doc_id) + ' '.join([str(round(g, 3)) for g in doc_m]))
                logger.info("new vhat: " + ' '.join([str(round(g, 3)) for g in doc_vsq]))
                logger.info("new tau: " + ' '.join([str(round(g, 3)) for g in doc_tau]) + "\n")

        return batch_lhood, words_lhood

    """=================================================================================================================
        Training and testing
    ================================================================================================================="""

    def fit(self, corpus, random_beta=False, max_training_minutes=None):
        """
        Performs EM-update until reaching target average change in the log-likelihood
        """
        # init from corpus
        self.init_from_corpus(corpus)
        self.parameter_init()
        if random_beta:
            self.init_beta_random()
        else:
            self.init_beta_from_corpus(corpus=corpus)

        if self.batch_size <= 0:
            batch_size = corpus.total_documents
        else:
            batch_size = self.batch_size

        prev_train_model_lhood, train_model_lhood = np.finfo(np.float32).min, np.finfo(np.float32).min
        train_results = []
        elapsed_time = 0.0
        total_training_docs = 0

        total_time = time.process_time()
        while self.total_epochs < self.max_epochs:
            batches = sample_batches(corpus.total_documents, batch_size)
            epoch_time = 0.0
            finished_epoch = True
            for batch_id, doc_ids in enumerate(batches):
                batch = [corpus.docs[d] for d in doc_ids]
                train_model_lhood, train_words_lhood, batch_time = self.em_step_s(docs=batch, total_docs=corpus.total_documents)

                train_model_pwll = train_model_lhood / corpus.total_words
                train_words_pwll = train_words_lhood / np.sum([np.sum(d.counts) for d in batch])
                total_training_docs += len(batch)
                epoch_time += batch_time
                elapsed_time += batch_time
                convergence = np.abs((train_model_lhood - prev_train_model_lhood) / prev_train_model_lhood)
                train_results.append([self.total_epochs, batch_id,
                                      train_model_lhood, train_model_pwll, train_words_pwll,
                                      convergence, batch_time, elapsed_time, total_training_docs])
                # report current stats
                log_str = "epoch: {}, batch: {}, model ll: {:.1f}, model pwll: {:.2f}, words pwll: {:.2f}, convergence: {:.3f}, time: {:.3f}"
                logger.info(log_str.format(self.total_epochs, batch_id,
                                           train_model_lhood, train_model_pwll, train_words_pwll,
                                           convergence, batch_time))

                prev_train_model_lhood = train_model_lhood
                if max_training_minutes is not None and (elapsed_time / 60.0) > max_training_minutes:
                    finished_epoch = (batch_id + 1) == len(batches)  # finished if we're already through the last batch
                    break

            if finished_epoch:
                # report stats after each full epoch
                self.total_epochs += 1
                self.print_topics_over_time(5)
                self.print_author_personas(max_show=25)
                self.print_topics(topn=8)
                docs_per_hour = total_training_docs / (epoch_time / 60.0 / 60.0)
                log_str = """{} Epochs Completed
                    train model lhood: {:.1f}, model per-word log-lhood: {:.2f}, words per-word log-lhood: {:.2f}, convergence: {:.3f},
                    total minutes training: {:.2f}, previous epoch minutes training: {:.2f}, epoch's docs/hr {:.1f}
                    """
                logger.info(log_str.format(self.total_epochs,
                                           train_model_lhood, train_model_pwll, train_words_pwll, convergence,
                                           elapsed_time / 60.0, epoch_time / 60.0, docs_per_hour))

            if max_training_minutes is not None and (elapsed_time / 60.0) > max_training_minutes:
                logger.info("Maxed training time has elapsed, stopping training.")
                break

        # print last stats
        total_time = time.process_time() - total_time
        self.print_topics_over_time(5)
        self.print_author_personas(max_show=25)
        self.print_topics(topn=8)
        log_str = """Finished after {} EM iterations ({} full epochs, {} total documents)
                    train model lhood: {:.1f}, model per-word log-lhood: {:.2f}, words per-word log-lhood: {:.2f},
                    total minutes elapsed: {:.1f}, total minutes training: {:.3f}, avg docs/hr {:.1f}
                    """
        docs_per_hour = total_training_docs / (elapsed_time / 60.0 / 60.0)
        logger.info(log_str.format(self.current_em_iter, self.total_epochs, total_training_docs,
                                   train_model_lhood, train_model_pwll, train_words_pwll,
                                   total_time / 60.0, elapsed_time / 60.0, docs_per_hour))

        self.print_convergence(train_results, show_batches=False)
        return train_results

    def init_beta_from_corpus(self, corpus, num_docs_init=None):
        """
        initialize model for training using the corpus
        :param corpus:
        :param num_docs_init:
        :return:
        """
        if num_docs_init is None:
            num_docs_init = min(500, int(round(corpus.total_documents * 0.01)))
        if num_docs_init == 0:
            return

        logger.info("Initializing beta from {} random documents in the corpus for each topics".format(num_docs_init))
        for k in range(self.num_topics):
            doc_ids = np.sort(np.random.randint(0, corpus.total_documents, num_docs_init))
            logger.debug("Initializing topic {} from docs: {}".format(k, ' '.join([str(d) for d in doc_ids])))

            for i in doc_ids:
                doc = corpus.docs[i]
                for n in range(doc.num_terms):
                    v = doc.words[n]
                    self._lambda[k, v] += doc.counts[n]

        # save log of beta for VI computations
        self.log_beta = np.log(self._lambda / np.sum(self._lambda, axis=1, keepdims=True))

    def init_beta_random(self):
        """
        random initializations before training
        :return:
        """
        self.log_beta = np.random.uniform(0.01, 0.99, (self.num_topics, self.vocab_size))
        row_sums = self.log_beta.sum(axis=1, keepdims=True)
        self.log_beta = np.log(self.log_beta / row_sums)

    def predict(self, test_corpus):
        """
        Performs E-step on test corpus using stored topics obtained by training
        Computes the average heldout log-likelihood
        """
        if self.num_workers > 1:
            _, test_words_lhood = self.e_step_parallel(docs=test_corpus.docs, save_ss=False)
        else:
            _, test_words_lhood = self.e_step(docs=test_corpus.docs, save_ss=False)

        test_words_pwll = test_words_lhood / test_corpus.total_words
        logger.info('Test words log-lhood: {:.1f}, words per-word log-lhood: {:.2f}'.format(
            test_words_lhood, test_words_pwll))
        return test_words_lhood, test_words_pwll

    def fit_predict(self, train_corpus, test_corpus, evaluate_every=None, max_training_minutes=None, random_beta=False):
        """
        Computes the heldout-log likelihood on the test corpus after "evaluate_every" iterations
        (mini-batches) of training.
        """
        # initializarions from the training corpus
        self.init_from_corpus(train_corpus)
        self.parameter_init()
        if random_beta:
            self.init_beta_random()
        else:
            self.init_beta_from_corpus(corpus=train_corpus)

        # verify that the training and test corpora have same metadata
        self._check_corpora(train_corpus=train_corpus, test_corpus=test_corpus)

        if self.batch_size <= 0:
            batch_size = train_corpus.total_documents
        else:
            batch_size = self.batch_size

        prev_train_model_lhood, train_model_lhood = np.finfo(np.float32).min, np.finfo(np.float32).min
        train_results = []
        test_results = []
        elapsed_time = 0.0
        total_training_docs = 0

        total_time = time.process_time()
        while self.total_epochs < self.max_epochs:
            batches = sample_batches(train_corpus.total_documents, batch_size)
            epoch_time = 0.0
            finished_epoch = True
            for batch_id, doc_ids in enumerate(batches):
                # do an EM iteration on this batch
                batch = [train_corpus.docs[d] for d in doc_ids]
                train_model_lhood, train_words_lhood, batch_time = self.em_step_s(docs=batch, total_docs=train_corpus.total_documents)

                # collect stats from EM iteration
                train_model_pwll = train_model_lhood / train_corpus.total_words
                train_words_pwll = train_words_lhood / np.sum([np.sum(d.counts) for d in batch])
                elapsed_time += batch_time
                epoch_time += batch_time
                total_training_docs += len(batch)
                convergence = np.abs((train_model_lhood - prev_train_model_lhood) / prev_train_model_lhood)
                train_results.append([self.total_epochs, batch_id,
                                      train_model_lhood, train_model_pwll, train_words_pwll,
                                      convergence, batch_time, elapsed_time, total_training_docs])

                # report current stats
                log_str = "epoch: {}, batch: {}, model ll: {:.1f}, model pwll: {:.2f}, words pwll: {:.2f}, convergence: {:.3f}, batch time: {:.3f}, total training minutes: {:.3f}"
                logger.info(log_str.format(self.total_epochs, batch_id, train_model_lhood, train_model_pwll, train_words_pwll, convergence, batch_time, elapsed_time / 60.))

                # test set evaluation:
                if evaluate_every is not None and (batch_id + 1) % evaluate_every == 0 and (batch_id + 1) != len(batches):
                    test_words_lhood, test_words_pwll = self.predict(test_corpus=test_corpus)
                    test_results.append([self.total_epochs, batch_id,
                                         0.0, 0.0, test_words_pwll, convergence,
                                         0.0, elapsed_time, total_training_docs])

                prev_train_model_lhood = train_model_lhood
                if max_training_minutes is not None and (elapsed_time / 60.0) > max_training_minutes:
                    finished_epoch = (batch_id + 1) == len(batches) # finished if we're already through the last batch
                    break

            if finished_epoch:
                self.total_epochs += 1
                # evaluate the log-likelihood at the end of each full epoch
                test_words_lhood, test_words_pwll = self.predict(test_corpus=test_corpus)
                test_results.append([self.total_epochs, -1,
                                     0.0, 0.0, test_words_pwll, convergence,
                                     epoch_time, elapsed_time, total_training_docs])
                self.print_topics_over_time(5)
                self.print_author_personas()
                self.print_topics(topn=8)
                # report stats on this epoch
                docs_per_hour = total_training_docs / (epoch_time / 60.0 / 60.0)
                log_str = """{} Epochs Completed
                    train model lhood: {:.1f}, model per-word log-lhood: {:.2f}, words per-word log-lhood: {:.2f},
                    test words lhood:  {:.1f}, words per-word log-lhood: {:.2f}, convergence: {:.3f},
                    total minutes training: {:.2f}, previous epoch minutes training: {:.2f}, epoch's docs/hr {:.1f}
                    """
                logger.info(log_str.format(self.total_epochs,
                                           train_model_lhood, train_model_pwll, train_words_pwll,
                                           test_words_lhood, test_words_pwll, convergence,
                                           elapsed_time / 60.0, epoch_time / 60.0, docs_per_hour))

            if max_training_minutes is not None and (elapsed_time / 60.0) > max_training_minutes:
                logger.info("Maxed training time has elapsed, stopping training.")
                break

        # print last stats
        total_time = time.process_time() - total_time
        self.print_topics_over_time(5)
        self.print_author_personas()
        self.print_topics(topn=8)
        log_str = """Finished after {} EM iterations ({} full epochs, {} total documents)
            train model lhood: {:.1f}, model per-word log-lhood: {:.2f}, words per-word log-lhood: {:.2f},
            test words lhood:  {:.1f}, words per-word log-lhood: {:.2f},
            total minutes elapsed: {:.1f}, total minutes training: {:.3f}, avg docs/hr {:.1f}
            """
        docs_per_hour = total_training_docs / (elapsed_time / 60.0 / 60.0)
        logger.info(log_str.format(self.current_em_iter, self.total_epochs, total_training_docs,
                                   train_model_lhood, train_model_pwll, train_words_pwll,
                                   test_words_lhood, test_words_pwll,
                                   total_time / 60.0, elapsed_time / 60.0, docs_per_hour))
        self.print_convergence(train_results, show_batches=False)
        self.print_convergence(test_results, show_batches=True)
        return train_results, test_results

    def print_topics_over_time(self, top_n_topics=None):
        for p in range(self.num_personas):
            if top_n_topics is not None:
                topic_totals = np.sum(self.alpha[:, :, p], axis=0)
                top_topic_ids = np.argsort(topic_totals)[-top_n_topics:]
                top_topic_ids.sort()
                alpha = self.alpha[:, top_topic_ids, p]
                logger.info('alpha[p={}] top topic ids: ' + '\t'.join([str(i) for i in top_topic_ids]))
                logger.info('alpha[p={}]\n'.format(p) + matrix2str(alpha, 3))
            else:
                logger.info('alpha[p={}]\n'.format(p) + matrix2str(self.alpha[:, :, p], 3))

    def print_convergence(self, results, show_batches=False):
        logger.info("EM Iteration\tMini-batch\tModel LL\tModel PWLL\tWords PWLL\tConvergence\tSeconds per Batch\tSeconds Training\tDocs Trained")
        for stats in results:
            em_iter, batch_id, model_lhood, model_pwll, words_pwll, convergence, batch_time, training_time, docs_trained = stats
            if batch_id == 0 or show_batches:
                if model_lhood == 0.0:
                    log_str = "{}\t\t{}\t\t{:.1f}\t\t{:.2f}\t\t{:.2f}\t\t{:.4f}\t\t{:.2f}\t\t\t{:.2f}\t\t\t{}"
                else:
                    log_str = "{}\t\t{}\t\t{:.1f}\t{:.2f}\t\t{:.2f}\t\t{:.4f}\t\t{:.2f}\t\t\t{:.2f}\t\t\t{}"
                logger.info(log_str.format(
                    em_iter, batch_id, model_lhood, model_pwll, words_pwll, convergence, batch_time, training_time, docs_trained))

    def print_author_personas(self, max_show=10):
        max_key_len = max([len(k) for k in self.id2author.values()])
        kappa = self._delta / np.sum(self._delta, axis=1, keepdims=True)

        logger.info("Kappa:")
        spaces = ' ' * (max_key_len - 6)
        logger.info("Author{} \t{}".format(spaces, '\t'.join(['p' + str(i) for i in range(self.num_personas)])))
        logger.info('-' * (max_key_len + 10 * self.num_personas))
        for author_id in range(self.num_authors):
            author = self.id2author[author_id]
            pad = ' ' * (max_key_len - len(author))
            if author_id == self.num_authors - 1:
                logger.info(author + pad + "\t" + "\t".join([str(round(k, 2)) for k in kappa[author_id, :]]) + '\n')
            else:
                logger.info(author + pad + "\t" + "\t".join([str(round(k, 2)) for k in kappa[author_id, :]]))

            if author_id > max_show:
                logger.info("...\n")
                break

    def print_topics(self, topn=10, tfidf=True):
        beta = self.get_topics(topn, tfidf)
        for k in range(self.num_topics):
            topic_ = ' + '.join(['%.3f*"%s"' % (p, w) for w, p in beta[k]])
            logger.info("topic #%i: %s", k, topic_)

    def get_topics(self, topn=10, tfidf=True):
        E_log_beta = dirichlet_expectation(self._lambda)
        beta = np.exp(E_log_beta)
        beta /= np.sum(beta, axis=1, keepdims=True)

        if tfidf:
            beta = self._term_scores(beta)

        rval = []
        if topn is not None:
            for k in range(self.num_topics):
                word_rank = np.argsort(beta[k, :])
                sorted_probs = beta[k, word_rank]
                sorted_words = np.array(list(self.vocab))[word_rank]
                rval.append([(w, p) for w, p in zip(sorted_words[-topn:][::-1], sorted_probs[-topn:][::-1])])
        else:
            for k in range(self.num_topics):
                rval.append([(w, p) for w, p in zip(self.vocab, beta[k, :])])
        return rval

    def _term_scores(self, beta):
        """
        TF-IDF type calculation for determining top topic terms
        from "Topic Models" by Blei and Lafferty 2009, equation 3
        :param beta:
        :return:
        """
        denom = np.power(np.prod(beta, axis=0), 1.0 / self.num_topics)
        if np.any(denom == 0):
            denom += 0.000001
        term2 = np.log(np.divide(beta, denom))
        return np.multiply(beta, term2)

    def save_topics(self, filename, topn=10, tfidf=True):
        beta = self.get_topics(topn, tfidf)
        with open(filename, "w") as f:
            for k in range(self.num_topics):
                topic_ = ', '.join([w for w, p in beta[k]])
                f.write("topic #{} ({:.2f}): {}\n".format(k, self.mu0[k], topic_))

    def save_author_personas(self, filename):
        kappa = self._delta / np.sum(self._delta, axis=1, keepdims=True)
        with open(filename, "w") as f:
            f.write("author\t" + "\t".join(["persona" + str(i) for i in range(self.num_personas)]) + "\n")
            for author_id in range(self.num_authors):
                author = self.id2author[author_id]
                f.write(author + "\t" + "\t".join([str(round(k, 7)) for k in kappa[author_id]]) + "\n")

    def save_persona_topics(self, filename):
        with open(filename, "w") as f:
            f.write("time_id\ttime\tpersona\t" + "\t".join(["topic" + str(i) for i in range(self.num_topics)]) + "\n")
            for t in range(self.num_times):
                for p in range(self.num_personas):
                    time_val = self.times[t]
                    f.write("{}\t{}\t{}\t{}\n".format(t, time_val, p, '\t'.join([str(k) for k in self.alpha[t, :, p]])))

    def save_convergnces(self, filename, results):
        with open(filename, 'w') as f:
            f.write("em_iter\tbatch_iter\tmodel_log_lhood\tmodel_pwll\twords_pwll\tconvergence\tbatch_time\ttraining_time\tdocs_trained\n")
            for stats in results:
                f.write('\t'.join([str(stat) for stat in stats]) + '\n')

    def __str__(self):
        rval = """\nDAP Model:
            training iterations: {} ({} epochs)
            trained on corpus with {} time points, {} authors, {} documents, {} total words
            using {} processors
            """.format(self.current_em_iter, self.total_epochs, self.num_times,
                       self.num_authors, self.total_documents, self.total_words, self.num_workers)
        rval += """\nModel Settings:
            regularization {:.2f}
            number of topics: {}
            number of personas: {}
            measurement noise: {}
            process noise: {}
            smoothing last {} gradients
            batch size: {}
            learning offset: {}
            learning decay: {:.2f}
            step size: {:.2f}
            normalization method: {}
            max epochs: {}
            em convergence: {:.2f}
            local parameter max iterations: {}
            """.format(self.regularization, self.num_topics, self.num_personas,
                       self.measurement_noise, self.process_noise,
                       self.queue_size,
                       self.batch_size, self.learning_offset, self.learning_decay, self.step_size,
                       self.normalization, self.max_epochs, self.em_convergence, self.local_param_iter)
        return rval


def _doc_e_step_worker(input_queue, result_queue):
    while True:
        dap, docs, save_ss = input_queue.get()

        if save_ss:
            # initialize sufficient statistics to gather information learned from each doc
            ss = SufficientStatistics(num_topics=dap.num_topics,
                                      vocab_size=dap.vocab_size,
                                      num_authors=dap.num_authors,
                                      num_personas=dap.num_personas,
                                      num_times=dap.num_times)

        # iterate over all documents
        batch_lhood = 0
        words_lhood = 0
        for doc in docs:
            # initialize gamma for this document
            doc_m = np.ones(dap.num_topics)  * (1.0 / dap.num_topics)
            doc_vsq = np.ones(dap.num_topics)
            doc_topic_param_1 = np.zeros(dap.num_topics)
            doc_topic_param_2 = np.zeros(dap.num_topics)
            doc_persona_param = np.zeros(dap.num_personas)
            doc_tau = np.ones(dap.num_personas) * (1.0 / dap.num_personas)
            doc_word_count = np.sum(doc.counts)

            # update zeta in close form
            doc_zeta_factor = doc_m + 0.5 * doc_vsq

            iters = 0
            while iters < dap.local_param_iter:
                prev_doc_m = doc_m
                iters += 1
                # update phi in closed form
                log_phi = doc_m[:, np.newaxis] + dap.log_beta[:, doc.words]
                log_phi -= logsumexp(log_phi, axis=0, keepdims=True)

                # CVI update to m and v
                sum_phi = np.sum(np.exp(log_phi) * doc.counts[np.newaxis, :], axis=1)
                doc_m, doc_vsq, doc_topic_param_1, doc_topic_param_2 = dap.cvi_gamma_update(
                    doc_topic_param_1, doc_topic_param_2, doc_zeta_factor, doc_tau, sum_phi,
                    doc_word_count, doc.time_id)

                # CVI update to tau
                doc_tau, doc_persona_param = dap.cvi_tau_update(doc_tau, doc_persona_param, doc_m,
                                                                doc.time_id, doc.author_id)

                # update zeta in closed form
                doc_zeta_factor = doc_m + 0.5 * doc_vsq

                mean_change = np.mean(abs(doc_m - prev_doc_m))
                if mean_change < dap.em_convergence:
                    break

            if save_ss:
                ss.update(doc, doc_m, doc_tau, log_phi)

            # compute likelihoods
            batch_lhood_d = dap.compute_doc_lhood(doc, doc_tau, doc_m, doc_vsq, log_phi)
            batch_lhood += batch_lhood_d
            words_lhood_d = dap.compute_word_lhood(doc, log_phi)
            words_lhood += words_lhood_d

        del docs
        del dap

        if save_ss:
            result_queue.put([batch_lhood, words_lhood, ss])
            del ss
        else:
            result_queue.put([batch_lhood, words_lhood, None])

