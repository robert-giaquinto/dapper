import numpy as np
import time
import logging
from scipy.special import psi, gammaln
from scipy.misc import logsumexp
from math import ceil
from multiprocessing import Pool, Queue, cpu_count



logger = logging.getLogger(__name__)
SHOW_EVERY = 0


def gen_batches(n, batch_size):
    indices = np.arange(n)
    np.random.shuffle(indices)
    num_splits = int(ceil(1.0 * n / batch_size))
    rval = [arr.tolist() for arr in np.array_split(indices, num_splits)]
    return rval


def softmax(x, axis):
    """
    Softmax for normalizing a matrix along an axis
    Use max substraction approach for numerical stability
    :param x:
    :return:
    """
    e_x = np.exp(x - np.max(x))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def sum_normalize(mat, axis):
    pos_mat = mat - np.min(mat, axis=axis, keepdims=True) + 0.001
    return pos_mat / np.sum(pos_mat, axis=axis, keepdims=True)


def compute_dirichlet_expectation(dirichlet_parameter):
    if len(dirichlet_parameter.shape) == 1:
        return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter))
    return psi(dirichlet_parameter) - psi(np.sum(dirichlet_parameter, 1))[:, np.newaxis]


def matrix2str(mat, num_digits=2):
    rval = ''
    for row in mat:
        s = '{:.' + str(num_digits) + '}'
        # rval += '\t'.join([s.format(round(elt, num_digits)) for elt in row]) + '\n'
        fpad = ['' if round(elt, num_digits) < 0 else ' ' for elt in row]
        bpad = [' ' * (6 + 7 - len(str(np.abs(round(elt, num_digits))))) for elt in row]
        rval += ''.join([f + s.format(round(elt, num_digits)) + b for elt, f, b in zip(row, fpad, bpad)]) + '\n'
    return rval


class DAPPER(object):
    """
    Dynamic Author-Personas Performed Exceedingly Rapidly (DAPPER) Topic Model
    """

    def __init__(self, num_topics, num_personas,
                 process_noise=0.2, measurement_noise=0.8, penalty=0.0, normalization="sum",
                 em_max_iter=10, em_convergence=1e-03, step_size=0.7,
                 local_param_iter=50, batch_size=-1, learning_offset=10, learning_decay=0.7,
                 num_workers=1):
        self.batch_size = batch_size
        self.learning_offset = learning_offset
        self.learning_decay = learning_decay
        self.step_size = step_size

        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.penalty = penalty
        self.normalization = normalization
        self.num_workers = num_workers

        self.num_topics = num_topics
        self.num_personas = num_personas

        self.current_em_iter = 0
        self.total_epochs = 0
        self.em_max_iter = em_max_iter
        self.em_convergence = em_convergence
        self.local_param_iter = local_param_iter
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
        if self.penalty < 0.0 or self.penalty > 0.5:
            raise ValueError("penalty parameter is recommended to be between [0, 0.5]")

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

        self.num_docs = corpus.num_docs
        self.total_words = corpus.total_words
        self.num_authors = corpus.num_authors
        self.max_length = corpus.max_length
        self.vocab = np.array(corpus.vocab)

    def init_latent_vars(self):
        self.regularization = (1.0 - np.tril(np.ones((self.num_personas, self.num_personas)))) *\
                              (self.penalty / self.num_personas)

        # initialize matrices
        self.omega = np.ones(self.num_personas) * (1.0 / self.num_personas)
        self._delta = np.random.gamma(500., 0.1, (self.num_authors, self.num_personas))
        self.kappa = np.random.uniform(0.1, 1.0, (self.num_authors, self.num_personas))
        self.kappa = self.kappa / self.kappa.sum(axis=1, keepdims=True)

        self.eta = 1.0 / self.num_topics
        self._lambda = np.random.gamma(500., 0.1, (self.num_topics, self.vocab_size))
        self.log_beta = np.zeros((self.num_topics, self.vocab_size))

        self.mu0 = np.ones(self.num_topics) * (1.0 / self.num_topics)
        self.alpha = np.zeros((self.num_times, self.num_topics, self.num_personas))
        self.sigma = np.zeros((self.num_times, self.num_topics, self.num_topics))
        self.sigma_inv = np.zeros((self.num_times, self.num_topics, self.num_topics))

        for t in range(self.num_times):
            alpha_init = np.random.uniform(0.01, 5.0, (self.num_topics, self.num_personas))
            # alpha_init = np.random.gamma(100., 0.01, (self.num_topics, self.num_personas))
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

    def cvi_gamma_update(self, doc_nat_param_1, doc_nat_param_2, doc_zeta_factor, doc_tau, sum_phi, doc_word_count, t):
        sigma_inv = self.sigma_inv[t, :, :]
        alpha = self.alpha[t, :, :]

        nat_param_1 = sigma_inv.dot(alpha.dot(doc_tau))
        nat_param_1 += sum_phi

        nat_param_2 = -0.5 * np.diag(sigma_inv)
        nat_param_2 -= (doc_word_count / (2 * np.sum(np.exp(doc_zeta_factor)))) * np.exp(doc_zeta_factor)

        new_doc_nat_param_1 = self.step_size * nat_param_1 + (1 - self.step_size) * doc_nat_param_1
        new_doc_nat_param_2 = self.step_size * nat_param_2 + (1 - self.step_size) * doc_nat_param_2

        new_doc_m = -1 * (new_doc_nat_param_1 / (2 * new_doc_nat_param_2))
        new_doc_vsq = -1 / (2 * new_doc_nat_param_2)

        return new_doc_m, new_doc_vsq, new_doc_nat_param_1, new_doc_nat_param_2

    def cvi_tau_update(self, doc_tau, doc_nat_param, doc_m, E_log_kappa, t):
        alpha = self.alpha[t, :, :]
        sigma_inv = self.sigma_inv[t, :, :]
        sigma = self.sigma[t, :, :]
        gradient = alpha.T.dot(sigma_inv).dot(doc_m - alpha.dot(doc_tau))
        for p in range(self.num_personas):
            S = np.eye(self.num_topics) * (np.diag(alpha[:, p] ** 2) + sigma)
            gradient[p] -= 0.5 * np.trace(sigma_inv.dot(S))

        new_doc_nat_param = self.step_size * (E_log_kappa + gradient) + (1 - self.step_size) * doc_nat_param
        # new_doc_tau = np.exp(new_doc_nat_param) / np.sum(np.exp(new_doc_nat_param))
        new_doc_tau = doc_tau * np.exp(new_doc_nat_param)
        new_doc_tau /= np.sum(new_doc_tau)

        return new_doc_tau, new_doc_nat_param

    """=================================================================================================================
            E-step and M-step of the Variational Inference algorithm
    ================================================================================================================="""

    def m_step_s(self, beta_ss, kappa_ss, alpha_ss, x_ss, xsq_ss, batch_size, num_docs_per_time):
        """
        Stochastic M-step: update the variational parameter for topics using a mini-batch of documents
        """
        rhot = np.power(self.learning_offset + self.current_em_iter, -self.learning_decay)

        # topic maximization
        new_beta = np.zeros((self.num_topics, self.vocab_size))
        for k in range(self.num_topics):
            total = np.sum(beta_ss[k, :])
            for v in range(self.vocab_size):
                if beta_ss[k, v] > 0:
                    new_beta[k, v] = beta_ss[k, v] / total
                else:
                    new_beta[k, v] = 1e-30

        self.log_beta = np.log((1 - rhot) * np.exp(self.log_beta) + rhot * new_beta)
        new_lambda = self.eta + self.num_docs * beta_ss / batch_size
        self._lambda = (1 - rhot) * self._lambda + rhot * new_lambda

        # update the kappa terms
        new_delta = self.omega + (self.num_docs * kappa_ss / batch_size)
        self._delta = (1 - rhot) * self._delta + rhot * new_delta
        new_kappa = np.zeros((self.num_authors, self.num_personas))
        for a in range(self.num_authors):
            total = np.sum(kappa_ss[a, :])
            if total == 0.0:
                new_kappa[a, :] = self.omega
            else:
                new_kappa[a, :] = (self.omega + kappa_ss[a, :]) / (total + 1.0)
        # kappa SVI update
        self.kappa = (1 - rhot) * self.kappa + rhot * new_kappa

        # update omega
        self.omega = self.kappa.sum(axis=0) / self.num_authors
        # logger.info('omega\n' + ' '.join([str(round(elt, 2)) for elt in self.omega]) + "\n")

        # estimate a new noisy estimate of alpha
        new_alpha_hat = self.estimate_alpha(alpha_ss, x_ss, xsq_ss, num_docs_per_time)
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

    def estimate_alpha(self, alpha_ss, x_ss, xsq_ss, num_docs_per_time):
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
                b = self.mu0[:, np.newaxis] + alpha_ss[t, :, :] - x_ss[t, :]
            else:
                b = alpha_hat[t-1, :, :] + alpha_ss[t, :, :] - x_ss[t, :]

            denom = xsq_ss[t, :] + 1.0
            if self.penalty > 0:
                A = np.ones((self.num_personas, self.num_personas))
                A *= (self.penalty / self.num_personas) * num_docs_per_time[t]
                A[np.diag_indices_from(A)] = denom
                try:
                    alpha_hat[t, :, :] = np.linalg.solve(A, b.T).T
                except np.linalg.linalg.LinAlgError:
                    print("alpha hat calc failure")
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
        E_log_kappa = psi(self._delta) - psi(self._delta.sum(axis=1, keepdims=True))
        rval = self.num_authors * (gammaln(self.omega.sum()) - gammaln(self.omega).sum())

        # note: cancellation of omega between model and entropy term
        rval += np.sum((self._delta - 1.0) * E_log_kappa)

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

    def compute_doc_lhood(self, doc, doc_tau, doc_m, doc_vsq, log_phi, E_log_kappa):
        sigma = self.sigma[doc.time_id, :, :]
        sigma_inv = self.sigma_inv[doc.time_id, :, :]
        alpha = self.alpha[doc.time_id, :, :]

        # term1: sum_P [ tau * (digamma(delta) - digamma(sum(delta))) ]
        lhood = E_log_kappa[doc.author_id, :].dot(doc_tau)

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

        # term 9: phi log phi
        lhood -= np.sum(np.exp(log_phi) * log_phi * doc.counts)
        return lhood

    def compute_word_lhood(self, doc, log_phi):
        lhood = np.sum(np.exp(log_phi + np.log(doc.counts)) * self.log_beta[:, doc.words])
        return lhood

    def em_step_s(self, docs, total_docs):
        """
        Performs stochastic EM-update for one iteration using a mini-batch of documents
        and compute the training log-likelihood
        """
        batch_size = len(docs)
        self.current_em_iter += 1

        # e-step
        clock_e_step = time.process_time()
        if self.num_workers > 1:
            # run e-step in parallel
            batch_lhood, words_lhood, beta_ss, kappa_ss, x_ss, xsq_ss, alpha_ss = self.e_step_parallel(docs=docs)
        else:
            batch_lhood, words_lhood, beta_ss, kappa_ss, x_ss, xsq_ss, alpha_ss = self.e_step(docs=docs)
        clock_e_step = time.process_time() - clock_e_step
        doc_lhood = batch_lhood * total_docs / batch_size
        per_word_lhood = words_lhood / np.sum([np.sum(d.counts) for d in docs])

        # count docs per timestep in this batch for adjusting the penalty
        num_docs_per_time = [0] * self.num_times
        for d in docs:
            time_id = d.time_id
            num_docs_per_time[time_id] += 1

        # m-step
        clock_m_step = time.process_time()
        self.m_step_s(beta_ss=beta_ss, kappa_ss=kappa_ss, alpha_ss=alpha_ss,
                      x_ss=x_ss, xsq_ss=xsq_ss, batch_size=batch_size, num_docs_per_time=num_docs_per_time)
        clock_m_step = time.process_time() - clock_m_step

        topic_lhood = self.compute_topic_lhood()
        persona_lhood = self.compute_author_lhood()
        alpha_lhood = self.compute_persona_lhood()

        lhood = doc_lhood + topic_lhood + persona_lhood + alpha_lhood
        total_time = clock_e_step + clock_m_step
        return lhood, total_time

    def e_step_parallel(self, docs):
        batch_size = len(docs)
        E_log_kappa = compute_dirichlet_expectation(self._delta)

        # initialize sufficient statistics
        beta_ss = np.zeros((self.num_topics, self.vocab_size))
        kappa_ss = np.zeros((self.num_authors, self.num_personas))
        x_ss = np.zeros((self.num_times, self.num_personas))
        xsq_ss = np.zeros((self.num_times, self.num_personas))
        alpha_ss = np.zeros((self.num_times, self.num_topics, self.num_personas))

        # set up pool of workers and arguments passed to each worker
        job_queue = Queue(maxsize=self.num_workers)
        result_queue = Queue()
        pool = Pool(self.num_workers, _doc_e_step_worker, (job_queue, result_queue,))

        queue_size, reallen = [0], 0
        batch_lhood, words_lhood = [0.0], [0.0]
        partial_ss = []

        def process_result_queue():
            """
            clear result queue, merge intermediate SS
            :return:
            """
            while not result_queue.empty():
                # collect summary statistics and merge all sufficient statistics
                bl, wl, ss = result_queue.get()
                batch_lhood[0] += bl
                words_lhood[0] += wl
                partial_ss.append(ss)
                queue_size[0] -= 1

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

        # loop through chunks of documents placing chunks on the queue
        for chunk_id, chunk_size in enumerate(docs_per_worker):
            doc_mini_batch = docs[reallen:(reallen + chunk_size)]
            reallen += len(doc_mini_batch)  # track how documents seen
            chunk_put = False
            while not chunk_put:
                try:
                    args = (self, doc_mini_batch, E_log_kappa)
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

        # close out pool
        pool.terminate()

        # accumulate partial ss from each worker
        for ss in partial_ss:
            partial_beta_ss, partial_kappa_ss, partial_x_ss, partial_xsq_ss, partial_alpha_ss = ss
            beta_ss += partial_beta_ss
            kappa_ss += partial_kappa_ss
            x_ss += partial_x_ss
            xsq_ss += partial_xsq_ss
            alpha_ss += partial_alpha_ss

        return batch_lhood[0], words_lhood[0], beta_ss, kappa_ss, x_ss, xsq_ss, alpha_ss

    def e_step(self, docs):
        """
        E-step: update the variational parameters for topic proportions and topic assignments.
        """
        E_log_kappa = compute_dirichlet_expectation(self._delta)

        # initialize sufficient statistics
        beta_ss = np.zeros((self.num_topics, self.vocab_size))
        kappa_ss = np.zeros((self.num_authors, self.num_personas))
        x_ss = np.zeros((self.num_times, self.num_personas))
        xsq_ss = np.zeros((self.num_times, self.num_personas))
        alpha_ss = np.zeros((self.num_times, self.num_topics, self.num_personas))

        # iterate over all documents
        batch_lhood = 0
        words_lhood = 0
        for doc in docs:
            # initialize gamma for this document
            doc_m = np.ones(self.num_topics)  * (1.0 / self.num_topics)
            doc_vsq = np.ones(self.num_topics)
            doc_nat_param_1 = np.zeros(self.num_topics)
            doc_nat_param_2 = np.zeros(self.num_topics)
            doc_nat_param = np.zeros(self.num_personas)
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
                doc_m, doc_vsq, doc_nat_param_1, doc_nat_param_2 = self.cvi_gamma_update(
                    doc_nat_param_1, doc_nat_param_2, doc_zeta_factor, doc_tau, sum_phi, doc_word_count, doc.time_id)

                # CVI update to tau
                doc_tau, doc_nat_param = self.cvi_tau_update(doc_tau, doc_nat_param, doc_m,
                                                             E_log_kappa[doc.author_id, :], doc.time_id)

                # update zeta in closed form
                doc_zeta_factor = doc_m + 0.5 * doc_vsq

                mean_change = np.mean(abs(doc_m - prev_doc_m))
                if mean_change < 0.0001:
                    break

            # collect sufficient statistics
            beta_ss[:, doc.words] += np.exp(log_phi + np.log(doc.counts))
            kappa_ss[doc.author_id, :] += doc_tau
            x_ss[doc.time_id, :] += doc_tau
            xsq_ss[doc.time_id, :] += doc_tau ** 2
            for p in range(self.num_personas):
                alpha_ss[doc.time_id, :, p] += doc_m * doc_tau[p]

            # compute likelihoods
            batch_lhood_d = self.compute_doc_lhood(doc, doc_tau, doc_m, doc_vsq, log_phi, E_log_kappa)
            batch_lhood += batch_lhood_d
            words_lhood_d = self.compute_word_lhood(doc, log_phi)
            words_lhood += words_lhood_d

            if SHOW_EVERY > 0 and doc.doc_id % SHOW_EVERY == 0:
                logger.info("Variational parameters for document: {}, converged in {} steps".format(doc.doc_id, iters))
                logger.info("Per-word likelihood for doc[{}]: ({} + {}) / {} = {:.2f}".format(
                    doc.doc_id, words_lhood_d, batch_lhood_d, doc_word_count, (words_lhood_d + batch_lhood_d) / doc_word_count))
                # logger.info("new zeta: " + ' '.join([str(round(g, 3)) for g in doc_zeta_factor]))
                logger.info("new doc_m[{}]: ".format(doc.doc_id) + ' '.join([str(round(g, 3)) for g in doc_m]))
            #     logger.info("new vhat: " + ' '.join([str(round(g, 3)) for g in doc_vsq]))
                logger.info("new tau: " + ' '.join([str(round(g, 3)) for g in doc_tau]) + "\n")

        return batch_lhood, words_lhood, beta_ss, kappa_ss, x_ss, xsq_ss, alpha_ss

    """=================================================================================================================
        Training and testing
    ================================================================================================================="""

    def fit(self, corpus):
        """
        Performs EM-update until reaching target average change in the log-likelihood
        """
        # init from corpus
        self.init_from_corpus(corpus)
        self.init_latent_vars()
        self.init_beta_from_corpus(corpus=corpus)
        id2author = {y: x for x, y in corpus.author2id.items()}

        prev_lhood, lhood = np.finfo(np.float32).min, np.finfo(np.float32).min
        convergences = []
        train_results = []

        if self.batch_size <= 0:
            batch_size = corpus.num_docs
        else:
            batch_size = self.batch_size

        total_time = time.process_time()
        while self.total_epochs < self.em_max_iter:
            for batch_id, doc_ids in enumerate(gen_batches(corpus.num_docs, batch_size)):
                batch = [corpus.docs[d] for d in doc_ids]
                lhood, batch_time = self.em_step_s(docs=batch, total_docs=corpus.num_docs)
                pwll = lhood / corpus.total_words
                convergence = np.abs((lhood - prev_lhood) / prev_lhood)
                train_results.append([self.total_epochs, batch_id, lhood, pwll, convergence, batch_time])

                # report current stats
                log_str = """epoch: {}, batch: {}, log-lhood: {:.2f}, per-word log-lhood: {:.2f}, convergence: {:.3f}, time: {:.3f}"""
                logger.info(log_str.format(self.total_epochs, batch_id, lhood, pwll, convergence, batch_time))
                # if batch_id % 10 == 0:
                #     for p in range(self.num_personas):
                #         logger.info('alpha[p={}]\n'.format(p) + matrix2str(self.alpha[:, :, p], 3))
                #     self.print_author_personas(id2author=id2author)
                #     self.print_topics(topn=8)

                # save list of convergences
                convergences.append(convergence)
                if len(convergences) >= 1:
                    av_conv = np.mean(np.asarray(convergences[-1:]))
                else:
                    av_conv = np.mean(np.asarray(convergences))

                # stop if converged
                if av_conv < self.em_convergence:
                    logger.info('Converged after {} epochs, log-lhood: {:.2f}, per-word log-lhood: {:.2f}'.format(
                        self.total_epochs, lhood, pwll))
                    break

                prev_lhood = lhood
            self.total_epochs += 1

        # print last stats
        total_time = time.process_time() - total_time
        # for p in range(self.num_personas):
        #     logger.info('alpha[p={}]\n'.format(p) + matrix2str(self.alpha[:, :, p], 3))
        # self.print_author_personas(id2author=id2author)
        # self.print_topics(topn=8)
        self.print_convergence(train_results, show_batches=True)
        log_str = """Finished after {} EM iterations ({} epochs)
                            train lhood: {:.1f}, train per-word log-lhood: {:.1f}
                            total time (sec): {:.1f}, avg docs/hr {:.1f}
                            """
        logger.info(log_str.format(self.current_em_iter, self.total_epochs,
                                   lhood, pwll,
                                   total_time, (self.total_epochs * corpus.num_docs) / (total_time / 60.0 / 60.0)))
        return train_results

    def init_beta_from_corpus(self, corpus, num_docs_init=500):
        """
        initialize model for training using the corpus
        :param corpus:
        :param num_docs_init:
        :return:
        """
        logger.info("Initializing beta from {} random documents in the corpus for each topics".format(num_docs_init))
        for k in range(self.num_topics):
            doc_ids = np.sort(np.random.randint(0, corpus.num_docs, num_docs_init))
            logger.debug("Initializing topic {} from docs: {}".format(k, ' '.join([str(d) for d in doc_ids])))

            for i in doc_ids:
                doc = corpus.docs[i]
                for n in range(doc.num_terms):
                    v = doc.words[n]
                    self.log_beta[k, v] += doc.counts[n]

        # smooth
        self.log_beta += np.random.uniform(0.01, 0.5, (self.num_topics, self.vocab_size))

        # normalize
        self.log_beta = np.log(self.log_beta / self.log_beta.sum(axis=1, keepdims=True))

    def predict(self, test_corpus):
        """
        Performs E-step on test corpus using stored topics obtained by training
        Computes the average heldout log-likelihood
        """
        test_lhood, test_words_lhood, _, _, _, _, _ = self.e_step(docs=test_corpus.docs)
        test_pwll = test_words_lhood / test_corpus.total_words
        logger.info('Test variational lhood: {:.2f}, test per-word log-lhood: {:.2f}'.format(
            test_lhood, test_pwll))
        return test_lhood, test_pwll

    def fit_predict(self, train_corpus, test_corpus, evaluate_every=None):
        """
        Computes the heldout-log likelihood on the test corpus after "evaluate_every" iterations
        (mini-batches) of training.
        """
        # init from corpus
        self.init_from_corpus(train_corpus)
        self.init_latent_vars()
        self.init_beta_from_corpus(corpus=train_corpus)
        id2author = {y: x for x, y in train_corpus.author2id.items()}

        prev_train_lhood, train_lhood = np.finfo(np.float32).min, np.finfo(np.float32).min
        convergences = []
        train_results = []
        test_results = []

        if self.batch_size <= 0:
            batch_size = train_corpus.num_docs
        else:
            batch_size = self.batch_size
        batches = gen_batches(train_corpus.num_docs, batch_size)
        num_batches = len(batches)

        total_time = time.process_time()
        while self.total_epochs < self.em_max_iter:
            epoch_time = time.process_time()
            for batch_id, doc_ids in enumerate(batches):
                batch = [train_corpus.docs[d] for d in doc_ids]
                train_lhood, batch_time = self.em_step_s(docs=batch, total_docs=train_corpus.num_docs)
                train_pwll = train_lhood / train_corpus.total_words
                convergence = np.abs((train_lhood - prev_train_lhood) / prev_train_lhood)
                train_results.append([self.total_epochs, batch_id, train_lhood, train_pwll, convergence, batch_time])

                # report current stats
                log_str = """epoch: {}, batch: {}, log-lhood: {:.2f}, per-word log-lhood: {:.2f}, convergence: {:.3f}, time: {:.3f}"""
                logger.info(log_str.format(self.total_epochs, batch_id, train_lhood, train_pwll, convergence, batch_time))
                if batch_id % 10 == 0:
                    for p in range(self.num_personas):
                        logger.info('alpha[p={}]\n'.format(p) + matrix2str(self.alpha[:, :, p], 3))
                    self.print_author_personas(id2author=id2author)
                    self.print_topics(topn=8)

                # test set evaluation:
                if evaluate_every is not None and (batch_id + 1) % evaluate_every == 0 and (batch_id + 1) != num_batches:
                    test_lhood, test_pwll = self.predict(test_corpus=test_corpus)
                    test_results.append([self.total_epochs, batch_id, test_lhood, test_pwll, convergence, batch_time])

                # save list of convergences
                convergences.append(convergence)
                if len(convergences) >= 1:
                    av_conv = np.mean(np.asarray(convergences[-1:]))
                else:
                    av_conv = np.mean(np.asarray(convergences))

                # stop if converged
                if av_conv < self.em_convergence:
                    logger.info('Converged after {} epochs, final log-lhood: {:.2f}, final per-word log-lhood: {:.2f}'.format(
                        self.total_epochs, train_lhood, train_pwll))
                    break

                prev_train_lhood = train_lhood

            # always evaluate the log-likelihood at the end of the epoch
            self.total_epochs += 1
            epoch_time = time.process_time() - epoch_time
            test_lhood, test_pwll = self.predict(test_corpus=test_corpus)
            log_str = """Epoch {}, train lhood: {:.1f}, train per-word log-lhood: {:.1f}
                test variational lhood: {:.2f} test per-word log-lhood: {:.2f}
                convergence: {:.3f}, epoch time: {:.3f}, docs/hr {:.1f}
                """
            logger.info(log_str.format(self.total_epochs, train_lhood, train_pwll,
                                       test_lhood, test_pwll,
                                       convergence, epoch_time, (60.0 ** 2) * train_corpus.num_docs / epoch_time))

        # print last stats
        total_time = time.process_time() - total_time
        for p in range(self.num_personas):
            logger.info('alpha[p={}]\n'.format(p) + matrix2str(self.alpha[:, :, p], 3))
        self.print_author_personas(id2author=id2author)
        self.print_topics(topn=8)
        log_str = """Finished after {} EM iterations ({} epochs)
                    train lhood: {:.1f}, train per-word log-lhood: {:.1f}
                    test variational lhood: {:.2f}, test per-word log-lhood: {:.2f}
                    total time (sec): {:.1f}, avg docs/hr {:.1f}
                    """
        logger.info(log_str.format(self.current_em_iter, self.total_epochs,
                                   train_lhood, train_pwll,
                                   test_lhood, test_pwll,
                                   total_time, (self.total_epochs + 1) * (60.0 ** 2) * train_corpus.num_docs / total_time))
        self.print_convergence(train_results, show_batches=False)
        return train_results, test_results

    def print_convergence(self, results, show_batches=False):
        logger.info("EM Iteration\tMini-batch Id\tLog Likelihood\tPer-Word Log Likelihood\tConvergence\tSeconds per Batch")
        for stats in results:
            em_iter, batch_id, lhood, perplexity, convergence, batch_time = stats
            if batch_id == 0 or show_batches:
                logger.info("{}\t\t{}\t\t{:.1f}\t{:.1f}\t\t\t{:.4f}\t\t{:.2f}".format(
                    em_iter, batch_id, lhood, perplexity, convergence, batch_time))

    def print_author_personas(self, max_show=10, id2author=None):
        if id2author is not None:
            max_key_len = max([len(k) for k in id2author.values()])
        else:
            max_key_len = 10
        logger.info("Kappa:")
        spaces = ' ' * (max_key_len - 6)
        logger.info("Author{} \t{}".format(spaces, '\t'.join(['p' + str(i) for i in range(self.num_personas)])))
        logger.info('-' * (max_key_len + 10 * self.num_personas))
        for author_id in range(self.num_authors):
            if id2author is not None:
                author = id2author[author_id]
            else:
                author = str(author_id)
            pad = ' ' * (max_key_len - len(author))
            if author_id == self.num_authors - 1:
                logger.info(author + pad + "\t" + "\t".join([str(round(k, 2)) for k in self.kappa[author_id, :]]) + '\n')
            else:
                logger.info(author + pad + "\t" + "\t".join([str(round(k, 2)) for k in self.kappa[author_id, :]]))

            if author_id > max_show:
                logger.info("...\n")
                break

    def print_topics(self, topn=10, tfidf=True):
        beta = self.get_topics(topn, tfidf)
        for k in range(self.num_topics):
            topic_ = ' + '.join(['%.3f*"%s"' % (p, w) for w, p in beta[k]])
            logger.info("topic #%i: %s", k, topic_)

    def get_topics(self, topn=10, tfidf=True):
        E_log_beta = compute_dirichlet_expectation(self._lambda)
        beta = np.exp(E_log_beta)
        row_sums = beta.sum(axis=1, keepdims=True)
        beta = beta / row_sums

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

    def save_author_personas(self, filename, id2author=None):
        with open(filename, "w") as f:
            f.write("author\t" + "\t".join(["persona" + str(i) for i in range(self.num_personas)]) + "\n")
            for author_id in range(self.num_authors):
                if id2author is not None:
                    author = id2author[author_id]
                else:
                    author = str(author_id)
                f.write(author + "\t" + "\t".join([str(round(k, 7)) for k in self.kappa[author_id]]) + "\n")

    def save_persona_topics(self, filename):
        with open(filename, "w") as f:
            f.write("time_id\ttime\tpersona\t" + "\t".join(["topic" + str(i) for i in range(self.num_topics)]) + "\n")
            for t in range(self.num_times):
                for p in range(self.num_personas):
                    time_val = self.times[t]
                    f.write("{}\t{}\t{}\t{}\n".format(t, time_val, p, '\t'.join([str(k) for k in self.alpha[t, :, p]])))

    def save_convergnces(self, filename, results):
        with open(filename, 'w') as f:
            f.write("em_iter\tbatch_iter\tlog_likelihood\tpwll\tconvergence\ttime\n")
            for stats in results:
                f.write('\t'.join([str(stat) for stat in stats]) + '\n')

    def __str__(self):
        rval = """\nDAP Model:
            training iterations: {} ({} epochs)
            trained on corpus with {} time points, {} authors, {} documents, {} total words
            using {} processors
            """.format(self.current_em_iter, self.total_epochs, self.num_times,
                       self.num_authors, self.num_docs, self.total_words, self.num_workers)
        rval += """\nModel Settings:
            regularization {:.2f}
            number of topics: {}
            number of personas: {}
            measurement noise: {}
            process noise: {}
            batch size: {}
            learning offset: {}
            learning decay: {:.2f}
            step size: {:.2f}
            normalization method: {}
            max em iterations: {}
            em convergence: {:.2f}
            local parameter max iterations: {}
            """.format(self.penalty, self.num_topics, self.num_personas,
                       self.measurement_noise, self.process_noise,
                       self.batch_size, self.learning_offset, self.learning_decay, self.step_size,
                       self.normalization, self.em_max_iter, self.em_convergence, self.local_param_iter)
        return rval


def _doc_e_step_worker(input_queue, result_queue):
    while True:
        dap, docs, E_log_kappa = input_queue.get()

        # initialize sufficient statistics
        beta_ss = np.zeros((dap.num_topics, dap.vocab_size))
        kappa_ss = np.zeros((dap.num_authors, dap.num_personas))
        x_ss = np.zeros((dap.num_times, dap.num_personas))
        xsq_ss = np.zeros((dap.num_times, dap.num_personas))
        alpha_ss = np.zeros((dap.num_times, dap.num_topics, dap.num_personas))

        # iterate over all documents
        batch_lhood = 0
        words_lhood = 0
        for doc in docs:
            # initialize gamma for this document
            doc_m = np.ones(dap.num_topics)  * (1.0 / dap.num_topics)
            doc_vsq = np.ones(dap.num_topics)
            doc_nat_param_1 = np.zeros(dap.num_topics)
            doc_nat_param_2 = np.zeros(dap.num_topics)
            doc_nat_param = np.zeros(dap.num_personas)
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
                doc_m, doc_vsq, doc_nat_param_1, doc_nat_param_2 = dap.cvi_gamma_update(
                    doc_nat_param_1, doc_nat_param_2, doc_zeta_factor, doc_tau, sum_phi, doc_word_count, doc.time_id)

                # CVI update to tau
                doc_tau, doc_nat_param = dap.cvi_tau_update(doc_tau, doc_nat_param, doc_m,
                                                             E_log_kappa[doc.author_id, :], doc.time_id)

                # update zeta in closed form
                doc_zeta_factor = doc_m + 0.5 * doc_vsq

                mean_change = np.mean(abs(doc_m - prev_doc_m))
                if mean_change < 0.0001:
                    break

            # collect sufficient statistics
            beta_ss[:, doc.words] += np.exp(log_phi + np.log(doc.counts))
            kappa_ss[doc.author_id, :] += doc_tau
            x_ss[doc.time_id, :] += doc_tau
            xsq_ss[doc.time_id, :] += doc_tau ** 2
            for p in range(dap.num_personas):
                alpha_ss[doc.time_id, :, p] += doc_m * doc_tau[p]

            # compute likelihoods
            batch_lhood_d = dap.compute_doc_lhood(doc, doc_tau, doc_m, doc_vsq, log_phi, E_log_kappa)
            batch_lhood += batch_lhood_d
            words_lhood_d = dap.compute_word_lhood(doc, log_phi)
            words_lhood += words_lhood_d

        del docs
        del dap
        partial_ss = [beta_ss, kappa_ss, x_ss, xsq_ss, alpha_ss]
        result_queue.put([batch_lhood, words_lhood, partial_ss])
        del partial_ss
        del beta_ss
        del kappa_ss
        del x_ss
        del xsq_ss
        del alpha_ss

