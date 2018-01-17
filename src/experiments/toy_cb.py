import numpy as np
from src.corpus import Corpus
from src.dapper import DAPPER
import time
import logging
import copy
import os


def main():
    """
    Example of call main program
    :return:
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    np.random.seed(2018)

    path_to_current_file = os.path.abspath(os.path.dirname(__file__))
    data_dir = os.path.join(path_to_current_file, "../../data/cb_toy/")
    train_cb = Corpus(input_file=data_dir + "cb_small.txt", vocab_file=data_dir + "cb_small_vocab.txt")

    num_topics = 10
    num_personas = 4

    max_epochs = 5
    e_iter = 30
    em_convergence = 1e-4

    num_workers = 1

    batch_size = 256
    queue_size = 1
    learning_offset = 5
    learning_decay = 0.6
    step_size = 0.7

    pn = 0.2
    mn = 0.8
    regularization = 0.25
    normalization = "sum"

    print("=====================STOCHASTIC CVI=====================")
    dap = DAPPER(num_topics=num_topics, num_personas=num_personas,
                 process_noise=pn, measurement_noise=mn,
                 regularization=regularization, normalization=normalization,
                 max_epochs=max_epochs, em_convergence=em_convergence,
                 step_size=step_size, local_param_iter=e_iter,
                 batch_size=batch_size, queue_size=queue_size,
                 learning_offset=learning_offset, learning_decay=learning_decay,
                 num_workers=num_workers)


    train_results = dap.fit(corpus=train_cb)
    logger.info(dap)

    model_sig = "K{}_P{}_bs{}_q{}_lo{}_ld{}_pn{}_mn{}_reg{}_{}_epochs{}_cpu{}_{}.txt".format(
        num_topics, num_personas,
        batch_size, queue_size,
        learning_offset, int(100 * learning_decay),
        int(100*pn), int(100*mn),
        int(100*regularization), normalization, dap.total_epochs,
        num_workers, time.strftime('%m_%d_%Y_%H%M'))

    dap.save_topics(filename="../../results/cb_toy/topics_" + model_sig, topn=10, tfidf=True)
    dap.save_author_personas(filename="../../results/cb_toy/personas_" + model_sig)
    dap.save_persona_topics(filename="../../results/cb_toy/alpha_" + model_sig)
    dap.save_convergnces(filename="../../results/cb_toy/convergence_" + model_sig, results=train_results)


if __name__ == "__main__":
    main()
