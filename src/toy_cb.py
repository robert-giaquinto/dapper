import numpy as np
from src.corpus import Corpus
from src.dapper import DAPPER
import time
import logging
import copy


def main():
    """
    Example of call main program
    :return:
    """

    logger = logging.getLogger(__name__)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    np.random.seed(2018)

    cb = Corpus()
    id2author = {y: x for x, y in cb.author2id.items()}

    num_topics = 10
    num_personas = 4

    em_iter = 2
    e_iter = 30
    num_workers = 1

    batch_size = 256
    learning_offset = 10
    learning_decay = 0.6
    step_size = 0.7

    pn = 0.2
    mn = 0.8
    penalty = 0.0
    normalization = "sum"

    print("=====================STOCHASTIC CVI=====================")
    dap = DAPPER(num_topics=num_topics, num_personas=num_personas,
              process_noise=pn, measurement_noise=mn, penalty=penalty, normalization=normalization,
              em_max_iter=em_iter,
              step_size=step_size, local_param_iter=e_iter, batch_size=batch_size,
              learning_offset=learning_offset, learning_decay=learning_decay,
              num_workers=num_workers)

    # logger.info("DAP 2 workers, batch size of 128")
    # dap_copy = copy.deepcopy(dap)
    # dap_copy.num_workers = 2
    # dap_copy.batch_size = 128
    # _ = dap_copy.fit(corpus=cb)
    # logger.info(dap_copy)
    # del dap_copy

    train_results = dap.fit(corpus=cb)
    # test_lhood, test_pwll = dap.predict(test_corpus=cb)
    # train_results, test_results = dap.fit_predict(train_corpus=cb, test_corpus=cb, evaluate_every=None)
    logger.info(dap)



    model_sig = "K{}_P{}_bs{}_lo{}_ld{}_pn{}_mn{}_penalty{}_{}_epochs{}_cpu{}_{}.txt".format(
        num_topics, num_personas, batch_size, learning_offset, int(100 * learning_decay),
        int(100*pn), int(100*mn), int(100*penalty), normalization, dap.total_epochs,
        num_workers, time.strftime('%m_%d_%Y_%H%M'))
    # dap.save_topics(filename="../results/cb/topics_" + model_sig, topn=10, tfidf=True)
    # dap.save_author_personas(filename="../results/cb/personas_" + model_sig, id2author=id2author)
    # dap.save_persona_topics(filename="../results/cb/alpha_" + model_sig)
    # dap.save_convergnces(filename="../results/cb/convergence_" + model_sig, results=train_results)


if __name__ == "__main__":
    main()
