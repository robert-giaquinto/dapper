import numpy as np
import time
import logging
import argparse
import os

from src.corpus import Corpus
from src.dapper import DAPPER


def main():
    """
    Example of call main program
    :return:
    """
    parser = argparse.ArgumentParser(description='Run dap model.')
    parser.add_argument('--train_file', type=str, help='Path to training data file.',
                        default="train_authors_apt_cv0.dat")
    parser.add_argument('--test_file', type=str, help='Path to testing data file. If None, no prediction is run')
    parser.add_argument('--vocab_file', type=str, help='Path to vocabulary file.',
                        default="train_authors_apt_cv0_vocab.dat")
    parser.add_argument('--evaluate_every', type=int,
                        help="If given a test file, number of EM iterations between evaluations of test set. Default of 0 = evaluate after each epoch.")
    parser.add_argument('--max_training_minutes', type=float,
                        help="If given this will stop training once the specified number of minutes have elapsed.")
    parser.add_argument('--normalization', type=str, default="sum",
                        help='Method for normalizing alpha values. Can be sum, none, or softmax.')
    parser.add_argument('--em_max_iter', type=int, default=10)
    parser.add_argument('--local_param_iter', type=int, default=30, help="max iterations to run on local parameters.")
    parser.add_argument('--em_convergence', type=float, default=1e-3,
                        help="Convergence threshold for e-step.")
    parser.add_argument('--process_noise', type=float, default=0.2)
    parser.add_argument('--measurement_noise', type=float, default=0.8)
    parser.add_argument('--num_topics', type=int, default=10)
    parser.add_argument('--num_personas', type=int, default=4)
    parser.add_argument('--regularization', type=float, default=0.2,
                        help="How much to penalize similar personas. Recommend [0, 0.5].")
    parser.add_argument('--batch_size', type=int, default=-1,
                        help="Batch size. Set to -1 for full gradient updates, else stochastic minibatches used.")
    parser.add_argument('--learning_offset', type=int, default=10,
                        help="Learning offset used to control rate of convergence of gradient updates.")
    parser.add_argument('--learning_decay', type=float, default=0.7,
                        help="Learning decay rate used to control rate of convergence of gradient updates.")
    parser.add_argument('--step_size', type=float, default=0.7,
                        help="Learning rate for CVI updates.")
    parser.add_argument('--queue_size', type=int, default=1,
                        help="Number of previous gradient to average over for smoothed gradient updates. Default 1 = no smoothing.")
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--print_log', dest="log", action="store_false",
                        help='Add this flag to print log to console instead of saving it to a file.')
    parser.set_defaults(log=True)
    parser.set_defaults(corpus_in_memory=False)
    args = parser.parse_args()
    np.random.seed(2018)

    logger = logging.getLogger(__name__)
    log_format = '%(asctime)s : %(levelname)s : %(message)s'
    path_to_current_file = os.path.abspath(os.path.dirname(__file__))
    log_dir = os.path.join(path_to_current_file, "../../scripts/log/cb_cv0/")
    if args.log:
        filename = log_dir + time.strftime('%m_%d_%Y_%H%M') +\
                   '_K{}_P{}_bs{}_q{}_lo{}_ld{}_pn{}_mn{}_reg{}_{}_cpu{}.log'.format(
                       args.num_topics, args.num_personas,
                       args.batch_size, args.queue_size,
                       args.learning_offset, int(100 * args.learning_decay),
                       int(100 * args.process_noise), int(100 * args.measurement_noise),
                       int(100 * args.regularization), args.normalization, args.num_workers)
        logging.basicConfig(filename=filename, format=log_format, level=logging.INFO)
    else:
        logging.basicConfig(format=log_format, level=logging.INFO)

    # initialize model
    dap = DAPPER(num_topics=args.num_topics, num_personas=args.num_personas,
                 process_noise=args.process_noise, measurement_noise=args.measurement_noise,
                 regularization=args.regularization, normalization=args.normalization,
                 em_max_iter=args.em_max_iter, em_convergence=args.em_convergence,
                 step_size=args.step_size, local_param_iter=args.local_param_iter,
                 batch_size=args.batch_size, queue_size=args.queue_size,
                 learning_offset=args.learning_offset, learning_decay=args.learning_decay,
                 num_workers=args.num_workers)

    # load training corpus
    data_dir = os.path.join(path_to_current_file, "../../data/cb_cv0/")
    train_cb = Corpus(input_file=data_dir + args.train_file, vocab_file=data_dir + args.vocab_file)

    # train (predict) model
    if args.test_file is None:
        logger.info("Fitting model to {}.".format(data_dir + args.train_file))
        train_results = dap.fit(corpus=train_cb, max_training_minutes=args.max_training_minutes)
    else:
        logger.info("Fitting model to {} and evaluating on {} every {} EM iterations.".format(
            data_dir + args.train_file, data_dir + args.test_file, args.evaluate_every))
        test_cb = Corpus(input_file=data_dir + args.test_file, vocab_file=data_dir + args.vocab_file,
                         author2id=train_cb.author2id)
        train_results, test_results = dap.fit_predict(train_corpus=train_cb, test_corpus=test_cb,
                                                      evaluate_every=args.evaluate_every,
                                                      max_training_minutes=args.max_training_minutes)

    logger.info(dap)

    # save model output
    results_dir = os.path.join(path_to_current_file, "../../results/cb_cv0/")
    model_sig = "K{}_P{}_bs{}_q{}_lo{}_ld{}_pn{}_mn{}_reg{}_{}_epochs{}_cpu{}_{}.txt".format(
        args.num_topics, args.num_personas,
        args.batch_size, args.queue_size,
        args.learning_offset, int(100 * args.learning_decay),
        int(100*args.process_noise), int(100*args.measurement_noise),
        int(100*args.regularization), args.normalization, dap.total_epochs,
        args.num_workers, time.strftime('%m_%d_%Y_%H%M'))
    dap.save_topics(filename=results_dir + "topics_" + model_sig, topn=10)
    dap.save_author_personas(filename=results_dir + "personas_" + model_sig)
    dap.save_persona_topics(filename=results_dir + "alpha_" + model_sig)
    dap.save_convergnces(filename=results_dir + "train_convergence_" + model_sig, results=train_results)

    if args.test_file is not None:
        dap.save_convergnces(filename=results_dir + "test_convergence_" + model_sig, results=test_results)

if __name__ == "__main__":
    main()
