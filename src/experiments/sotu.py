import numpy as np
from src.corpus import Corpus
from src.dapper import DAPPER
import time
import logging
import os
import argparse

def main():
    """
    Example of call main program
    :return:
    """
    parser = argparse.ArgumentParser(description='Run dap model on state of the union data.')
    parser.add_argument('--evaluate_every', type=int,
                        help="If given a test file, number of EM iterations between evaluations of test set. Default of 0 = evaluate after each epoch.")
    parser.add_argument('--max_training_minutes', type=float,
                        help="If given this will stop training once the specified number of minutes have elapsed.")
    parser.add_argument('--normalization', type=str, default="sum",
                        help='Method for normalizing alpha values. Can be sum, none, or softmax.')
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--max_local_iters', type=int, default=30, help="max iterations to run on local parameters.")
    parser.add_argument('--local_convergence', type=float, default=1e-3,
                        help="Convergence threshold for e-step.")
    parser.add_argument('--process_noise', type=float, default=0.2)
    parser.add_argument('--measurement_noise', type=float, default=0.8)
    parser.add_argument('--num_topics', type=int, default=30)
    parser.add_argument('--num_personas', type=int, default=20)
    parser.add_argument('--regularization', type=float, default=0.2,
                        help="How much to penalize similar personas. Recommend [0, 0.5].")
    parser.add_argument('--batch_size', type=int,
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
    path_to_current_file = os.path.abspath(os.path.dirname(__file__))

    disable_log = True
    if disable_log:
        logging.disable(logging.INFO)
    else:
        log_format = '%(asctime)s : %(levelname)s : %(message)s'
        log_dir = os.path.join(path_to_current_file, "../../scripts/log/sotu/")
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

    np.random.seed(2018)

    # initialize model
    dap = DAPPER(num_topics=args.num_topics, num_personas=args.num_personas,
                 process_noise=args.process_noise, measurement_noise=args.measurement_noise,
                 regularization=args.regularization,
                 normalization=args.normalization,
                 max_epochs=args.max_epochs, max_training_minutes=args.max_training_minutes,
                 local_convergence=args.local_convergence, max_local_iters=args.max_local_iters,
                 step_size=args.step_size,
                 batch_size=args.batch_size, queue_size=args.queue_size,
                 learning_offset=args.learning_offset, learning_decay=args.learning_decay,
                 num_workers=args.num_workers)

    data_dir = os.path.join(path_to_current_file, "../../data/sotu/")

    train = Corpus(input_file=data_dir + "sotu_train.txt", vocab_file=data_dir + "sotu_dpp_vocab.txt")
    test = Corpus(input_file=data_dir + "sotu_test.txt",
              vocab_file=data_dir + "sotu_dpp_vocab.txt", author2id=train.author2id)

    train_results, test_results = dap.fit_predict(train_corpus=train, test_corpus=test,
                                                  evaluate_every=args.evaluate_every)
    print(dap)

    # save model output
    results_dir = os.path.join(path_to_current_file, "../../results/sotu/")
    model_sig = "K{}_P{}_bs{}_q{}_lo{}_ld{}_pn{}_mn{}_reg{}_{}_epochs{}_cpu{}_{}.txt".format(
        args.num_topics, args.num_personas,
        args.batch_size, args.queue_size,
        args.learning_offset, int(100 * args.learning_decay),
        int(100 * args.process_noise), int(100 * args.measurement_noise),
        int(100 * args.regularization), args.normalization, dap.total_epochs,
        args.num_workers, time.strftime('%m_%d_%Y_%H%M'))
    dap.save_topics(filename=results_dir + "topics_" + model_sig, topn=10)
    dap.save_author_personas(filename=results_dir + "personas_" + model_sig)
    dap.save_persona_topics(filename=results_dir + "alpha_" + model_sig)
    dap.save_convergnces(filename=results_dir + "train_convergence_" + model_sig, results=train_results)
    dap.save_convergnces(filename=results_dir + "test_convergence_" + model_sig, results=test_results)


if __name__ == "__main__":
    main()
