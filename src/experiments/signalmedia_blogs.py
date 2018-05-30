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
    parser = argparse.ArgumentParser(description='Run dap model on signalmedia data.')
    parser.add_argument('--train_file', type=str, help='Path to training data file.',
                        default="train_signalmedia.dap")
    parser.add_argument('--test_file', type=str, help='Path to testing data file. If None, no prediction is run',
                        default="test_signalmedia.dap")
    parser.add_argument('--vocab_file', type=str, help='Path to vocabulary file.',
                        default="signalmedia.bow.vocab")
    parser.add_argument('--data_dir', type=str, help='directory where all data files reside.',
                        default="../../data/signalmedia/blogs_aligned_3_30/")
    parser.add_argument('--evaluate_every', type=int,
                        help="If given a test file, number of EM iterations between evaluations of test set. Default of 0 = evaluate after each epoch.")
    parser.add_argument('--max_training_minutes', type=float,
                        help="If given this will stop training once the specified number of minutes have elapsed.")
    parser.add_argument('--max_epochs', type=int)
    parser.add_argument('--process_noise', type=float, default=0.2)
    parser.add_argument('--measurement_noise', type=float, default=0.8)
    parser.add_argument('--num_topics', type=int, default=75)
    parser.add_argument('--num_personas', type=int, default=25)
    parser.add_argument('--regularization', type=float, default=0.1,
                        help="How much to penalize similar personas. Recommend [0, 0.5].")
    parser.add_argument('--batch_size', type=int, default=512,
                        help="Batch size. Set to -1 for full gradient updates, else stochastic mini-batches used.")
    parser.add_argument('--num_workers', type=int, default=1)
    args = parser.parse_args()

    path_to_current_file = os.path.abspath(os.path.dirname(__file__))
    np.random.seed(2018)

    disable_log = False
    if disable_log:
        logging.disable(logging.INFO)
    else:
        log_format = '%(asctime)s : %(levelname)s : %(message)s'
        logging.basicConfig(format=log_format, level=logging.INFO)

    # initialize model
    dap = DAPPER(num_topics=args.num_topics, num_personas=args.num_personas,
                 process_noise=args.process_noise, measurement_noise=args.measurement_noise,
                 regularization=args.regularization,
                 normalization="sum",
                 max_epochs=args.max_epochs, max_training_minutes=args.max_training_minutes,
                 batch_size=args.batch_size,
                 step_size=0.7, queue_size=1, learning_offset=10, learning_decay=0.7,
                 num_workers=args.num_workers)

    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    # train = Corpus(input_file=data_dir + "signalmedia.dap", vocab_file=data_dir + "signalmedia.bow.vocab")
    train = Corpus(input_file=args.data_dir + args.train_file, vocab_file=args.data_dir + args.vocab_file)
    test = Corpus(input_file=args.data_dir + args.test_file,
                  vocab_file=args.data_dir + args.vocab_file,
                  author2id=train.author2id)

    train_results, test_results = dap.fit_predict(train_corpus=train, test_corpus=test,
                                                  evaluate_every=args.evaluate_every,
                                                  random_beta=False,
                                                  check_model_lhood=True)
    # train_results = dap.fit(corpus=train, random_beta=False, check_model_lhood=False)
    print(dap)

    # save model output
    results_dir = args.data_dir.replace("/data/", "/results/")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    model_sig = "K{}_P{}_bs{}_pn{}_mn{}_reg{}_epochs{}_cpu{}_{}.txt".format(
        args.num_topics, args.num_personas,
        args.batch_size,
        int(100 * args.process_noise), int(100 * args.measurement_noise),
        int(100 * args.regularization), dap.total_epochs,
        args.num_workers, time.strftime('%m_%d_%Y_%H%M'))
    dap.save_topics(filename=results_dir + "topics_" + model_sig, topn=10)
    dap.save_author_personas(filename=results_dir + "personas_" + model_sig)
    dap.save_persona_topics(filename=results_dir + "alpha_" + model_sig)
    dap.save_convergnces(filename=results_dir + "train_convergence_" + model_sig, results=train_results)
    dap.save_convergnces(filename=results_dir + "test_convergence_" + model_sig, results=test_results)


if __name__ == "__main__":
    main()
