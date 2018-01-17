import numpy as np
import logging
import os

from src.doc import Doc

logger = logging.getLogger(__name__)


# TODO should setup test corpus capable of having *fewer* time steps than training set
class Corpus(object):
    """
    Container of corpus information

    For now, just store entire corpus in memory
    TODO: provide iterators over corpus, do one pass over file to collect meta information
    """
    def __init__(self, input_file="cb_small.txt", vocab_file="cb_small_vocab.txt", author2id=None, log=True):
        if log:
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        self.docs = []
        self.total_documents = 0
        self.times = []
        self.num_times = 0
        self.vocab_size = 0
        if author2id is None:
            self.author2id = {}
            self.num_authors = 0
        else:
            self.author2id = author2id
            self.num_authors = len(author2id)
        self.vocab = []
        self.input_file = input_file
        self.num_docs_per_time = []
        self.total_words = 0

        # read and process the input file
        skipped_docs, skipped_times = self.read_file()

        # read and process the vocabulary file
        self.read_vocab(vocab_file)

        logger.info("PROCESSED CORPUS")
        logger.info("Number of time points: " + str(self.num_times))
        logger.info("Number of authors: " + str(self.num_authors))
        logger.info("Number of documents: " + str(self.total_documents))
        logger.info("Total number of words: " + str(self.total_words))
        logger.info("Found ids for " + str(self.vocab_size) + " terms in vocabulary")
        logger.info("Number of documents skipped (no words): " + str(skipped_docs))
        logger.info("Number of times skipped (no documents): " + str(skipped_times))
        logger.info("Number of documents per time-step: {}".format(
            ", ".join([str(i) + ":" + str(n) for i, n in enumerate(self.num_docs_per_time)])))

    def read_file(self):
        """
        Main processing method for reading and parsing information in the file
        """
        skipped_docs = 0
        skipped_times = 0
        doc_id = 0
        for_training = len(self.author2id) == 0 # don't include new authors in a test set
        if for_training:
            logger.info("Building training corpus.")
        else:
            logger.info("Building testing corpus. Will skip any new authors.")
        
        with open(self.input_file, "r") as f:
            self.num_times = int(f.readline().replace('\n', ''))
            self.num_docs_per_time = [0] * self.num_times
            for t in range(self.num_times):
                # catch newlines at the end of the file
                line = f.readline().replace('\n', '')
                if line == '':
                    break

                time_stamp = int(float(line))
                num_docs = int(f.readline().replace('\n', ''))
                if num_docs == 0:
                    skipped_times += 1
                    continue

                self.times.append(time_stamp)
                self.total_documents += num_docs

                for d in range(num_docs):
                    doc = Doc()
                    doc.time = time_stamp
                    doc.time_id = t

                    # read one line = one document
                    fields = f.readline().replace('\n', '').split()

                    # extract author
                    doc.author = fields[0]

                    # convert author to a unique author id
                    if doc.author not in self.author2id:
                        if for_training:
                            self.author2id[doc.author] = self.num_authors
                            self.num_authors += 1
                        else:
                            # skip this doc, don't want to add new authors to a test set
                            skipped_docs += 1
                            continue

                    # save author id by looking up author name
                    doc.author_id = self.author2id[doc.author]

                    # length of document (i.e. number of unique terms
                    doc.num_terms = int(fields[1])

                    # extract words and corresponding counts in this document
                    word_counts = [[int(elt) for elt in wc.split(":")] for wc in fields[2:]]
                    if len(word_counts) == 0:
                        self.total_documents -= 1
                        skipped_docs += 1
                        continue

                    doc.doc_id = doc_id
                    doc_id += 1
                    self.num_docs_per_time[t] += 1

                    doc.words = np.array([w for w, c in word_counts])
                    doc.counts = np.array([c for w, c in word_counts])
                    self.docs.append(doc)
                    self.total_words += np.sum(doc.counts)

                    if max(doc.words) > self.vocab_size:
                        self.vocab_size = max(doc.words) + 1

        return skipped_docs, skipped_times

    def read_vocab(self, vocab_file):
        with open(vocab_file, "r") as f:
            self.vocab = tuple([v.replace("\n", "") for v in f.readlines()])
            self.vocab_size = len(self.vocab)
            logger.info("Number of words in vocabulary: " + str(len(self.vocab)))

    def __iter__(self):
        """
        Iterator returning each doc
        :return:
        """
        for doc in self.docs:
            yield doc

    def __len__(self):
        return len(self.docs)

    def __str__(self):
        return "Corpus with " + str(self.num_times) + " time periods, and " + str(self.total_documents) + " total documents."
