import re
from gensim import corpora
import datetime
import subprocess
import logging
import os
import numpy as np
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# define variables, lists needed to preprocess each speech
lemmatizer = WordNetLemmatizer()
base_stopwords = stopwords.words("english")
custom_stopwords = ['take', 'say', 'back', 'see', 'come', 'still', 'make', 'thing', 'get',
                    'go', 'well', 'good', 'think', 'much', 'like', 'through', 'up', 'down', 'over', 'under',
                    'got', 'get', 'til', 'also', 'would', 'could', 'should', 'really',
                    'didnt', 'cant', 'thats', 'doesnt', 'didnt', 'wont', 'wasnt', 'hows',
                    'hadnt', 'hasnt', 'willnt', 'isnt', 'arent', 'werent', 'havent',
                    'wouldnt', 'couldnt', 'shouldnt', 'shouldve', 'couldve', 'wouldve', 'upon', 'may', 'subject',
                    'theres', 'whats', 'whens', 'whos', 'wheres'] + list('abcdefghjklmnopqrstuvwxyz')
stopword_set = set(base_stopwords + custom_stopwords)
split_dash2 = re.compile(r"([0-9])([\-/])([a-z])", re.IGNORECASE)
split_dash3 = re.compile(r"([a-z])([\-/])([0-9])", re.IGNORECASE)
iam = re.compile(r"\bi'm\b", re.IGNORECASE)
ive = re.compile(r"\bive\b", re.IGNORECASE)
hes = re.compile(r"\bhes\b", re.IGNORECASE)
shes = re.compile(r"\bshes\b", re.IGNORECASE)
weve = re.compile(r"\bweve\b", re.IGNORECASE)
youve = re.compile(r"\byouve\b", re.IGNORECASE)
willnot = re.compile(r"\bwon't\b", re.IGNORECASE)
cannot = re.compile(r"\bcan't\b", re.IGNORECASE)
itis = re.compile(r"\bit's\b", re.IGNORECASE)
letus = re.compile(r"\blet's\b", re.IGNORECASE)
heis = re.compile(r"\bhe's\b", re.IGNORECASE)
sheis = re.compile(r"\bshe's\b", re.IGNORECASE)
howis = re.compile(r"\bhow's\b", re.IGNORECASE)
thatis = re.compile(r"\bthat's\b", re.IGNORECASE)
thereis = re.compile(r"\bthere's\b", re.IGNORECASE)
whatis = re.compile(r"\bwhat's\b", re.IGNORECASE)
whereis = re.compile(r"\bwhere's\b", re.IGNORECASE)
whenis = re.compile(r"\bbwhen's\b", re.IGNORECASE)
whois = re.compile(r"\bwho's\b", re.IGNORECASE)
whyis = re.compile(r"\bwhy's\b", re.IGNORECASE)
youall = re.compile(r"y'all|ya'll", re.IGNORECASE)
youare = re.compile(r"\byou're\b", re.IGNORECASE)
would = re.compile(r"'d\b", re.IGNORECASE)
has = re.compile(r"'s\b", re.IGNORECASE)
nt = re.compile(r"n't\b", re.IGNORECASE)
will = re.compile(r"'ll\b", re.IGNORECASE)
have = re.compile(r"'ve\b", re.IGNORECASE)
s_apostrophe = re.compile(r"s'\b", re.IGNORECASE)
years = re.compile(r"\b(16|17|18|19|20)[0-9]{2}\b")
dollars = re.compile(r"\$[1-9][0-9,\.]*")
percents = re.compile(r"[1-9][0-9\.]*\%")
percent = re.compile(" per cent ", re.IGNORECASE)
times = re.compile(r"(2[0-3]|[01]?[0-9]):([0-5]?[0-9])")
dates = re.compile(r"\b1?[0-9]?[-/]1?[0-9]?[-/](18|19|20)?[0-9]{2}\b")
punct = re.compile(r"[^a-zA-Z_ ]")
united_states_of_america = re.compile(r"\bunited states of america\b", re.IGNORECASE)
united_states = re.compile(r"\bunited states\b", re.IGNORECASE)
usp = re.compile(r"\bU.S.\b")
usap = re.compile(r"\bU.S.A\b")
us = re.compile(r"\bUS\b")
usa = re.compile(r"\bUSA\b")

def get_wordnet_pos(treebank_tag):
    """
    Function to help in lemmatizing words
    :param treebank_tag:
    :return:
    """
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


def clean_text(sotu):
    """
    Defines how to each the text portion of each speech paragraph
    :param sotu:
    :return:
    """
    # treat hyphens between words as a single word
    sotu = re.sub(r"([a-zA-Z])\-([a-zA-Z])", r"\1_\2", sotu)

    # but remove hyphens between words
    sotu = re.sub(r"([0-9])\-([0-9])", r"\1 \2", sotu)

    # expand contractions
    sotu = iam.sub("i am", sotu)
    sotu = ive.sub("i have", sotu)
    sotu = hes.sub("he is", sotu)
    sotu = shes.sub("she is", sotu)
    sotu = weve.sub("we have", sotu)
    sotu = youve.sub("you have", sotu)
    sotu = willnot.sub("will not", sotu)
    sotu = cannot.sub("can not", sotu)
    sotu = itis.sub("it is", sotu)
    sotu = letus.sub("let us", sotu)
    sotu = heis.sub("he is", sotu)
    sotu = sheis.sub("she is", sotu)
    sotu = howis.sub("how is", sotu)
    sotu = thatis.sub("that is", sotu)
    sotu = thereis.sub("there is", sotu)
    sotu = whatis.sub("what is", sotu)
    sotu = whereis.sub("where is", sotu)
    sotu = whenis.sub("when is", sotu)
    sotu = whois.sub("who is", sotu)
    sotu = whyis.sub("why is", sotu)
    sotu = youall.sub("you all", sotu)
    sotu = youare.sub("you are", sotu)
    sotu = would.sub(" would", sotu)
    sotu = will.sub(" will", sotu)
    sotu = s_apostrophe.sub("s has ", sotu)
    sotu = has.sub(" has", sotu)
    sotu = nt.sub(" not", sotu)
    sotu = have.sub(" have", sotu)

    # capture common patterns
    sotu = united_states_of_america.sub(" _usa_ ", sotu)
    sotu = united_states.sub(" _usa_ ", sotu)
    sotu = us.sub(" _usa_ ", sotu)
    sotu = usa.sub(" _usa_ ", sotu)
    sotu = usp.sub(" _usa_ ", sotu)
    sotu = usap.sub(" _usa_ ", sotu)
    sotu = years.sub(" _year_ ", sotu)
    sotu = dollars.sub(" _dollars_ ", sotu)
    sotu = percent.sub(" percent ", sotu)
    sotu = percents.sub(" percent ", sotu)
    sotu = times.sub(" _time_ ", sotu)
    sotu = dates.sub(" _date_ ", sotu)
    sotu = punct.sub(" ", sotu)

    # tokenize and remove stopwords
    sotu = [w for w in sotu.split() if w.lower() not in stopword_set]

    # lemmatize and lowercase
    sotu = [lemmatizer.lemmatize(w, pos=get_wordnet_pos(p)).lower() \
                if get_wordnet_pos(p) != '' else lemmatizer.lemmatize(w).lower() \
            for w, p in pos_tag(sotu)]
    sotu = ['us' if w == 'u' else w for w in sotu]
    return sotu


def doc_generator(fname):
    with open(fname, "r") as f:
        for line in f:
            fields = line.replace("\n", "").split("\t")
            doc = fields[-1]
            yield doc.split()


def keys_generator(fname):
    """
    returns generator of (timestep, author) pairs
    """
    with open(fname, "r") as f:
        for line in f:
            fields = line.replace("\n", "").split("\t")
            yield int(fields[0]), fields[1]



def main():
    # loop through each paragraph in each speech, collect keys and preprocess each text
    data_dir = "../../data/sotu/"
    # infile = "sotu_doc_per_paragraph.txt"
    # infile = "sotu_1901_2016.txt"
    infile = "sotu_raw.txt"
    min_year = 1901

    words_per_speech = {} # total number of lines per speech
    docs = []
    keys = []
    year = None
    word_count = 0
    with open(data_dir + infile, "r") as ifile:
        for i, line in enumerate(ifile):
            line = line.replace("\n", "")

            if line[0:3] == "***":
                # this line is the beginning of a new speech
                fields = line.split("\t")
                author = fields[1].replace(' ', '_').replace('.', '_').replace(",", "_")
                year = int(fields[2][-4:])
                speech = str(year) + "_" + author
                word_count = 0
                text = fields[-1]
            else:
                # another line from the same speech
                text = line

            if year < min_year:
                continue

            doc = clean_text(text)
            docs.append(doc)
            keys.append((speech, word_count)) # save word count at the start of this document
            words_per_speech[speech] = word_count
            word_count += len(doc)


    print("Number of words in each speech:")
    for a, n in words_per_speech.items():
        print(a, n)



    # normalize timesteps so each document's "time step" indicates position in speech [0, 100]
    norm_keys = []
    for speech, word_count in keys:
        norm_keys.append((speech, int(round(100.0 * word_count / words_per_speech[speech]))))

    keys = norm_keys
    # sort documents by time then speech
    docs = np.array(docs)
    keys = np.array(keys)
    sorting_values = np.array([str(n).zfill(4) + a for a, n in zip(keys[:, 0], keys[:, 1])])
    sort_order = np.argsort(sorting_values)
    docs = docs[sort_order]
    keys = keys[sort_order]


    # how many docs per time step
    timestep_counts = {}
    for _, timestep in keys:
        ts = int(timestep)
        if ts not in timestep_counts:
            timestep_counts[ts] = 1
        else:
            timestep_counts[ts] += 1

    # write out the keys and cleaned text to seperate file, possibly useful later
    with open(data_dir + "sotu_bow.txt", "w") as outfile:
        for (speech, pct_in_speech), doc in zip(keys, docs):
            outfile.write(str(pct_in_speech) + "\t" + str(speech) + "\t" + ' '.join(doc) + "\n")


    resort = False
    if resort:
        print("sorting keys and docs")
        cmd = """/bin/bash -c "sort %s -n -t $'\t' -k1,1 -k2,2 -o %s -S %s" """ % ("sotu_bow.txt", "sotu_bow.txt", "75%")
        subprocess.call(cmd, shell=True)


    build_corpus = True
    if build_corpus:
        print("creating vocab")
        docs = doc_generator(data_dir + "sotu_bow.txt")
        vocab = corpora.Dictionary(docs)
        vocab.filter_extremes(no_below=3, no_above=0.95, keep_n=10000)
        vocab.compactify()

        corpus = (vocab.doc2bow(tokens) for tokens in doc_generator(data_dir + "sotu_bow.txt"))
        corpora.MmCorpus.serialize(data_dir + 'sotu.mm', corpus)
        corpus = corpora.MmCorpus(data_dir + 'sotu.mm')
        corpora.BleiCorpus.save_corpus(data_dir + 'sotu.ldac', corpus, id2word=vocab)
        os.remove(data_dir + "sotu.mm")
        os.remove(data_dir + "sotu.mm.index")
        os.rename(data_dir + "sotu.ldac.vocab", data_dir + "sotu_vocab.txt")
        del docs
        del corpus
        del vocab

    # write out data in format for DAP model:
    # total_timesteps
    # timestep[0]
    # num_docs[t=0]
    # author num_terms term_0:count ... term_n:count
    # author num_terms term_0:count ... term_n:count
    # ...
    # author num_terms term_0:count ... term_n:count
    # ...
    # timestep[T]
    # num_docs[t=T]
    # author num_terms term_0:count ... term_n:count
    # author num_terms term_0:count ... term_n:count
    dap = True
    if dap:
        num_timesteps = len(timestep_counts)
        lda_file = open(data_dir + "sotu.ldac", "r")
        prev_ts = -1
        with open(data_dir + "sotu_full.txt", "w") as dap_file:
            dap_file.write(str(num_timesteps) + "\n")
            for ts, speech in keys_generator(data_dir + "sotu_bow.txt"):
                if ts != prev_ts:
                    # new time step
                    dap_file.write(str(ts) + "\n")
                    dap_file.write(str(timestep_counts[ts]) + "\n")

                # otherwise, write out next speech + doc
                ldac = lda_file.readline()
                dap_file.write(speech + " " + ldac)
                prev_ts = ts

        lda_file.close()
        os.remove(data_dir + 'sotu.ldac')

    split = True
    if split:
        # split DAP file into training and test sets
        print("Split into training and test sets")
        pct_train = 0.1
        with open(data_dir + "sotu_full.txt", "r") as dap_file, \
            open(data_dir + "sotu_train.txt", "w") as train, \
            open(data_dir + "sotu_test.txt", "w") as test:
            num_timesteps = int(dap_file.readline().replace("\n", ""))
            train.write(str(num_timesteps) + "\n")
            test.write(str(num_timesteps) + "\n")
            for t in range(num_timesteps):
                ts = int(dap_file.readline().replace("\n", ""))
                num_docs_t = int(dap_file.readline().replace("\n", ""))
                print("t:", t, "total:", num_docs_t)
                test_ids = np.random.choice(num_docs_t, size=int(np.ceil(num_docs_t * pct_train)), replace=False)
                train_ids = np.delete(np.arange(num_docs_t), test_ids)
                print("\ttrain:", len(train_ids), "test:", len(test_ids))
                train.write(str(ts) + "\n")
                test.write(str(ts) + "\n")
                train.write(str(len(train_ids)) + "\n")
                test.write(str(len(test_ids)) + "\n")
                for i in range(num_docs_t):
                    doc = dap_file.readline()
                    if i in test_ids:
                        test.write(doc)
                    else:
                        train.write(doc)


if __name__ == "__main__":
    main()
