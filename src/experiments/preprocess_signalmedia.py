import re
import os
import json
import datetime as dt
from collections import Counter
from gensim import corpora
import numpy as np
import subprocess
import argparse

from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag


def get_wordnet_pos(treebank_tag):
    """
    Need to know the part of speech of each word to properly lemmatize.
    this function standardizes the POS codes so that they're understandable
    by the lemmatizing function.
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


stopword_set = set(["a", "all", "am", "an",
                    "and", "any", "are", "as", "at", "be", "because", "been",
                    "being", "but", "by", "can",
                    "cannot", "could", "did", "do", "does", "doing", "down",
                    "each", "few", "for", "had", "has", "have", "having",
                    "here",
                    "if", "in", "into", "is", "it", "its", "itself", "me", "more",
                    "most", "nor", "not", "of", "off", "on",
                    "only", "or", "other", "ought", "ourselves", "out", "over",
                    "own", "same", "should", "so", "some", "such", "than", "that", "the",
                    "then", "there", "these",
                    "this", "those", "through", "to", "too", "under", "until", "up", "very",
                    "was", "were", "what", "when", "where", "which", "while", "who",
                    "whom", "why", "with", "would", "return", "arent", "cant", "couldnt", "didnt", "doesnt",
                    "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt",
                    "its", "lets", "mustnt", "shant", "shes", "shouldnt", "thats", "theres",
                    "theyll", "theyre", "theyve", "wasnt", "were", "werent", "whats", "whens",
                    "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre",
                    "youve", "will", "came", "though",
                    "way", "come", "might", "now", "much",
                    "i", "he", "she", "we", "they", "you", "say", "their", "his" ,"her", "your", "him", "ve", "re",
                    "think", "thing", "about", "tell", "many", "give", "before", "after", "my", "start", "end", "go",
                    "about", "make", "get", "also", "our", "them"] +
                   list("abcdefghjklmnopqrstuvwxyz"))
lemmatizer = WordNetLemmatizer()
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
punct = re.compile(r"[^a-zA-Z_ ]")
special_chars = re.compile(r"&[a-z]+;")
urls = re.compile(r'((http|ftp|https)://)?([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?',
                  re.IGNORECASE)
times = re.compile(r"(2[0-3]|[01]?[0-9]):([0-5]?[0-9])")
dates = re.compile(r"\b1?[0-9]?[-/]1?[0-9]?[-/](18|19|20)?[0-9]{2}\b")
percent = re.compile(r"[1-9][0-9\.]*\%")
dollars = re.compile(r"\$[1-9][0-9,\.]*")
years = re.compile(r"\b(18|19|20)[0-9]{2}\b")
html = re.compile(r"<[^>]*>")
emails = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,4}\b", re.IGNORECASE)


def keys_generator(fname, rel_date=False):
    """
    returns generator of (source, date) pairs
    """
    with open(fname, "r") as f:
        for line in f:
            fields = line.replace("\n", "").split("\t")
            if rel_date:
                t = int(fields[1])
            else:
                t = dt.datetime.strptime(fields[1], "%Y-%m-%d %H:%M:%S")
            yield fields[0], t


def doc_generator(fname):
    with open(fname, "r") as f:
        for line in f:
            fields = line.replace("\n", "").split("\t")
            doc = fields[-1]
            yield doc.split()


def text2corpus(input_fname, corpus_fname, keep_n=25000, print_top_word_counts=False):
    # define the the vocabulary
    docs = doc_generator(input_fname)
    vocab = corpora.Dictionary(docs)
    vocab.filter_extremes(no_above=0.5, keep_n=keep_n)
    vocab.compactify()

    # create BOW corpus
    corpus = (vocab.doc2bow(tokens) for tokens in doc_generator(input_fname))
    corpora.BleiCorpus.save_corpus(corpus_fname, corpus, id2word=vocab)

    if print_top_word_counts:
        word_counts = {v: 0 for v in vocab.values()}
        for doc in doc_generator(input_fname):
            for word in doc:
                if word in word_counts:
                    word_counts[word] += 1

        names = ['word', 'count']
        formats = ['S100', 'int']
        dtype = dict(names=names, formats=formats)
        word_counts = np.array(list(word_counts.items()), dtype=dtype)
        word_counts = np.sort(word_counts, order=['count'])
        print("top words", word_counts[-50:])


def compute_relative_dates(fname):
    keys = list(keys_generator(fname))
    # identify first date for each src
    src_min = {}
    overall_min = dt.datetime.today()
    for src, d in keys:
        if d < overall_min:
            overall_min = d

        if src not in src_min:
            src_min[src] = d
        else:
            if src_min[src] > d:
                src_min[src] = d

    print("overall min date:", overall_min)

    # compute relative dates
    rel_dts = []
    for src, d in keys:
        days_dif = int(round((d - overall_min).days + ((d - overall_min).seconds / 60. / 60. / 24.)))
        rel_dts.append(days_dif)

    # compute counts per time step
    time_counts = Counter(rel_dts)

    # replace the current date in the data with the relative date.
    tmp_file = "tmp.txt"
    os.rename(fname, tmp_file)
    with open(tmp_file, "r") as fin, open(fname, "w") as fout:
        for line, rel_dt in zip(fin, rel_dts):
            fields = line.replace("\n", "").split("\t")
            arr = [fields[0], rel_dt, fields[2]]
            fout.write('\t'.join([str(s) for s in arr]) + "\n")

    return time_counts


def dappify(time_counts, keys_fname, corpus_fname, full_fname):
    num_timesteps = len(time_counts)
    lda_file = open(corpus_fname, "r")
    prev_ts = -1
    with open(full_fname, "w") as fout:
        fout.write(str(num_timesteps) + "\n")
        for src, ts in keys_generator(keys_fname, rel_date=True):
            if ts != prev_ts:
                # new time step
                fout.write(str(ts) + "\n")
                fout.write(str(time_counts[ts]) + "\n")

            # otherwise, write out next source + doc
            ldac = lda_file.readline()
            fout.write(src + " " + ldac)
            prev_ts = ts

    lda_file.close()


def split_train_test(full_fname, train_fname, test_fname, test_ratio=0.0):
    # split DAP file into training and test sets
    with open(full_fname, "r") as dap_file, \
            open(train_fname, "w") as train, \
            open(test_fname, "w") as test:

        num_timesteps = int(dap_file.readline().replace("\n", ""))
        train.write(str(num_timesteps) + "\n")
        test.write(str(num_timesteps) + "\n")

        for t in range(num_timesteps):
            ts = int(dap_file.readline().replace("\n", ""))
            num_docs_t = int(dap_file.readline().replace("\n", ""))
            print("t:", t, "total:", num_docs_t)

            test_ids = np.random.choice(num_docs_t, size=int(np.ceil(num_docs_t * test_ratio)), replace=False)
            train_ids = np.delete(np.arange(num_docs_t), test_ids)

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


def filter_sources(keep_sources, input_fname, filtered_fname, min_date=dt.datetime(year=2015,month=9, day=1)):
    # replace the current date in the data with the relative date.
    num_docs = 0
    with open(input_fname, "r") as fin, open(filtered_fname, "w") as fout:
        for line in fin:
            fields = line.replace("\n", "").split("\t")
            if fields[0] in keep_sources and dt.datetime.strptime(fields[1], "%Y-%m-%d %H:%M:%S") >= min_date:
                fout.write(line)
                num_docs += 1

    return num_docs


def check_for_news(json_dict):
    if json_dict["media-type"] == "News":
        is_news = True
    else:
        is_news = False

    return is_news


def extract_keys(json_dict):
    """
    extract keys from the json data
    :param json_dict:
    :return:
    """
    # TODO: transform date into something usable?
    src = re.sub(r"\s+", "_", json_dict['source'])
    src = re.sub(r"[^a-zA-Z0-9_]", "_", src)
    src = re.sub(r"[_]+", "_", src)
    src = re.sub(r'^([0-9])', r'_\1', src)
    published_date = dt.datetime.strptime(json_dict['published'], "%Y-%m-%dT%H:%M:%SZ")
    # keys = [src, published_date, json_dict["media-type"]]
    keys = [src, published_date]
    return keys


def extract_text(json_dict):
    # TODO: transform date into something usable?
    text = ' '.join([json_dict['title'], json_dict['content']]).strip()
    text = re.sub(r"\s+", ' ', text)
    return text


def scrub_text(text):
    """
        Defines how to clean each of the texts
        :param text:
        :return:
        """
    # all to lowercase
    text = text.lower()

    text = html.sub(" ", text)
    text = years.sub(" _year_ ", text)
    text = dollars.sub(" _dollars_ ", text)
    text = percent.sub(" _percent_ ", text)
    text = times.sub(" _time_ ", text)
    text = urls.sub(" _url_ ", text)
    text = dates.sub(" _date_ ", text)
    text = special_chars.sub(" ", text)
    text = emails.sub(" _email_ ", text)

    # treat hyphens between words as a single word
    text = re.sub(r"([a-zA-Z])\-([a-zA-Z])", r"\1_\2", text)

    # expand contractions
    text = iam.sub("i am", text)
    text = ive.sub("i have", text)
    text = hes.sub("he is", text)
    text = shes.sub("she is", text)
    text = weve.sub("we have", text)
    text = youve.sub("you have", text)
    text = willnot.sub("will not", text)
    text = cannot.sub("can not", text)
    text = itis.sub("it is", text)
    text = letus.sub("let us", text)
    text = heis.sub("he is", text)
    text = sheis.sub("she is", text)
    text = howis.sub("how is", text)
    text = thatis.sub("that is", text)
    text = thereis.sub("there is", text)
    text = whatis.sub("what is", text)
    text = whereis.sub("where is", text)
    text = whenis.sub("when is", text)
    text = whois.sub("who is", text)
    text = whyis.sub("why is", text)
    text = youall.sub("you all", text)
    text = youare.sub("you are", text)
    text = would.sub(" would", text)
    text = will.sub(" will", text)
    text = s_apostrophe.sub("s has ", text)
    text = has.sub(" has", text)
    text = nt.sub(" not", text)
    text = have.sub(" have", text)

    # remove punctuation
    text = punct.sub(" ", text)

    # tokenize and lemmatize
    text = [lemmatizer.lemmatize(w, pos=get_wordnet_pos(p)).lower() \
                if get_wordnet_pos(p) != '' \
                else lemmatizer.lemmatize(w).lower() \
            for w, p in pos_tag(text.split())]

    # remove stopwords
    text = [w for w in text if w not in stopword_set]

    return ' '.join(text)


def parse_json(input, output):
    sources = Counter()
    i = 0
    with open(input, 'r') as fin, open(output, "w") as fout:
        for line in fin:
            # parse the json into a dictionary
            json_dict = json.loads(line)

            is_news = check_for_news(json_dict)
            if is_news:
                continue

            i += 1
            # pull out the data we need from the text
            a_key = extract_keys(json_dict)
            a_text = extract_text(json_dict)
            sources.update([a_key[0]])

            # clean text
            clean_text = scrub_text(a_text)

            fout.write('\t'.join([str(s) for s in a_key]) + "\t" + clean_text + "\n")

    return sources


def main():
    parser = argparse.ArgumentParser(description='Run dap model on signalmedia data.')
    parser.add_argument('--min_threshold', type=int, default=3)
    parser.add_argument('--max_threshold', type=int, default=30)
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--input_file', type=str, default="../../data/signalmedia/signalmedia-1m.jsonl")
    parser.add_argument('--keys_and_text_file', type=str, default="signalmedia_keys_and_text.txt")
    parser.add_argument('--filtered_file', type=str, default="filtered.txt")
    parser.add_argument('--corpus_file', type=str, default="signalmedia.bow")
    parser.add_argument('--dap_file', type=str, default="signalmedia.dap")
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()

    keys_and_text = os.path.join(args.data_dir, args.keys_and_text_file)
    filtered = os.path.join(args.data_dir, args.filtered_file)
    corpus = os.path.join(args.data_dir, args.corpus_file)
    all_dap = os.path.join(args.data_dir, args.dap_file)
    train = os.path.join(args.data_dir, "train_" + args.dap_file)
    test = os.path.join(args.data_dir, "test_" + args.dap_file)

    # parse out the keys and cleaned up text
    rerun = False
    if rerun:
        sources = parse_json(args.input_file, keys_and_text)
    else:
        sources = Counter()
        with open(keys_and_text, 'r') as fin:
            for line in fin:
                fields = line.replace("\n", "").split("\t")
                sources.update([fields[0]])


    # save only the top n most commonly appearing sources
    keep_sources = set([src for src in sources if args.min_threshold <= sources[src] <= args.max_threshold])
    num_docs = filter_sources(keep_sources, keys_and_text, filtered, min_date=dt.datetime(year=2015, month=9, day=1))
    counts_dict = {k: sources[k] for k in keep_sources}
    print("(Doc counts: Num souces) - ", Counter(counts_dict.values()))
    print("Keeping {} documents in the corpus and {} sources".format(
        num_docs, len(keep_sources)))

    # transform the dates into relative dates
    time_counts = compute_relative_dates(filtered)
    print("time counts", time_counts)

    # sort the data by src and date
    cmd = """/bin/bash -c "sort %s -n -t $'\t' -k1,1 -k2,2 -o %s -S %s" """ % (
        filtered, filtered, "75%")
    subprocess.call(cmd, shell=True)

    # convert texts to bow format
    text2corpus(filtered, corpus, print_top_word_counts=True)

    # transform data into DAP format
    dappify(time_counts, filtered, corpus, all_dap)

    # split train test
    split_train_test(all_dap, train, test, test_ratio=args.test_ratio)


if __name__ == "__main__":
    main()
