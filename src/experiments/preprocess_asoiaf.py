import re
from gensim import corpora
import subprocess
import os
import numpy as np


stopword_set = set(["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "cannot", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its", "itself", "me", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours ", "ourselves", "out", "over", "own", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "why", "with", "would", "you", "your", "yours", "yourself", "yourselves", "return", "arent", "cant", "couldnt", "didnt", "doesnt", "dont", "hadnt", "hasnt", "havent", "hes", "heres", "hows", "im", "isnt", "its", "lets", "mustnt", "shant", "shes", "shouldnt", "thats", "theres", "theyll", "theyre", "theyve", "wasnt", "were", "werent", "whats", "whens", "wheres", "whos", "whys", "wont", "wouldnt", "youd", "youll", "youre", "youve", "said", "will", "like", "say", "came", "away", "though", "took", "good", "must", "way", "come", "might", "told", "now", "make", "made", "see", "know", "well", "back", "asked", "looked", "tell", "much", "one"])

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


def clean_text(text):
    """
    Defines how to clean each of the texts
    :param text:
    :return:
    """
    # all to lowercase
    text = text.lower()

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

    # tokenize and remove stopwords
    text = [w for w in text.split() if w not in stopword_set]
    return text


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
    print(stopword_set)

    data_dir = "../../data/asoiaf/"
    docs = []
    keys = []
    characters = ["AERON", "AREO", "ARIANNE", "ARYA", "ARYS", "ASHA", "BARRISTAN", "BRAN", "BRIENNE", "CATELYN", "CERSEI", "CONNINGTON", "DAENERYS", "DAVOS", "EDDARD", "JAIME", "JON", "MELISANDRE", "QUENTYN", "SAMWELL", "SANSA", "THEON", "TYRION", "VICTARION"]
    chapter_counts = {name: 0 for name in characters}
    wpc = {name: 0 for name in characters}  # words per character

    with open(data_dir + "asoiaf.txt", "r") as f:
        for line in f:
            line = line.replace("\n", "")
            if line in characters:
                # new chapter
                chapter_counts[line] += 1
                current_char = line
                current_chapter = chapter_counts[line]
            else:
                # continuation of previous chapter
                text = clean_text(line)
                keys.append([current_char, wpc[current_char]])
                docs.append(text)
                wpc[current_char] += len(text)

    print("Number of words in each speech:")
    for c, wc in wpc.items():
        print(c, wc)

    # normalize timesteps so each document's "time step" indicates position in plot [0, 100]
    norm_keys = []
    for c, wc in keys:
        norm_keys.append((c, int(round(chapter_counts[c] * 10.0 * wc / wpc[c]))))

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
    for _, ts in keys:
        ts = int(ts)
        if ts not in timestep_counts:
            timestep_counts[ts] = 1
        else:
            timestep_counts[ts] += 1

    # write out the keys and cleaned text to seperate file, possibly useful later
    with open(data_dir + "asoiaf_bow.txt", "w") as outfile:
        for (character, pct_in_arc), doc in zip(keys, docs):
            outfile.write(str(pct_in_arc) + "\t" + str(character) + "\t" + ' '.join(doc) + "\n")

    build_corpus = True
    if build_corpus:
        print("creating vocab")
        bow = doc_generator(data_dir + "asoiaf_bow.txt")
        vocab = corpora.Dictionary(bow)
        vocab.filter_extremes(no_above=0.5, keep_n=5000)
        vocab.compactify()

        print("creating corpus")
        corpus = (vocab.doc2bow(tokens) for tokens in doc_generator(data_dir + "asoiaf_bow.txt"))
        corpora.MmCorpus.serialize(data_dir + 'asoiaf.mm', corpus)
        corpus = corpora.MmCorpus(data_dir + 'asoiaf.mm')
        corpora.BleiCorpus.save_corpus(data_dir + 'asoiaf.ldac', corpus, id2word=vocab)
        os.remove(data_dir + "asoiaf.mm")
        os.remove(data_dir + "asoiaf.mm.index")
        os.rename(data_dir + "asoiaf.ldac.vocab", data_dir + "asoiaf_vocab.txt")

        # print term frequencies
        print("word freq")
        word_counts = {v: 0 for v in vocab.values()}
        print(word_counts['mlord'])

        for doc in docs:
            for word in doc:
                if word in word_counts:
                    word_counts[word] += 1

        print(word_counts['mlord'])

        names = ['word','count']
        formats = ['S100','int']
        dtype = dict(names=names, formats=formats)
        word_counts = np.array(list(word_counts.items()), dtype=dtype)
        word_counts = np.sort(word_counts, order=['count'])
        print(word_counts[-50:])
        print(word_counts[0:50])

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
    print("dap format")
    dap = True
    if dap:
        num_timesteps = len(timestep_counts)
        lda_file = open(data_dir + "asoiaf.ldac", "r")
        prev_ts = -1
        with open(data_dir + "asoiaf_full.txt", "w") as dap_file:
            dap_file.write(str(num_timesteps) + "\n")
            for ts, character in keys_generator(data_dir + "asoiaf_bow.txt"):
                if ts != prev_ts:
                    # new time step
                    dap_file.write(str(ts) + "\n")
                    dap_file.write(str(timestep_counts[ts]) + "\n")

                # otherwise, write out next character + doc
                ldac = lda_file.readline()
                dap_file.write(character + " " + ldac)
                prev_ts = ts

        lda_file.close()
        os.remove(data_dir + 'asoiaf.ldac')

    split = True
    if split:
        # split DAP file into training and test sets
        print("Split into training and test sets")
        pct_train = 0.1
        with open(data_dir + "asoiaf_full.txt", "r") as dap_file, \
            open(data_dir + "asoiaf_train.txt", "w") as train, \
            open(data_dir + "asoiaf_test.txt", "w") as test:
            num_timesteps = int(dap_file.readline().replace("\n", ""))
            train.write(str(num_timesteps) + "\n")
            test.write(str(num_timesteps) + "\n")
            for t in range(num_timesteps):
                ts = int(dap_file.readline().replace("\n", ""))
                num_docs_t = int(dap_file.readline().replace("\n", ""))
                # print("t:", t, "total:", num_docs_t)
                test_ids = np.random.choice(num_docs_t, size=int(np.ceil(num_docs_t * pct_train)), replace=False)
                train_ids = np.delete(np.arange(num_docs_t), test_ids)
                # print("\ttrain:", len(train_ids), "test:", len(test_ids))
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










