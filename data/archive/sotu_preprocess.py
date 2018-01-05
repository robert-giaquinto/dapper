import re
from gensim import corpora
import logging
import os
import numpy as np
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def get_wordnet_pos(treebank_tag):
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


lemmatizer = WordNetLemmatizer()


base_stopwords = stopwords.words("english")
custom_stopwords = ['take', 'say', 'back', 'see', 'come', 'still', 'make', 'thing', 'get',
    'go', 'well', 'good', 'think', 'much', 'like', 'through', 'up', 'down', 'over', 'under',
    'got', 'get', 'til', 'also', 'would', 'could', 'should', 'really',
    'didnt', 'cant', 'thats', 'doesnt', 'didnt', 'wont', 'wasnt', 'hows',
    'hadnt', 'hasnt', 'willnt', 'isnt', 'arent', 'werent', 'havent',
    'wouldnt', 'couldnt', 'shouldnt',  'shouldve', 'couldve', 'wouldve', 'upon', 'may', 'subject',
    'theres', 'whats', 'whens', 'whos', 'wheres'] + list('abcdefghjklmnopqrstuvwxyz')
stopword_set = set(base_stopwords + custom_stopwords)
print("stopwords:\n", stopword_set)



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

data_dir = "../../data/"
prev_author = ''
ct = 0
docs = []
keys = []
sotu_dict = {}
with open(data_dir + "sotu_raw.txt", "r") as f:
    for i, line in enumerate(f):

        fields = line.replace("\n", "").split("\t")
        author = fields[0].replace(" ", "_").replace(".", "")
        date = fields[1]
        year = int(date[-4:])
        sotu = fields[2]

        if author != prev_author:
            prev_author = author
            ct = 0
        else:
            ct += 1

        if ct > 7:
            continue

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

        if ct not in sotu_dict:
            sotu_dict[ct] = 1
        else:
            sotu_dict[ct] += 1

        docs.append(sotu)
        keys.append((author, ct, year, date))

# sort by time then author
docs = np.array(docs)
keys = np.array(keys)
sorting_values = np.array([int(i) + (int(y) / 2018.0) for i, y in zip(keys[:, 1], keys[:, 2])])
sort_order = np.argsort(sorting_values)
docs = docs[sort_order]
keys = keys[sort_order]

# write out the keys and cleaned text to seperate file
with open(data_dir + "sotu_keys.txt", "w") as out_keys, open(data_dir + "sotu_clean.txt", "w") as out_text:
    for (author, sotu_order, year, date), text_arr in zip(keys, docs):
        out_keys.write("\t".join([str(author), str(sotu_order), str(year), date]) + "\n")
        out_text.write(' '.join(text_arr) + "\n")

# convert docs to a proper corpus
vocab = corpora.Dictionary(docs)
vocab.filter_extremes(no_below=3, no_above=0.99, keep_n=15000)
vocab.compactify()
corpus = (vocab.doc2bow(tokens) for tokens in docs)
corpora.MmCorpus.serialize(data_dir + 'sotu.mm', corpus)
corpus = corpora.MmCorpus(data_dir + 'sotu.mm')
corpora.BleiCorpus.save_corpus(data_dir + 'sotu.ldac', corpus, id2word=vocab)
os.remove(data_dir + "sotu.mm")
os.remove(data_dir + "sotu.mm.index")

# write out data in dap format
num_timesteps = len(sotu_dict)
lda_file = open(data_dir + "sotu.ldac", "r")
keys_ptr = 0
with open(data_dir + "sotu_dap.txt", "w") as dap_file:
    dap_file.write(str(num_timesteps) + "\n")
    for ts in sorted(sotu_dict.keys()):
        dap_file.write(str(ts) + "\n")
        num_docs = sotu_dict[ts]
        dap_file.write(str(num_docs) + "\n")
        for d in range(num_docs):
            ldac = lda_file.readline()
            author = keys[keys_ptr, 0]
            dap_file.write(author + " " + ldac)
            if int(keys[keys_ptr, 1]) != int(ts):
                print("keys pointer points to time step", keys[keys_ptr, 1], "but this should only be", ts)
            keys_ptr += 1

lda_file.close()
