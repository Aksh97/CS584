from gensim.summarization.textcleaner import clean_text_by_word
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from pandarallel import pandarallel
import nltk
import pandas as pd
import multiprocessing

pandarallel.initialize()

# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# read in data
train_labels = []
train_comments = []
valid_labels = ["+1", "-1"]

with open('trainhw1.txt') as file_reader:
    lines = file_reader.readlines()
    for index, line in enumerate(lines):
        labeled_review = line.strip().split("\t")
        label = labeled_review[0].strip()
        if label not in valid_labels:
            label = "<SKIP>"
        try:
            comment = labeled_review[1]
        except IndexError:
            label = "<SKIP>"
            comment = ""
        train_labels.append(label)
        train_comments.append(comment)

test_comments = []

with open('testdatahw1.txt') as file_reader:
    lines = file_reader.readlines()
    for line in lines:
        test_comments.append(line.strip())

train_dict = {"comment": train_comments, "label": train_labels}
x_train = pd.DataFrame(train_dict, columns=["comment", "label"])

test_dict = {"comment": train_comments}
x_test = pd.DataFrame(test_dict, columns=["comment"])

def paragraph_to_tokens(paragraph):
    try:
        text_dict = clean_text_by_word(paragraph)
        # try with removed stopwords
        # tokens = [value.token for key, value in text_dict.items() if not value.token in stop_words]
        tokens = [value.token for key, value in text_dict.items()]
        return tokens
    except TypeError:
        print('warning: nan found in paragraph - {}'.format(paragraph))
    return []

# step 1: convert paragraph to tokens
x_train["tokens"] = x_train["comment"].parallel_apply(lambda row: paragraph_to_tokens(row))
x_test["tokens"]  = x_test["comment"].parallel_apply(lambda row: paragraph_to_tokens(row))

# train the vocabolary with all words
all_tokens = pd.concat([x_train, x_test], keys=['tokens'], names=['tokens'])

# doc2vec model
# docs: https://radimrehurek.com/gensim/models/doc2vec.html
cores = multiprocessing.cpu_count()
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(all_tokens["tokens"].tolist())]
model_d2v = Doc2Vec(
    documents,
    dm=1,
    vector_size=300,
    negative=5,
    hs=0,
    sample=0,
    window=5,
    min_count=1,
    workers=cores,
    alpha=0.005,
    min_alpha=0.001,
    dbow_words=1, # 0-bow, 1-skip gram
    epochs=200)

# retrieve embeddings from model
x_train["embeddings"] = x_train["tokens"].parallel_apply(lambda row: model_d2v.infer_vector(row))
x_test["embeddings"]  = x_test["tokens"].parallel_apply(lambda row: model_d2v.infer_vector(row))

# save dataframes to file
x_train.to_pickle("x_train.pkl")
x_test.to_pickle("x_test.pkl")
