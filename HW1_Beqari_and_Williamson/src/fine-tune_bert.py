import os
os.environ["CUDA_VISIBLE_DEVICES"]=" "
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['TF_CPP_MIN_LOG_LEVEL']="3"
from sentence_transformers import SentenceTransformer, SentencesDataset, InputExample, losses, evaluation
from torch.utils.data import DataLoader
# from pandarallel import pandarallel
import pandas as pd
import numpy as np

# pandarallel.initialize()

# https://www.sbert.net/docs/training/overview.html

def get_bad_prediction(row):
    if row["label"] != row["prediction"]: 
        return row

def cosine_similarity(x1, x2):
        return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))

# read in the data
x_train = pd.read_pickle("x_train.pkl", compression="infer")

# get all bad predictions
bad_predictions = pd.DataFrame()
bad_predictions = x_train.apply(lambda row: get_bad_prediction(row), axis=1, result_type="broadcast")
bad_predictions = bad_predictions.drop("label", axis=1).dropna()
bad_predictions = bad_predictions.rename(columns={"comment": "bad_comment", "embeddings": "bad_embeddings", "prediction": "bad_prediction"})

# sample some % of x_train
x_sample = pd.DataFrame()
x_sample = x_train.sample(frac=0.2, replace=False, random_state=0)
x_sample = x_sample.sample(frac=0.1, replace=False, random_state=0).reset_index(drop=True)

def decrease_score(score):
    score = 1 / ( 1.5 + np.exp(np.absolute((10 * score) - 5)))
    return score

def increase_score(score):
    score = 1 - decrease_score(score)
    return score

def adjust_score(row):
    embeddings     = row["embeddings"]
    bad_embeddings = row["bad_embeddings"]
    label          = row["label"]
    bad_prediction = row["bad_prediction"]
    score = cosine_similarity(embeddings, bad_embeddings)

    if label == bad_prediction:
        # means that they have a high score, hence decrease
        score = decrease_score(score)
    elif label != bad_prediction:
        # means that they have a low score, hence increase
        score = increase_score(score)
    return score

# match each row with a bad prediction
x_sample_size = pd.Index(x_sample).size
bad_predictions = bad_predictions.sample(x_sample_size, replace=True, random_state=0)
bad_predictions = bad_predictions.sample(frac=1, replace=False, random_state=0).reset_index(drop=True)
x_sample = x_sample.join(bad_predictions)

# calculate score between embeddings and bad embeddings
score = pd.DataFrame()
x_sample["score"] = x_sample.apply(lambda row: adjust_score(row), axis=1)

def create_example(comment, bad_comment, score):
    input_example = InputExample(texts=[comment, bad_comment], label=score)
    return input_example
    
x_sample["input"] = x_sample.apply(lambda row: create_example(row["comment"], row["bad_comment"], row["score"]), axis=1)

# load model
embedder = SentenceTransformer('bert-large-nli-mean-tokens') # or any other pretrained model
print("embedder loaded...")

# define your train dataset, the dataloader, and the train loss
train_dataset = SentencesDataset(x_sample["input"].tolist(), embedder)
train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=1, num_workers=1)
train_loss = losses.CosineSimilarityLoss(embedder)

# dummy evaluator to make the api work
sentences1 = ['This list contains the first column', 'With your sentences', 'You want your model to evaluate on']
sentences2 = ['Sentences contains the other column', 'The evaluator matches sentences1[i] with sentences2[i]', 'Compute the cosine similarity and compares it to scores[i]']
scores = [0.3, 0.6, 0.2]
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)

# tune the model
embedder.fit(train_objectives=[(train_dataloader, train_loss)], 
    epochs=1, 
    warmup_steps=100, 
    evaluator=evaluator, 
    evaluation_steps=1,
    output_path="fine_tuned_bert")
