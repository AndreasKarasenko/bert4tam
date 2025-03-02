import os
from typing import List
from datasets import Dataset
from transformers import (AutoTokenizer, TFAutoModelForSequenceClassification,
                          DataCollatorWithPadding, create_optimizer)
from scipy.special import softmax
from scipy.stats import pearsonr

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import (roc_auc_score, accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix, r2_score)

os.chdir("./models/karasenko/")


def weighted_continous(probs: List[float]):
    score = 0
    probability = 1e-10 # assure that no zero division can occur
    for idx, value in enumerate(probs):
        score += idx * value
        probability += value
    return score / probability


df = pd.read_excel("../../data/IKEA_Umfrage.xlsx")

data = df.loc[:,["Review","Sterne"]]
data = data.rename(columns={"Review":"text",
                            "Sterne":"label"})


checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"


model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

dataset = Dataset.from_pandas(data)#.remove_columns(["__index_level_0__"])
dataset = dataset.map(preprocess_function, batched=True)



collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

train = dataset.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=4,
    collate_fn=collator
)

# we want to classify based on reivews, here 2 classes
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if "bert_base_multilingual_score.index" in os.listdir("./final_eval/pretrained/"):
    model.load_weights("./pretrained/bert_base_multilingual_score")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


dataset = Dataset.from_pandas(data)
dataset = dataset.map(preprocess_function, batched=True)

d2 = dataset.map(preprocess_function, batched=True) # for later reintegration into df

dataset = dataset.train_test_split(test_size=0.2)


collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")


batch_size = 4
num_epochs = 3
batches_per_epoch = len(dataset) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model.compile(optimizer=optimizer)
print(model.summary())

if "bert_base_multilingual_score.index" not in os.listdir("./final_eval/pretrained/"):
    model.fit(train, epochs=3)
    model.save_weights("./final_eval/pretrained/bert_base_multilingual_score")

preds_train = model.predict(train)

probs_train = pd.DataFrame(softmax(preds_train.logits, axis=1))


def weighted_continous(probs: List[float]):
    score = 0
    probability = 1e-10 # assure that no zero division can occur
    for idx, value in enumerate(probs):
        score += idx * value
        probability += value
    return score / probability


probs_train["weighted_score"] = probs_train.apply(lambda x: weighted_continous(x), axis=1)
probs_train["weighted_score"] += 1

probs_train["actual"] = dataset["label"]

# print(model.evaluate(test))


actual_train = dataset["label"]
predicted_train = np.argmax(preds_train.logits, axis=1)
predicted_train += 1


print(confusion_matrix(actual_train, predicted_train))

print(precision_recall_fscore_support(actual_train, predicted_train))

pd.DataFrame(confusion_matrix(actual_train, predicted_train)).to_clipboard()

print(r2_score(probs_train["actual"],probs_train["weighted_score"]))


print(pearsonr(probs_train["weighted_score"],probs_train["actual"]))

# add in the old stuff
old = pd.read_csv("./final_eval/data.csv",index_col=0)
old = old.drop(columns=["text","label"])
