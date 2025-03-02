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

construct_type = "PE"

# duplicate 
df = pd.read_excel("./final_eval/20231113_IKEA_BASE.xlsx", index_col=0)
# uncomment the BI lines if you want non repeated
BI = df.loc[:,["content","PE", "train"]]
BIY = df.loc[:,["content","PEY", "train"]]
BIY.columns = ["content","PE", "train"]
BIZ = df.loc[:,["content","PEZ", "train"]]
BIZ.columns = ["content","PE", "train"]

data = pd.concat([BI,BIY,BIZ]).reset_index(drop=True)
print(data.shape)
data = data.loc[:,["content","PE","train"]]

# data = df.loc[:,["content","PE","train"]] # use for single non repeated constr

data = data.rename(columns={"content":"text",
                            "PE":"label"})

data = pd.DataFrame(data[["text","label", "train"]])
data.label = data.label - 1

# replace = {1:0,
#            2:0,
#            3:1,
#            4:1}

# data.label = data.label.replace(replace)

# model to use
checkpoint = "nlptown/bert-base-multilingual-uncased-sentiment"

# we want to classify based on reivews, here 2 classes
model = TFAutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

if "bert_base_multilingual_pe.index" in os.listdir("./final_eval/pretrained/"):
    model.load_weights("./final_eval/pretrained/bert_base_multilingual_pe")


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)

train = Dataset.from_pandas(data[data["train"] == 1])
train = train.map(preprocess_function, batched=True)

test = Dataset.from_pandas(data[data["train"] == 0])
test = test.map(preprocess_function, batched=True)

tr2 = train.map(preprocess_function, batched=True)
te2 = test.map(preprocess_function, batched=True)

d2 = Dataset.from_pandas(data)
d2 = d2.map(preprocess_function, batched=True)

collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")

train = train.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=True,
    batch_size=3,
    collate_fn=collator
)

train2 = tr2.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=3,
    collate_fn=collator
)

test = test.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=3,
    collate_fn=collator
)

d22 = d2.to_tf_dataset(
    columns=["attention_mask", "input_ids", "label"],
    shuffle=False,
    batch_size=3,
    collate_fn=collator
)

batch_size = 3
num_epochs = 3
batches_per_epoch = len(train) // batch_size
total_train_steps = int(batches_per_epoch * num_epochs)
optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

model.compile(optimizer=optimizer)
print(model.summary())

if "bert_base_multilingual_pe.index" not in os.listdir("./final_eval/pretrained/"):
    model.fit(train, epochs=3)
    model.save_weights("./final_eval/pretrained/bert_base_multilingual_pe")

preds_train = model.predict(train2)
preds_test = model.predict(test)
preds_all = model.predict(d22)

probs_train = pd.DataFrame(softmax(preds_train.logits, axis=1))
probs_test = pd.DataFrame(softmax(preds_test.logits, axis=1))
probs_all = pd.DataFrame(softmax(preds_all.logits, axis=1))


def weighted_continous(probs: List[float]):
    score = 0
    probability = 1e-10 # assure that no zero division can occur
    for idx, value in enumerate(probs):
        score += idx * value
        probability += value
    return score / probability

probs_train["weighted_score"] = probs_train.apply(lambda x: weighted_continous(x), axis=1)
probs_test["weighted_score"] = probs_test.apply(lambda x: weighted_continous(x), axis=1)
probs_all["weighted_score"] = probs_all.apply(lambda x: weighted_continous(x), axis=1)

probs_train["actual"] = data[data["train"]==1].label.values
probs_test["actual"] = data[data["train"]==0].label.values
probs_all["actual"] = d2["label"]

print(model.evaluate(test))

actual_test = data[data["train"]==0]["label"].values
predicted_test = np.argmax(preds_test.logits, axis=1)

actual_train = data[data["train"]==1]["label"].values
predicted_train = np.argmax(preds_train.logits, axis=1)


print(confusion_matrix(actual_train, predicted_train))
print(confusion_matrix(actual_test, predicted_test))

print(precision_recall_fscore_support(actual_train, predicted_train, average="macro"))
print(precision_recall_fscore_support(actual_test, predicted_test, average="macro"))

pd.DataFrame(confusion_matrix(actual_train, predicted_train)).to_clipboard()
pd.DataFrame(confusion_matrix(actual_test, predicted_test)).to_clipboard()

print(r2_score(probs_train["actual"],probs_train["weighted_score"]))
print(r2_score(probs_test["actual"],probs_test["weighted_score"]))


print(pearsonr(probs_train["weighted_score"],probs_train["actual"]))
print(pearsonr(probs_test["weighted_score"],probs_test["actual"]))

# add in the old stuff
old = pd.read_csv("./final_eval/data.csv",index_col=0)
old = old.drop(columns=["text","label"])

data[str("predicted_"+construct_type)] = probs_all["weighted_score"].values
data[old.columns] = old
data.to_csv("./final_eval/data_final.csv")