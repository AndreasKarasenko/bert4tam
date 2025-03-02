import os

import pandas as pd
import numpy as np


os.chdir("./models/karasenko/")
df = pd.read_excel("./final_eval/playstore_ikea_gesamt3.xlsx", index_col=0)

train_indices = np.random.rand(len(df)) < 0.8
train = df[train_indices]
test = df[~train_indices]
print(len(train)/len(df))

df["train"] = train_indices
df["train"] = df["train"].astype("int")

df.to_csv("final_base.csv")
df.to_excel("final_base.xlsx")