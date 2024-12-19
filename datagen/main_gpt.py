# import libraries
import pandas as pd
from skllm.classification import ZeroShotGPTClassifier
from datagen.key import key
from datagen.prompts import PI # our custom prompt for perceived informativeness
# load data
data = pd.read_excel("./data/Ikea1.xlsx")

X = data["content"]

# initialize the model
clf = ZeroShotGPTClassifier(model="gpt-4o", key=key)

clf.fit(None, [1,2,3,4,5])
clf.prompt_template = PI

# predict
y_pred = clf.predict(X, num_workers=1) # took 1:21:30 to run
data["PI_gpt4o"] = y_pred
data.to_csv("./data/Ikea1_PI_gpt4o.csv", index=False)