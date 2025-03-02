# import libraries
import pandas as pd
from skollama.classification import ZeroShotOllamaClassifier
from skllm.classification import ZeroShotGPTClassifier
from prompts import PI # our custom prompt for perceived informativeness

# load data
data = pd.read_excel("./data/Ikea1.xlsx")

X = data["content"]

# initialize the model
clf = ZeroShotOllamaClassifier(model="gemma2:9b")

clf.fit(None, [1,2,3,4,5])
clf.prompt_template = PI

# predict
y_pred = clf.predict(X, num_workers=4) # took 1:08:30 to run
data["PI_gemma2"] = y_pred
data.to_csv("./data/Ikea1_PI_gemma2.csv", index=False)