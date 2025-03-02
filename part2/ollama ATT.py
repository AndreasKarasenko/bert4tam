import json
import pandas as pd
from tqdm import tqdm
from ollama import chat
from pydantic import BaseModel


class ResponseModel(BaseModel):
    label: int

data = pd.read_excel("./data/Ikea1.xlsx")

base_message = """The construct “attitude toward using” reflects the following five items when evaluating an app on 5-point Likert scales:
1. I am positive about the app.
2. The app is so interesting that you just want to learn more about it.
3. It just makes sense to use the app.
4. The use of the app is a good idea.
5. Other people should also use the app.

Available responses to the items and the construct are “[1,2,3,4,5]” on 5-point Likert scales.
These correspond to “1 (strongly disagree)”, “2 (disagree)”, “3 (don’t know, neutral)”, “4 (agree)”, and “5 (disagree)

For the following comment: How would the author evaluate the construct “attitute toward using”?

Review: ```{x}```
Provide your response in a JSON format containing a single key `label`
Do not provide any other information."""

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def process_row(i, row):
    response = chat(
        model="gemma2:9b",
        messages=[
            {
                "role": "user",
                "content": base_message.format(x=row["content"]),
            },
        ],
        format=ResponseModel.model_json_schema(),
    )
    return i, json.loads(response["message"]["content"])["label"]

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(process_row, i, row) for i, row in data.iterrows()]
    for future in tqdm(as_completed(futures), total=len(futures)):
        i, label = future.result()
        data.loc[i, "PI_gemma2"] = label


data.to_csv("./data/Ikea1_ATT_gemma2_v2.csv", index=False)
