## Main study
So far a lot of focus has been on predicting sentiment scores to derive both general attitude towards products but also product improvement opportunities.
In Marketing a common tool for this is the technology acceptance model. This part shows our approach to automatic TAM construct scoring from online customer reviews.

### Approach

For our paper we first asked 3 human experts to label our TAM constructs on a regular 5-point likert scale.
We then randomly split the data into train and test (see [bert_random_sample.py](./bert_random_sample.py)). From this we predict the 6 TAM constructs individually. See also  Fig. 1 
below. We also predict the star rating as a sanity check and to validate, that our chosen model is adequate.

More details on the model can be found in [Rese et al. 2014](https://doi.org/10.1016/j.jretconser.2014.02.011).

![Fig. 1](research%20model.png)

For each construct we follow the following steps:

1. Load the data
2. 

This folder contains the codes for our experiments.

|File|Task|
|------|-|
|[./data/](./data/)| file to store your data in.
|[main.py](./main.py)| Exemplary code to predict the perceived informativeness using Scikit-Ollama and Gemma2:9b.
|Ollama*.py| Alternative approach using the Ollama chat module. Both files use the same concept and arrive at similar results. This approach is simply more verbose.
|[main_gpt.py](./main_gpt.py)| ChatGPT alternative to main.py.
|[gpt.py](./gpt.py)| ChatGPT alternative to Ollama*.py.
|[key.py](./key.py)| File to store your OpenAI key in.
|[prompts](./prompts.py)| File that contains the prompts used in Ollama, gpt, or the main scripts.

We recommend readers to look at the various Ollama files to understand how we make use of things like the structured output.