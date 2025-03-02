## Synthetic data generation
As some researchers have shown (e.g. [Arora et al. 2024](https://journals.sagepub.com/doi/abs/10.1177/00222429241276529)) large language models can be used to replicate human user responses for quantitative and qualitative surveys. This allows for rapid survey development and more versatile studies.

For our paper we first asked 3 human experts to label our TAM constructs on a regular 5-point likert scale.
We then additionally asked ChatGPT-4o in a zero-shot manner to label the reviews as well.
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

For the results we refer to the [README](../README.md) on the main page.