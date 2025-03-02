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
2. Select the relevant construct for train and test
3. Define the BERT model from the NLPTown checkpoint using 5 output labels
4. Preprocess the data using the NLPTown tokenizer (do NOT remove stopwords etc.)
5. Train the model for 3 epochs using a low learning rate (2e-5)
6. Save the model weights to final_eval/pretrained/...
7. Predict the test labels
8. Evaluate the performance
9. Save all predictions under final_eval

This folder contains the codes for our experiments.

|File|Task|
|------|-|
|[./final_eval/](./final_eval/)| file to store your data in and to store fine-tuned models in once they are trained.
|[./final_eval/pretrained](./final_eval/pretrained/)| file to store fine-tuned models in.
|bert_base_multilingual_*.py| Script for each construct (and star rating) to train and predict a machine learning model.
|[bert_random_sample.py](./bert_random_sample.py)| script to sample the data ONCE randomly.
|[naive_Bayes_IKEA_gesamt.r](./naive_Bayes_IKEA_gesamt.r)| Naive bayes approach using R.

For the results we refer to the [README](../README.md) on the main page.