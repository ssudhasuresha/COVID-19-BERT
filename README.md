# COVID-19-BERT

The application evaluate the different configuration of BERT on creating a language representation for COVID-19 research papers to aid in answering research questions.

For this project, we are using two BERT models; a. BERT-Base b. SciBert. 

# Data 

## 1.1 Data Source and Format

The data source is collected from COVID-19 kaggel(https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge). The corpus contains ~51,000 scholarly articles, each of which are stored in JSON format (at least at the time, 04/05/2020). The total size of the dataset amounts to 6GB. The files contain semantic metadata such as section names (abstract, introduction, etc), the title, authors, and body text which can be useful in classifying the text. For our project, we are focuing solely on body text.

## 1.2 Packages

To analyze this data, we are using Python3 with the following packages:

1.	Google BERT
2.	SciBERT
3.	Transformers


# Approach

The project is implemented using an unsupervised learning algorithm since the data set is lacking the labels. We have used the Masked Language Model approach because it does not use labels to learn the corpus. Using a Masked Language Model will also let us know if the model can properly learn about the novel coronavirus.

We are comparing the performance of different configurations of the BERT masked language model to the SciBERT masked language model. Here we have used scibert-scivocab-uncased model. 

# Implementation and Evaluation Metrics

We created BERT masked language model using the BERT GitHub(https://github.com/google-research/bert) created by Google and SciBERT masked language model using the SciBERT GitHub(https://github.com/allenai/scibert#pytorch-huggingface-models) created by the huggingface. In our different BERT models, we used SentencePiece to generate the vocabulary. We pre-trained Bert from scratch on the COVID-19 data set by varying the number of training steps, training batch size, and learning rate. We also trained BERT by varying the proportion of data allocated to the testing, development, and training dataset. 

## 1.1 Pretraining Data

We created two pretraining data configuration for Bert model and according to MLM probability, each data set's words are masked.

Configuration 1: a vocab of 32,000 words, MLM probability of 15%, and 10% of data in the training set, 5% in the development set, and 85% in the testing set.

Configuration 2: a vocab of 32,000 words, MLM probability of 15%, and 50% of data in the training set, 10% in the development set, and 40% in the testing set. 

## 1.2 Experiments

1. Training Steps - We did a couple of experiments on the BERT model to evaluate the performance of the model by varying training steps of 100, 1,000, 10,000 and 100, 000.

2. Learning Rate - We varied lower and higher learning rate: 1e-05 and 4e-05 to evaluate the effect of learning rate on the masked language model performance. 

3. Train Batch Size -  We varied lower and higher learning rate: 8, 16, 32, 64, 128, and 256 to evaluate the effect of learning rate on the masked language model performance.

## 1.3 Evaluating BERT and SciBERT

From the experiments , we found that for pretraining configuration 1 the best model we found was with learning rate 2e-05, 128 training batch size, and 1,000 training steps. we evaluated the BERT model with the above-mentioned parameters on the testing set, which contains 85% of the COVID-19 corpus, we got an MLM accuracy of 16% and Next Sentence accuracy of 69%.

For our SciBERT model, we decided to use the scibert-scivocab-uncased model. This is analogous to our BERT- Base COVID-19 model, which also uses its vocab and uncased. We had to create new pretraining data for SciBERT because the vocab we generated is different from the SciBERT vocab. We used 40% of the COVID-19 dataset and masked words with an MLM probability of 15% for the evaluation of SciBERT. We ran the SciBERT on the same configuration which we used for our BERT- Base COVID-19 model. SciBERT produced 66% of MLM accuracy and 86.25% Next Sentence accuracy. 

## 1.4 Masked Word Prediction

Our second goal of the project was to evaluate which BERT model could well predict the masked tokens: BERT- Base COVID-19 model or SciBERT. For this we supplied tokens masked during the pretraining process. To evaluate the prediction, we masked specific words in sentences, which are taken from the COVID-19 article. Then we passed these masked sentences to our BERT-Base COVID-19 model and SciBERT to predict the masked word. We found that SciBERT consistently outperformed in predicting the masked word than our BERT-Base COVID-19 model. SciBERT was able to predict the true words and words that make sense.

For example, when we passed this sentence to both the models:  
"[CLS] severe acute respiratory syndrome coronavirus covid19, cause of the potentially deadly atypical [MASK], infects many organs, such as lung, liver. [SEP]"

SciBERT: **‘pneumonia’** and the next most probable words predicted like **‘syndrome’, ‘disease’**. 

Our BERT model: **‘study’, ‘identified’**


