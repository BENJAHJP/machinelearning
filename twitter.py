import torch
import torch.nn as nn
import torch.nn.functional as F
from spacy.lang.en import STOP_WORDS
from torch.utils.data.dataset import T_co
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import spacy
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained("bert-base-uncased")

nlp = spacy.load('en_core_web_lg')

df = pd.read_csv("/home/b3njah/Downloads/datasets/nlp-getting-started/train.csv")


def data_preparation(dataframe=df):
    dataframe['context'] = dataframe['keyword']+"."+dataframe['text']
    dataframe = dataframe.drop(labels=['keyword', 'text', 'location'], axis=1)
    dataframe = dataframe[['id', 'context', 'target']]
    dataframe = dataframe.dropna()
    return dataframe


df = data_preparation()


def text_normalisation(text):
    doc = nlp(text)
    lemma_list = [str(token.lemma_).lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
    return lemma_list


df['norm'] = df['context'].apply(text_normalisation)


def tokenizer(text):
    encoded_input = tokenizer.encode_plus(text, return_tensors='pt')
    return encoded_input


df['bert_tokens'] = df['norm'].apply(tokenizer)
print(df['bert_tokens'])