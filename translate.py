import numpy as np 
import pandas as pd 
import requests
from tenerflow import query

import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense
from tensorflow.keras.optimizers import Adam, RMSprop
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

import re
import string
from string import digits

import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize

import matplotlib.pyplot as plt
import os


def translated_english_to_hindi(input_sentence):
    input_sentence = input_sentence.lower()
    input_sentence = re.sub("'", '', input_sentence)
    input_sentence = ''.join(ch for ch in input_sentence if ch not in exclude)
    input_sentence = input_sentence.translate(remove_digits)
    input_sentence = input_sentence.strip()
    input_sentence = re.sub(" +", " ", input_sentence)

    encoder_input_data = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
    for t, char in enumerate(input_sentence):
        if re.findall("[a-zA-Z]", char) or char == ' ':
            encoder_input_data[0, t, input_token_index[char]] = 1
    encoder_input_data[0, t+1:, input_token_index[' ']] = 1

    decoded_sentence = decode_sequence(encoder_input_data)
    return decoded_sentence


if __name__=='__main__':
    translated_english_to_hindi('Hi, how are you?')