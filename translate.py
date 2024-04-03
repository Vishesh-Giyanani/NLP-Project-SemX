import numpy as np 
import pandas as pd 
import requests

import pickle
import tenerflow as TF
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


def decode_sequence(input_seq):
    encoder_model,decoder_model = TF.load_model()
    num_decoder_tokens = 90
    max_decoder_seq_length = 1145
    target_token_index = pickle.load(open('assets/target_token.pkl', 'rb'))
    reverse_target_char_index = pickle.load(open('assets/reverse_target.pkl', 'rb'))
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    target_seq[0, 0, target_token_index['%']] = 1.
    stop_condition = False
    decoded_sentence = ''
    
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)
        
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        if (sampled_char == '$' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        states_value = [h, c]

    return decoded_sentence

def translated_english_to_hindi(input_sentence):
    exclude = set(string.punctuation) # Set of all special characters
    remove_digits = str.maketrans('', '', digits)
    max_encoder_seq_length = 1309
    num_encoder_tokens=27
    input_token_index = pickle.load(open('assets/input_token.pkl', 'rb'))
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

    try:
        decoded_sentence = decode_sequence(encoder_input_data)
    except:
        decoded_sentence = TF.load(input_sentence)
    return decoded_sentence
if __name__=='__main__':
    translated_english_to_hindi('Hi, how are you?')