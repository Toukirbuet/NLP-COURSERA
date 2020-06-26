# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 23:09:07 2020

@author: Real
"""
import wget
#data=wget.download('https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json','sarcasm/')
import tensorflow as tf

tf.config.experimental.list_physical_devices('GPU') 
tf.debugging.set_log_device_placement(True) 
import json
with open(data, 'r') as f:
    datastore = json.load(f)


sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])



from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)