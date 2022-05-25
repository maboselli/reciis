
# Structured for fkfd_fit_040.py

import pandas as pd
import numpy as np

import gc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback
from keras.callbacks import CSVLogger

vocab_size = 5000
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
max_length = 500


model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
#model.add(tf.keras.layers.LSTM(embedding_dim))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)
))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.load_weights('fkfd_weigts.h5')

# reading data to analise 

data = 'web_lupa_2021-03-01.csv'
df = pd.read_csv(data)

len(df)
df = df.dropna()
len(df)

# pre-processing the text elements

import re
import nltk

from nltk.corpus import stopwords
from string import punctuation

# this cleaner is note so importante here, as it is now.
def cleaner(texto):
    #texto = nltk.word_tokenize(texto)
    texto = re.sub(r"\\n", "", texto)
    texto = re.sub(r"\\xa0", '', texto)
    texto = re.sub(r'\(.*?\)', ' ', texto)
    texto = re.sub(r'nº', '', texto)
    texto = re.sub("\d+", "", texto)
    texto = re.sub("\.", "", texto)
    texto = re.sub(",", "", texto)
    texto = re.sub("\'", "", texto)
    texto = re.sub(r'"', "", texto)
    texto = re.sub(r' /', "", texto)
    texto = re.sub(r'―', ' ', texto)
    texto = re.sub(' +', ' ', texto)
    return texto


# Not working
#stopwords = set(stopwords.words('portuguese_tw') + list(punctuation))
#df["Texto"] = df["Texto"].apply(lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))


def remove_stop_words(sentence):
    stop = stopwords.words('portuguese_new')
    stop.append(list(punctuation))
    word_list = nltk.word_tokenize(sentence)
    clean_sentence=' '.join([w for w in word_list if w.lower() not in stop])
    return(clean_sentence)

df["Texto"] = df["Texto"].apply(remove_stop_words)
df["Texto"] = df["Texto"].apply(cleaner)

df["Key"] = df["Key"].replace('FALSO', 0)
df["Key"] = df["Key"].replace('VERDADEIRO', 1)

target_text = df["Texto"].tolist()
target_key = df["Key"].tolist()

csv_path = '/home/boselli/Marco/prog/python/fake_finder/Fake.br-Corpus-master/preprocessed/'

df_ext = pd.read_csv(csv_path + 'pre-processed.csv', usecols = ['label', 'preprocessed_news'])
df_ext = df_ext.replace('fake', 0)
df_ext = df_ext.replace('true', 1)

df_ext = df_ext.sample(frac=1)

cut = int(len(df_ext)*0.8)
tot = len(df_ext)

train = df_ext[:cut]
test = df_ext[cut:]

train_sentences = train['preprocessed_news'].tolist()


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index


targeting_text = tokenizer.texts_to_sequences(target_text)
padded = pad_sequences(targeting_text, maxlen=max_length)
pred = model.predict(padded)

print(pred)

def convert_bin(x):
    a = []
    for i in x:
        if i >= 0.5:
            a.append(1)
        else:
            a.append(0)
    return a

pred_bin = convert_bin(pred)

count = 0
for i in pred_bin:
    if i == 1:
        count += 1


print(count)
print(count/len(pred_bin))



df['predict'] = pred

df.to_csv(f'P{data}')




