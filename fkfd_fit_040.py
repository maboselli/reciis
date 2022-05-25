
# numpy == 1.19.5 for compatibility to tf

import re
import pandas as pd
import statistics
import numpy as np

csv_path = 'Fake.br-Corpus-master/preprocessed/'

df_ext = pd.read_csv(csv_path + 'pre-processed.csv', usecols = ['label', 'preprocessed_news'])
df_ext = df_ext.replace('fake', 0)
df_ext = df_ext.replace('true', 1)


# Corpus data analysis

list(df_ext)

import gc
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.backend import clear_session
from tensorflow.keras.callbacks import Callback
from keras.callbacks import CSVLogger


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        clear_session()

csv_logger = CSVLogger('log.csv', append=True, separator=';')

#locate the first true entry.
idx = df_ext.label.ne(0).idxmax()

df1 = df_ext.sample(frac=1)

cut = int(len(df1)*0.8)
tot = len(df1)

train = df1[:cut]
test = df1[cut:]

train_sentences = train['preprocessed_news'].tolist()
test_sentences = test['preprocessed_news'].tolist()
train_index = train['label'].tolist()
test_index = test['label'].tolist()

vocab_size = 5000
embedding_dim = 16
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
max_length = 500


tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index


training_sequences = tokenizer.texts_to_sequences(train_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type)

testing_sequences = tokenizer.texts_to_sequences(test_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)



model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length))
#model.add(tf.keras.layers.LSTM(embedding_dim))
model.add(tf.keras.layers.Bidirectional(
    tf.keras.layers.LSTM(64, dropout=0.3, recurrent_dropout=0.3)
))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


cb = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

#data_model = model.fit(training_padded, train['label'], epochs=50, validation_data=(testing_padded, test['label']), callbacks=[cb])
# o notebook esquentou demais, diminuindo o numero de iteracos de 50 para 10

data_model = model.fit(training_padded, train['label'], epochs=50, validation_data=(testing_padded, test['label']), callbacks=[csv_logger, cb, ClearMemory()])
model.save('fkfd_model_saved')
model.save_weights('fkfd_weigts.h5')



