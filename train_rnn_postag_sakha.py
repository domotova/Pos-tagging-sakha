import warnings
warnings.filterwarnings("ignore")

import numpy as np
import argparse

from matplotlib import pyplot as plt
import seaborn as sns

from gensim.models import KeyedVectors

from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dense, Input
from keras.layers import TimeDistributed
from keras.layers import LSTM, GRU, Bidirectional, SimpleRNN, RNN
from keras.models import Model
from keras.preprocessing.text import Tokenizer

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import pyconll

def main():
	data = './data/postag_sakha.conllu'
	full = pyconll.load_from_file(data)

	X = []
	Y = []
	for sentence in full:
    		x = []
    		y = []
    		for token in sentence:
        		x.append(token.form)
        		y.append(token.upos)
    		X.append(x)
    		Y.append(y)

	test_sentences = []
	with open("./data/test.txt", "r") as f:
		for line in f:
			test_sentences.append(line)

	for sentence in test_sentences:
    		x = []
    		for token in sentence.split():
        		x.append(token)
    		X.append(x)

	num_words = len(set([word.lower() for sentence in X for word in sentence]))
	num_tags   = len(set([word.lower() for sentence in Y for word in sentence]))

	print("Общее количество предложений: {}".format(len(X)))
	print("Общее количество словаря: {}".format(num_words))
	print("Общее количество тегов: {}".format(num_tags))

	# encode X

	word_tokenizer = Tokenizer()                      
	word_tokenizer.fit_on_texts(X)                   
	X_encoded = word_tokenizer.texts_to_sequences(X)  

	# encode Y

	tag_tokenizer = Tokenizer()
	tag_tokenizer.fit_on_texts(Y)
	Y_encoded = tag_tokenizer.texts_to_sequences(Y)

	# look at first encoded data point

	print("Пример входных данных", "\n")
	print('X: ', X[0], '\n')
	print('Y: ', Y[0], '\n')

	MAX_SEQ_LENGTH = 80  

	X_padded = pad_sequences(X_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")
	Y_padded = pad_sequences(Y_encoded, maxlen=MAX_SEQ_LENGTH, padding="pre", truncating="post")

	print("Пример преобразованных входных данных", "\n")
	print('X: ', X_padded[0], '\n')
	print('Y: ', Y_padded[0], '\n')


	# assign padded sequences to X and Y
	X, Y = X_padded, Y_padded

	np.save('test_sents', X[2380:])

	word2vec = KeyedVectors.load("./word2vec/word2vec_sakha", mmap='r')

	EMBEDDING_SIZE  = 300  # each word in word2vec model is represented using a 300 dimensional vector
	VOCABULARY_SIZE = len(word_tokenizer.word_index) + 1

	# create an empty embedding matix
	embedding_weights = np.zeros((VOCABULARY_SIZE, EMBEDDING_SIZE))

	# create a word to index dictionary mapping
	word2id = word_tokenizer.word_index

	# copy vectors from word2vec model to the words present in corpus
	for word, index in word2id.items():
    		try:
        		embedding_weights[index, :] = word2vec[word]
    		except KeyError:
        		pass

	# use Keras' to_categorical function to one-hot encode Y
	Y = to_categorical(Y)

	X_train = X[:1700]
	X_test = X[1700:2380]
	Y_train = Y[:1700]
	Y_test = Y[1700:]

	# split training data into training and validation sets
	VALID_SIZE = 0.15
	X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=VALID_SIZE, random_state=4)

	# print number of samples in each set
	print("TRAINING DATA")
	print('Shape of input sequences: {}'.format(X_train.shape))
	print('Shape of output sequences: {}'.format(Y_train.shape))
	print("-"*50)
	print("VALIDATION DATA")
	print('Shape of input sequences: {}'.format(X_validation.shape))
	print('Shape of output sequences: {}'.format(Y_validation.shape))
	print("-"*50)
	print("TESTING DATA")
	print('Shape of input sequences: {}'.format(X_test.shape))
	print('Shape of output sequences: {}'.format(Y_test.shape))

	NUM_CLASSES = Y.shape[2]

	bidirect_model = Sequential()
	bidirect_model.add(Embedding(input_dim     = VOCABULARY_SIZE,
                             output_dim    = EMBEDDING_SIZE,
                             input_length  = MAX_SEQ_LENGTH,
                             weights       = [embedding_weights],
                             trainable     = True
	))
	bidirect_model.add(Bidirectional(LSTM(64, return_sequences=True)))
	bidirect_model.add(TimeDistributed(Dense(NUM_CLASSES, activation='softmax')))

	bidirect_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

	# check summary of model
	bidirect_model.summary()

	bidirect_training = bidirect_model.fit(X_train, Y_train, batch_size=128, epochs=25, validation_data=(X_validation, Y_validation))

	"""Model evaluation"""

	loss, accuracy = bidirect_model.evaluate(X_test, Y_test, verbose = 1)
	print("Loss: {0},\nAccuracy: {1}".format(loss, accuracy))

	"""анализ результата"""

	from sklearn.metrics import classification_report
	UNIQUE_TAGS1 =['NOUN','VERB','ADJ','PRON','ADV','NUM','PART','CONJ','PR','AUX','INTJ']
	test_labels = Y_padded[1700:]

	test_pred = bidirect_model.predict_classes(X_test)
	print(classification_report(test_labels.reshape(-1), test_pred.reshape(-1),  labels=[1,2,4,5,6,8,9,10,11,12,13], target_names=UNIQUE_TAGS1))

	bidirect_model.save('./models/rnn_pos')


if __name__ == '__main__':
	  main()