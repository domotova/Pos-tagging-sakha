import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import classification_report

import numpy as np
import collections
import re

import pyconll

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

from utils import tokenize_corpus, build_vocabulary, \
    character_tokenize, pos_corpus_to_tensor, POSTagger, \
    train_eval_loop, predict_with_model, init_random_seed, \
    SentenceLevelPOSTagger, StackedConv1d

init_random_seed(4)

def main():
    # Загрузка данных. Разделение на тренировочные и тестовые
	  full = pyconll.load_from_file('./data/postag_sakha.conllu')
	  full_train = full[:1700]
	  full_test = full[1700:]
	  print('Количество тренировочных предложений = ', len(full_train))
	  print('Количество тестовых предложений = ', len(full_test))

	  # Посчитаем максимальную длину слова и предложения
	  MAX_SENT_LEN = max(len(sent) for sent in full_train)
	  MAX_ORIG_TOKEN_LEN = max(len(token.form) for sent in full_train for token in sent)
	  print('Наибольшая длина предложения', MAX_SENT_LEN)
	  print('Наибольшая длина токена', MAX_ORIG_TOKEN_LEN)

	  all_train_texts = [' '.join(token.form for token in sent) for sent in full_train]

    # Создаем словарь символов
	  train_char_tokenized = tokenize_corpus(all_train_texts, tokenizer=character_tokenize) 
	  char_vocab, word_doc_freq = build_vocabulary(train_char_tokenized, max_doc_freq=1.0, 	min_count=5, pad_word='<PAD>')
	  print("Количество уникальных символов", len(char_vocab))
	  print(list(char_vocab.items())[:10])

	  # Создаем словарь тегов
	  UNIQUE_TAGS = ['<NOTAG>'] + sorted({token.upos for sent in full_train for token in sent if token.upos})
	  label2id = {label: i for i, label in enumerate(UNIQUE_TAGS)}
	  print(label2id)

	  # Преобразование данных в числа
	  train_inputs, train_labels = pos_corpus_to_tensor(full_train, char_vocab, label2id, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)
	  train_dataset = TensorDataset(train_inputs, train_labels)

	  test_inputs, test_labels = pos_corpus_to_tensor(full_test, char_vocab, label2id, MAX_SENT_LEN, MAX_ORIG_TOKEN_LEN)
	  test_dataset = TensorDataset(test_inputs, test_labels)


	  sentence_level_model = SentenceLevelPOSTagger(len(char_vocab), len(label2id), embedding_size=64,
                                              	      single_backbone_kwargs=dict(layers_n=4, kernel_size=5, dropout=0.3),
                                              	      context_backbone_kwargs=dict(layers_n=4, kernel_size=3, dropout=0.3))
	  print('Количество параметров', sum(np.product(t.shape) for t in sentence_level_model.parameters()))

	  (best_val_loss, best_sentence_level_model) = train_eval_loop(sentence_level_model,
                                                     train_dataset,
                                                     test_dataset,
                                                     F.cross_entropy,
                                                     lr=5e-3,
                                                     epoch_n=30,
                                                     batch_size=64,
                                                     device='cuda',
                                                     early_stopping_patience=5,
                                                     max_batches_per_epoch_train=500,
                                                     max_batches_per_epoch_val=100,
                                                     lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2,
                                                                                                                         factor=0.5,
                                                                                                                         verbose=True))

	  torch.save(best_sentence_level_model, './models/cnn_pos')

	  UNIQUE_TAGS1 = ['ADJ','ADV','AUX','CONJ','INTJ','NOUN','NUM','PART','PR','PRON','VERB']

	  from sklearn.metrics import confusion_matrix
	  train_pred = predict_with_model(sentence_level_model, train_dataset)
	  train_loss = F.cross_entropy(torch.tensor(train_pred), torch.tensor(train_labels))
	  print('Среднее значение функции потерь на обучении', float(train_loss))
	  print(classification_report(train_labels.view(-1), train_pred.argmax(1).reshape(-1),  labels=[1,2,3,4,5,6,7,8,9,10,13], target_names=UNIQUE_TAGS1))
	  print()

	  test_pred = predict_with_model(sentence_level_model, test_dataset)
	  test_loss = F.cross_entropy(torch.tensor(test_pred), torch.tensor(test_labels))
	  print('Среднее значение функции потерь на валидации', float(test_loss))
	  print(classification_report(test_labels.view(-1), test_pred.argmax(1).reshape(-1),  labels=[1,2,3,4,5,6,7,8,9,10,13], target_names=UNIQUE_TAGS1))

if __name__ == '__main__':
	  main()

