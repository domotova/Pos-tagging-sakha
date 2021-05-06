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
    train_eval_loop, predict_with_model, init_random_seed

init_random_seed(4)

# Вспомогательная сверточная структура
class StackedConv1d(nn.Module):
    	def __init__(self, features_num, layers_n=1, kernel_size=3, conv_layer=nn.Conv1d, dropout=0.0):
        	super().__init__()
        	layers = []
        	for _ in range(layers_n):
            		layers.append(nn.Sequential(conv_layer(features_num, features_num, kernel_size, padding=kernel_size//2),
                		      nn.Dropout(dropout),
                		      nn.LeakyReLU()))
        	self.layers = nn.ModuleList(layers)
    
    	def forward(self, x):
        	"""x - BatchSize x FeaturesNum x SequenceLen"""
        	for layer in self.layers:
            		x = x + layer(x)
        	return x

class SentenceLevelPOSTagger(nn.Module):
    	def __init__(self, vocab_size, labels_num, embedding_size=32, 
                     single_backbone_kwargs={}, context_backbone_kwargs={}):
          	super().__init__()
        	self.embedding_size = embedding_size
        	self.char_embeddings = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
        	self.single_token_backbone = StackedConv1d(embedding_size, **single_backbone_kwargs)
        	self.context_backbone = StackedConv1d(embedding_size, **context_backbone_kwargs)
        	self.global_pooling = nn.AdaptiveMaxPool1d(1)
        	self.out = nn.Conv1d(embedding_size, labels_num, 1)
        	self.labels_num = labels_num
    
    	def forward(self, tokens):
        	"""tokens - BatchSize x MaxSentenceLen x MaxTokenLen"""
        	batch_size, max_sent_len, max_token_len = tokens.shape
        	tokens_flat = tokens.view(batch_size * max_sent_len, max_token_len)
        
        	char_embeddings = self.char_embeddings(tokens_flat)  # BatchSize*MaxSentenceLen x MaxTokenLen x EmbSize
        	char_embeddings = char_embeddings.permute(0, 2, 1)  # BatchSize*MaxSentenceLen x EmbSize x MaxTokenLen
        	char_features = self.single_token_backbone(char_embeddings)
        
        	token_features_flat = self.global_pooling(char_features).squeeze(-1)  # BatchSize*MaxSentenceLen x EmbSize

        	token_features = token_features_flat.view(batch_size, max_sent_len, self.embedding_size)  # BatchSize x MaxSentenceLen x EmbSize
        	token_features = token_features.permute(0, 2, 1)  # BatchSize x EmbSize x MaxSentenceLen
        	context_features = self.context_backbone(token_features)  # BatchSize x EmbSize x MaxSentenceLen

        	logits = self.out(context_features)  # BatchSize x LabelsNum x MaxSentenceLen
       		return logits

def main():

	# Загрузка данных. Разделение на тренировочные и тестовые
	full = pyconll.load_from_file('./data/postag_sakha.conllu')
	full_train = full[:1700]
	full_test = full[1700:]
        print('Количество тренировочных предложений = ',len(full_train))
	print('Количество тестовых предложений = ',len(full_test))

	# Посчитаем максимальную длину слова и предложения
	MAX_SENT_LEN = max(len(sent) for sent in full_train)
	MAX_ORIG_TOKEN_LEN = max(len(token.form) for sent in full_train for token in sent)
	print('Наибольшая длина предложения', MAX_SENT_LEN)
	print('Наибольшая длина токена', MAX_ORIG_TOKEN_LEN)

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
                                              	      single_backbone_kwargs=dict(layers_n=5, kernel_size=5, dropout=0.3),
                                              	      context_backbone_kwargs=dict(layers_n=5, kernel_size=5, dropout=0.3))
	print('Количество параметров', sum(np.product(t.shape) for t in sentence_level_model.parameters()))

	(best_val_loss, best_sentence_level_model) = train_eval_loop(sentence_level_model,
                                                     train_dataset,
                                                     test_dataset,
                                                     F.cross_entropy,
                                                     lr=5e-3,
                                                     epoch_n=50,
                                                     batch_size=64,
                                                     device='cuda',
                                                     early_stopping_patience=5,
                                                     max_batches_per_epoch_train=500,
                                                     max_batches_per_epoch_val=100,
                                                     lr_scheduler_ctor=lambda optim: torch.optim.lr_scheduler.ReduceLROnPlateau(optim, patience=2,
                                                                                                                         factor=0.5,
                                                                                                                         verbose=True))

	torch.save(best_sentence_level_model.state_dict(), './models/sentence_level_pos.pth')

if __name__ == '__main__':
	main()

