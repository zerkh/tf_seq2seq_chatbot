"""Utilities for processing data"""
import os
import re
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tf_seq2seq_chatbot.configs.config import *

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]

PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

# Regular expressions used to tokenize.
_WORD_SPLIT = re.compile("([.,!?\"':;)(]) ")
_DIGIT_RE = re.compile(r"\d{3,}")


def basic_tokenizer(sentence):
	"""Very basic tokenizer: split the sentence into a list of tokens."""
	words = []
	for space_sperated_fragment in sentence.strip().split():
		words.extend(re.split(_WORD_SPLIT, space_sperated_fragment))
	return [w.lower() for w in words if w]


def create_vocab(vocab_path, data_path, max_vocab_size, tokenizer=None, normalize_digits=True):
	"""Create vocabulary file (if it does not exist yet) from data file.

	  Data file is assumed to contain one sentence per line. Each sentence is
	  tokenized and digits are normalized (if normalize_digits is set).
	  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
	  We write it to vocabulary_path in a one-token-per-line format, so that later
	  token in the first line gets id=0, second line gets id=1, and so on.

	  Args:
		vocab_path: path where the vocabulary will be created.
		data_path: data file that will be used to create vocabulary.
		max_vocab_size: limit on the size of the created vocabulary.
		tokenizer: a function to use to tokenize each data sentence;
		  if None, basic_tokenizer will be used.
		normalize_digits: Boolean; if true, all digits are replaced by 0s.
	"""

	if not os.path.exists(vocab_path):
		print "Createing vocabulary %s from data %s" % (vocab_path, data_path)
		vocab = {}
		with open(data_path, mode="r") as fin:
			counter = 0
			for line in fin:
				counter += 1
				if counter % 10000 == 0:
					print " Processing line %d" % (counter)
				tokens = tokenizer(
					line) if tokenizer else basic_tokenizer(line)
				for w in tokens:
					word = w
					if word in vocab:
						vocab[word] += 1
					else:
						vocab[word] = 1
			vocab_list = _START_VOCAB + \
				sorted(vocab, key=vocab.get, reverse=True)
			if len(vocab_list) > max_vocab_size:
				vocab_list = vocab_list[:max_vocab_size]
			with open(vocab_path, "w") as fout:
				for w in vocab_list:
					fout.write(w + "\n")

def sentence_to_ids(sentence, vocab_list):
	"""Convert a sentence to tokens.
	args:
		sentence: a sequence of words.
		vocab_list: dictory of vocabulary.
	return:
		a sequence of tokens.
	"""
	ids = []
	words = basic_tokenizer(sentence)

	for word in words:
		ids.append(str(vocab_list.get(word, UNK_ID)))

	return ids

def data_to_ids(ids_path, data_path, vocab_path):
	"""Convert sentences to tokens.
	args:
		ids_path: tokens to save.
		data_path: data directory.
		vocab_path: vocabulary directory.
	"""
	if os.path.exists(ids_path):
		return

	print "Createing ids %s from data %s" % (ids_path, data_path)

	vocab_list = {}
	with open(vocab_path, "r") as fin:
		idx = 0
		for line in fin:
			vocab_list[line.strip()] = idx
			idx += 1

	with open(ids_path, "w") as fout:
		with open(data_path, "r") as fin:
			counter = 0
			for sen in fin:
				counter += 1
				if counter % 10000 == 0:
					print " Processing line %d" % (counter)
				ids = sentence_to_ids(sen, vocab_list)
				fout.write(" ".join(ids) + "\n")

def prepare_dialog_data(train_data_dir, dev_data_dir, vocab_size):
	"""Get dialog data into data_dir, create vocabularies and tokenize data.

	Args:
		data_dir: directory in which the data sets will be stored.
		vocab_size: size of the English vocabulary to create and use.

	Return:
		A tuple of 3 elements:
			(1) path to the token-ids for chat training data-set
			(2) path to the token-ids for chat development data-set.
			(3) path to the chat vocabulary file
	"""

	vocab_path = os.path.join(DATA_DIR, "tmp/vocab%d.in" % vocab_size)
	create_vocab(vocab_path, train_data_dir, vocab_size)

	train_ids_path = os.path.join(DATA_DIR, "tmp/train.ids%d.in" % vocab_size)
	dev_ids_path = os.path.join(DATA_DIR, "tmp/dev.ids%d.in" % vocab_size)
	data_to_ids(train_ids_path, train_data_dir, vocab_path)
	data_to_ids(dev_ids_path, dev_data_dir, vocab_path)

def read_data(data_path, max_size=None):
	"""Convert data which is processed to src and tgt

	Args:
		data_path: path to the files with token-ids.
		max_size: maximum number of lines to read, all other will be ignored

	Return:
		data_set:  a list of length len(_buckets); data_set[n] contains a list of
		(source, target) pairs read from the provided data files that fit
		into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
		len(target) < _buckets[n][1]; source and target are lists of token-ids.
	"""
	data_set = [[] for _ in BUCKETS]

	with open(data_path, "r") as fin:
		last_sen = fin.readline()
		counter = 0
		for sen in fin:
			if counter >= max_size and max_size != None:
				break
			counter += 1
			if counter % 10000 == 0 :
				print "  reading data line %d" % counter
				sys.stdout.flush()
			source_ids = [int(x) for x in last_sen.strip().split()]
			target_ids = [int(x) for x in sen.strip().split()]

			target_ids.append(EOS_ID)

			for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
				if len(source_ids) < source_size and len(target_ids) < target_size:
					data_set.append((source_ids, target_ids))
					break

			last_sen = sen
	return data_set
