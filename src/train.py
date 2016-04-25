import tensorflow as tf
from tf_seq2seq_chatbot.src.data_utils import prepare_dialog_data, read_data
from tf_seq2seq_chatbot.configs.config import FLAGS, DATA_DIR, BUCKETS
from tf_seq2seq_chatbot.src.seq2seq_model import Seq2SeqModel
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_model(...):
	pass

def train():
	print "Prepare the data for training and development..."
	prepare_dialog_data(FLAGS.train_data_dir, FLAGS.dev_data_dir, FLAGS.vocab_size)

	tmp_dir = DATA_DIR + "tmp/"
	train_data = read_data(tmp_dir+"train.ids%d.in" %(FLAGS.vocab_size))
	dev_data = read_data(tmp_dir+"dev.ids%d.in" %(FLAGS.vocab_size))

	model = create_model(...)
