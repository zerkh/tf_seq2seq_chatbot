"""Construct seq2seq model"""

class Seq2SeqModel(object):
	"""Sequence-to-sequence model with attention and for multiple buckets."""
	def __init__(self, source_vocab_size, target_vocab_size, buckets, size,\
			num_layers, max_gradient_norm, batch_size, learning_rate,\
			num_samples=512, forward_only=512):
		self.source_vocab_size = source_vocab_size
		self.target_vocab_size = target_vocab_size
		self.buckets = buckets
		self.batch_size = batch_size

		self.learning_rate = tf.Variable(float(learning_rate), trainable=False)
		self.global_step = tf.Variable(0, trainable=False)
