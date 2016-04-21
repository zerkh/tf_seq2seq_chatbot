import tensorflow as tf

DATA_DIR = '/home/kh/Fun/tf_seq2seq_chatbot/'

tf.app.flags.DEFINE_integer('train_data_dir', DATA_DIR+'data/train/movie_lines_cleaned.txt'\
                            , 'Data directory')
tf.app.flags.DEFINE_integer('dev_data_dir', DATA_DIR+'data/train/movie_lines_cleaned_10k.txt'\
                            , 'Data directory')
tf.app.flags.DEFINE_integer('model_dir', DATA_DIR+'models', 'Model Directory')
tf.app.flags.DEFINE_integer('results_dir', DATA_DIR+'results', 'Result directory')

tf.app.flags.DEFINE_float('learning_rate', 0.5, 'Learning rate.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.99, 'Learning rate decays by this much.')
tf.app.flags.DEFINE_float('max_gradient_norm', 5.0, 'Clip gradients to this norm.')
tf.app.flags.DEFINE_integer('batch_size', 128, 'Batch size to use during training.')

tf.app.flags.DEFINE_integer('vocab_size', 80000, 'Dialog vocabulary size.')
tf.app.flags.DEFINE_integer('size', 2048, 'Size of each model layer.')
tf.app.flags.DEFINE_integer('num_layers', 4, 'Number of layers in the model.')

tf.app.flags.DEFINE_integer('max_train_data_size', 0, 'Limit on the size of training data (0: no limit).')
tf.app.flags.DEFINE_integer('steps_per_checkpoint', 100, 'How many training steps to do per checkpoint.')

FLAGS = tf.app.flags.FLAGS

BUCKETS = [(5,10), (10,15), (20,25), (40,50)]