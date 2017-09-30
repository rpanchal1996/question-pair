import numpy as np
import tensorflow as tf
import numpy as np
import json
maxSeqLength = 30
number_of_examples_to_take = 100000
global_pair_counter = 0
def load_matrices():
	q1_ids = np.load('q1_ids_matrix.npy')
	q2_ids = np.load('q2_ids_matrix.npy')
def load_data_saved():
	with open('stemmed_split_sentences','r') as myfile:
		data = json.load(myfile)
	return data
#load_matrices()

wordVectors = np.load('word_vectors.npy')
print wordVectors.shape
batchSize = 1
lstmUnits = 64
numClasses = 30
iterations = 100000
numDimensions = 300
tf.reset_default_graph()

tf.variable_scope('Inference',reuse=False):
	input_data_q1 = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
	data_q1 = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
	data_q1 = tf.nn.embedding_lookup(wordVectors,input_data_q1)
	weights_q1 = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
	bias_q1 = tf.Variable(tf.constant(0.1, shape=[numClasses]))	
	lstmCell_1 = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
	lstmCell_1 = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	value_1, _ = tf.nn.dynamic_rnn(lstmCell_1, data_q1, dtype=tf.float32)
	last_1 = tf.gather(value_1, int(value.get_shape()[0]) - 1)
	prediction_1 = (tf.matmul(last, weights_q1) + bias_q1)
tf.variable_scope('Inference',reuse=True):
	input_data_q2 = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
	data_q2 = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
	data_q2 = tf.nn.embedding_lookup(wordVectors,input_data)
	weights_q1 = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
	bias_q1 = tf.Variable(tf.constant(0.1, shape=[numClasses]))	
	lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
	lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
	value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
	last = tf.gather(value, int(value.get_shape()[0]) - 1)
	prediction_2 = (tf.matmul(last, weights_q1) + bias_q1)


def load_question_pair():
	global global_pair_counter
	question_one_matrice = np.load('q1_ids_matrix.npy')
	question_two_matrice = np.load('q2_ids_matrix.npy')
	is_same_matrice = np.load('is_same_matrix.npy')
	if np.sum(question_one_matrice[global_pair_counter]) == 0 or np.sum(question_one_matrice[global_pair_counter])==0:
		global_pair_counter+=1
		question_one,question_two,is_same = load_question_pair()
		return question_one,question_two,is_same
	else:
		try:
			zero_index = question_one_matrice[global_pair_counter].tolist().index(0)
			question_one = np.roll(question_one_matrice[global_pair_counter],30-zero_index)
		except ValueError:
			question_one = question_one_matrice[global_pair_counter]

		try:
			zero_index = question_two_matrice[global_pair_counter].tolist().index(0)
			question_one = np.roll(question_two_matrice[global_pair_counter],30-zero_index)
		except ValueError:
			question_two = question_two_matrice[global_pair_counter]
		is_same = is_same_matrice[global_pair_counter]
		global_pair_counter+=1
		print global_pair_counter
		return question_one,question_two,is_same
for i in xrange(0,5):
	load_question_pair()