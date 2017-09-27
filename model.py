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
'''
wordVectors = np.load('word_vectors.npy')
print wordVectors.shape
batchSize = 1
lstmUnits = 64
numClasses = 30
iterations = 100000
numDimensions = 300
tf.reset_default_graph()
input_data = tf.placeholder(tf.int32,[batchSize, maxSeqLength])
labels = tf.placeholder(tf.float32, [batchSize, numClasses])
data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.nn.embedding_lookup(wordVectors,input_data)
lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)
weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)
'''
def load_question_pair():
	global global_pair_counter
	question_one_matrice = np.load('q1_ids_matrix.npy')
	question_two_matrice = np.load('q2_ids_matrix.npy')
	is_same_matrice = np.load('is_same_matrix.npy')
	if np.sum(question_one_matrice[global_pair_counter]) == 0 or np.sum(question_one_matrice[global_pair_counter])==0:
		global_pair_counter+=1
	else:
		print question_one_matrice[global_pair_counter]
		zero_index = question_one_matrice[global_pair_counter].tolist().index(0)
		print np.roll(question_one_matrice[global_pair_counter],30-zero_index)
		global_pair_counter+=1
		print global_pair_counter
for i in xrange(0,5):
	load_question_pair()