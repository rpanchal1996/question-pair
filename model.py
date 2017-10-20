import numpy as np
import tensorflow as tf
import numpy as np
import json
import datetime
from tensorflow.python.ops import variable_scope as vs
maxSeqLength = 30
number_of_examples_to_take = 100000
global_pair_counter = 0
total_global_index_counter = 0


def verbose(original_function):
	def new_function(*args, **kwargs):
		print('get variable:', '/'.join((tf.get_variable_scope().name, args[0])))
		result = original_function(*args, **kwargs)
		return result
	return new_function

vs.get_variable = verbose(vs.get_variable)

def load_matrices():
	q1_ids = np.load('q1_ids_matrix.npy')
	q2_ids = np.load('q2_ids_matrix.npy')

def load_data_saved():
	with open('stemmed_split_sentences','r') as myfile:
		data = json.load(myfile)
	return data

def load_question_pair():
	global global_pair_counter
	question_one_matrice = np.load('q1_ids_matrix.npy')
	question_two_matrice = np.load('q2_ids_matrix.npy')
	is_same_matrice = np.load('is_same_matrix.npy')
	
	added = 0
	question_one_batches = []
	question_two_batches = []
	is_same_batches = []
	while added<20:
		if np.sum(question_one_matrice[global_pair_counter]) == 0 or np.sum(question_one_matrice[global_pair_counter])==0 or len(is_same_matrice[global_pair_counter])>1:
			global_pair_counter+=1
			error = 1
			question_one = question_one_matrice
			question_two = question_two_matrice
			is_same = is_same_matrice
		else:
			try:
				zero_index = question_one_matrice[global_pair_counter].tolist().index(0)
				question_one = np.roll(question_one_matrice[global_pair_counter],30-zero_index)
			except ValueError:
				question_one = question_one_matrice[global_pair_counter]

			try:
				zero_index = question_two_matrice[global_pair_counter].tolist().index(0)
				question_two = np.roll(question_two_matrice[global_pair_counter],30-zero_index)
			except ValueError:
				question_two = question_two_matrice[global_pair_counter]
			is_same = is_same_matrice[global_pair_counter]
			global_pair_counter+=1
			error = 0
			question_one = question_one.reshape(question_one.shape[0],-1).T
			question_two = question_two.reshape(question_two.shape[0],-1).T
			question_one[question_one==3999999] = 214476 
			question_two[question_two==3999999] = 214476
			question_one_batches.append(question_one.flatten().tolist())
			question_two_batches.append(question_two.flatten().tolist())
			is_same_batches.append(is_same)
			added+=1
	question_one_final = np.array(question_one_batches)
	question_two_final = np.array(question_two_batches)
	is_same_final = np.array(is_same_batches)
	return question_one_final,question_two_final,is_same_final

wordVectors = np.load('word_vectors.npy')
print wordVectors.shape
batchSize = 20
lstmUnits = 64
numClasses = 30
iterations = 100000
numDimensions = 300
learning_rate = 0.0001
tf.reset_default_graph()
keep_prob = 0.75
graph = tf.Graph()
lstm_layers = 3
number_of_epochs = 20
total_number_of_iterations = int(number_of_examples_to_take/batchSize)-1
with graph.as_default():
	#The following function was taken from a StackOverflow answer: https://stackoverflow.com/questions/44769200/how-do-i-share-weights-across-different-rnn-cells-that-feed-in-different-inputs
	def create_lstm_multicell(name,n_layers,nstates):
		def lstm_cell(i, s):
			print('creating cell %i in %s' % (i, s))
			return tf.contrib.rnn.LSTMCell(nstates, reuse=tf.get_variable_scope().reuse)
		lstm_multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(i, name) for i in range(n_layers)])
		return lstm_multi_cell
	
	label = tf.placeholder(tf.int32, [batchSize,1], name='label')
	
	with tf.variable_scope('Inference',reuse=False):
		input_data_q1 = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
		data_q1 = tf.nn.embedding_lookup(wordVectors,input_data_q1)
		data_q1 = tf.cast(data_q1,tf.float32)
		question1_multi_lstm = create_lstm_multicell('lstm1',lstm_layers,lstmUnits)
		q1_initial_state = question1_multi_lstm.zero_state(batchSize, tf.float32)
		question1_outputs, question1_final_state = tf.nn.dynamic_rnn(question1_multi_lstm, data_q1, initial_state=q1_initial_state)
	with tf.variable_scope('Inference',reuse=True) as scope:
		scope.reuse_variables()
		input_data_q2 = tf.placeholder(tf.int32, [batchSize, maxSeqLength])
		data_q2 = tf.nn.embedding_lookup(wordVectors,input_data_q2)
		data_q2 = tf.cast(data_q2,tf.float32)
		question2_multi_lstm = create_lstm_multicell('lstm2',lstm_layers,lstmUnits)
		q2_initial_state = question2_multi_lstm.zero_state(batchSize, tf.float32)
		question2_outputs, question2_final_state = tf.nn.dynamic_rnn(question2_multi_lstm, data_q2, initial_state=q2_initial_state)

	'''
	diff = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(question1_outputs[:, -1, :], question2_outputs[:, -1, :])), reduction_indices=1))
	margin = tf.constant(1.) 
	labels = tf.to_float(label)
	match_loss = tf.expand_dims(tf.square(diff, 'match_term'), 0)
	mismatch_loss = tf.expand_dims(tf.maximum(0., tf.subtract(margin, tf.square(diff)), 'mismatch_term'), 0)
	loss = tf.add(tf.matmul(labels, match_loss), tf.matmul((1 - labels), mismatch_loss), 'loss_add')
	final_loss = tf.reduce_mean(loss)
	'''
	margin = tf.constant(1.) 
	labels = tf.to_float(label)
	
	d = tf.reduce_sum(tf.square(tf.subtract(question1_outputs, question2_outputs)), 1, keep_dims=True)
	d_sqrt = tf.sqrt(d)
	loss = labels * tf.square(tf.maximum(0.0, margin - d_sqrt)) + (1 - labels) * d
	final_loss = tf.reduce_mean(loss)

	
	optimizer = tf.train.AdamOptimizer(learning_rate).minimize(final_loss)
	sess = tf.InteractiveSession()
	
	logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
	loss_summary = tf.summary.scalar('Loss', final_loss)
	merged = tf.summary.merge_all()
	#writer = tf.summary.FileWriter(logdir, sess.graph)
	writer = tf.summary.FileWriter(logdir)
	saver = tf.train.Saver(max_to_keep=1)
	sess.run(tf.global_variables_initializer())
	
	for epoch in xrange(0,number_of_epochs):
		for iteration_number in xrange(0,total_number_of_iterations):
			total_global_index_counter+=1
			question_one,question_two,is_same = load_question_pair()
			loss_obtained = sess.run([final_loss], {input_data_q1: question_one, input_data_q2:question_two,label:is_same})
			if iteration_number%100==0:
				print 'LOSS AT STEP ' + str(iteration_number) + ' IS == ' +str(loss_obtained)
			if iteration_number%25 == 0 and iteration_number !=0:
				summary = sess.run(loss_summary, {input_data_q1: question_one, input_data_q2:question_two,label:is_same})
				writer.add_summary(summary, total_global_index_counter)
				#print 'SAVING TO TENSORBOARD'
			if iteration_number%1000 == 0 and iteration_number !=0:
				save_path = saver.save(sess, "models/siamese.ckpt", global_step=total_global_index_counter)
				print("saved to %s" % save_path)
		global_pair_counter = 0
		print ' EPOCH DONE == ' + str(epoch)
	writer.close()