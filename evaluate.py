import numpy as np
import tensorflow as tf
import numpy as np
import json
import datetime

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

def load_question_pair():
	global global_pair_counter
	question_one_matrice = np.load('q1_ids_matrix.npy')
	question_two_matrice = np.load('q2_ids_matrix.npy')
	is_same_matrice = np.load('is_same_matrix.npy')
	if np.sum(question_one_matrice[global_pair_counter]) == 0 or np.sum(question_one_matrice[global_pair_counter])==0:
		global_pair_counter+=1
		error = 1
		question_one = question_one_matrice
		question_two = question_two_matrice
		is_same = is_same_matrice
		return question_one, question_two, is_same, error
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
		#print global_pair_counter
		error = 0
		#question_one = 
		question_one = question_one.reshape(question_one.shape[0],-1).T
		question_two = question_two.reshape(question_two.shape[0],-1).T
		question_one[question_one==3999999] = 214476 
		question_two[question_two==3999999] = 214476
		return question_one,question_two,is_same,error
same = 0
not_same = 0
for i in xrange(0,100000):
	question_one,question_two,is_same,error = load_question_pair()
	if len(is_same) < 2:
		if is_same == 1:
			same+=1
		else:
			not_same+=1
	if i%50 == 0:
		print  'SAME ' +str(same)
		print 'NOT SAME  ' +str(not_same)
		
#print same
#print not_same