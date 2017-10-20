# -*- coding: utf-8 -*-
from nltk.stem.porter import PorterStemmer
import json
import re
import sys
ps = PorterStemmer()
#path_to_vectors = '/home/rudresh/Desktop/quora/numberbatch-en.txt'
path_to_vectors = sys.argv[1]
import numpy as np

def dump_stemmed(filepath):
	vectors = []
	with open(filepath,'r') as myfile:
		vectors = myfile.readlines()
		vectors = [vector.strip() for vector in vectors]
	word_vector_dict = {}
	for word_vector in vectors:
		word = word_vector.split()[0].																																																																																																																																																																																																																																																																																																																																																																																																																																																																																																				decode("utf-8").encode('ascii', 'ignore')
		vector = word_vector.split()[1:]
		if '#' not in word and '_' not in word:
			word = ps.stem(word)
			if word in word_vector_dict:
				pass
			else:
				word_vector_dict[word] = vector
	with open('stemmed_vectors','w') as myfile:
		json.dump(word_vector_dict,myfile)

def generate_in_correct_format(filename):
	word_vector_dict = {}
	with open(filename,'r') as myfile:
		word_vector_dict = json.load(myfile)
	for word,vector in word_vector_dict.iteritems():
		vector_string = ' '.join(vector)
		string_to_write = str(word) + ' ' + vector_string + '\n'
		with open('final_vectors','a') as myfile:
			myfile.write(string_to_write)


#generate_in_correct_format(filename)

def clean_vectors():
	ctr = 0
	with open('final_vectors','r') as myfile:
		vectors = myfile.readlines()
	for vector in vectors:
			vector_word = vector.split()[0]
			if re.match('[a-zA-Z]+',vector_word):
				with open('final_clean_vectors','a') as myfile:
					myfile.write(vector)
def generate_word_list(filename_of_vectors):
	with open(filename_of_vectors,'r') as myfile:
		vectors = myfile.readlines()
	for vector in vectors:
		word = vector.split()[0] + '\n'
		with open('wordlist','a') as myfile:
			myfile.write(word)
def create_vector_matrix():
	with open('final_clean_vectors','r') as myfile:
		vectors = myfile.readlines()
		vectors = [vector.strip() for vector in vectors]
	number_of_vectors = len(vectors)
	ids = np.zeros((number_of_vectors, 300), dtype='float32')
	print ids.shape
	final_vectors = []
	for vector in vectors:
		values = vector.split()[1:]
		values = [float(value) for value in values]
		final_vectors.append(values)
	final_numpy_array = np.array(final_vectors)
	print final_numpy_array.shape
	np.save('word_vectors',final_numpy_array)


dump_stemmed(path_to_vectors)
filename = 'stemmed_vectors'
generate_in_correct_format(filename)
clean_vectors()
generate_word_list('final_clean_vectors')
create_vector_matrix()