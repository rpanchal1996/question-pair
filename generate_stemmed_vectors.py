# -*- coding: utf-8 -*-
from nltk.stem.porter import PorterStemmer
import json
ps = PorterStemmer()
path_to_vectors = '/home/rudresh/Desktop/quora/numberbatch-en.txt'
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
filename = 'stemmed_vectors'
def read_stemmed(filename):
	with open(filename,'r') as myfile:
		word_vector_dict = json.load(myfile)
	print len(word_vector_dict)
#read_stemmed(filename)